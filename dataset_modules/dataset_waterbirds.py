"""
Waterbirds loader for concept-bottleneck / meta-feature models.

CSV layouts
───────────────────────────────────────
  1. waterbirds_train.csv / waterbirds_validation.csv / waterbirds_test.csv
        image_id, label (0=landbird / 1=waterbird), place (0=land / 1=water)
        [optional: split, y, img_filename, and any `family::value` concept
         columns]
  2. the upstream single-file layout: metadata.csv with a numeric `split`
     column (0=train, 1=valid, 2=test), `y`, `place`, `img_filename`.

summary:
    image_id   str    immutable sample id  (id_i)
    filepath   str    path to the image on disk
    target     int    task label y_i        (0 = landbird, 1 = waterbird)
    place      int    background            (0 = land, 1 = water)
    subgroup   int    g = 2*y + place       (s_i, the canonical group label)
    <concepts> float  a_{i,k} in [0, 1]     when use_meta=True
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── constants ────────────────────────────────────────────────────────────
PLACE_LAND, PLACE_WATER = 0, 1
LABEL_LANDBIRD, LABEL_WATERBIRD = 0, 1

CONCEPT_SEP = '::'
PLACE_CONCEPTS = ['background::land', 'background::water']
PLACE_CONCEPT_SINGLE = ['background::water']

_SPLIT_FILES = {
    'train': 'waterbirds_train.csv',
    'valid': 'waterbirds_validation.csv',
    'test': 'waterbirds_test.csv',
}
_SINGLE_FILES = ['waterbirds.csv', 'metadata.csv', 'waterbirds_metadata.csv']

# Upstream column names -> the names this module uses.
_RENAME = {'label': 'target', 'y': 'target', 'img_filename': 'image_id'}

# Columns that are bookkeeping rather than concepts.
_NON_ATTR_COLS = {
    'target', 'image_id', 'filepath', 'place', 'subgroup', 'group',
    'split', 'index', 'Unnamed: 0', 'y', 'label', 'img_filename', 'bird',
    'image_name', 'class', 'class_label',
}

# g = 2 * target + place — the canonical Waterbirds group index.
GROUP_NAMES = {
    0: 'landbird_on_land',     # majority
    1: 'landbird_on_water',    # minority
    2: 'waterbird_on_land',    # minority
    3: 'waterbird_on_water',   # majority
}
N_GROUPS = 4

# The positive class for AUC / the binary task metrics.
MEL_IDX = LABEL_WATERBIRD

# ══════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════
def _load_image(path: str) -> np.ndarray:
    """cv2.imread returns None on a missing/corrupt file — fail loudly."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _to_int(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if s.isna().any():
        raise ValueError(f'Non-numeric / missing values in column {name!r}')
    return s.astype(np.int64)


def _to_bit(series: pd.Series) -> pd.Series:
    """Concept columns are 0/1; convert defensively (bool / str / +-1)."""
    if series.dtype == bool or str(series.dtype) == 'bool':
        return series.astype(np.int8)
    if series.dtype == object:
        return series.map(
            lambda v: 1 if str(v).strip().lower() in
            ('1', '1.0', 'true', 't', 'yes', 'y') else 0).astype(np.int8)
    return (pd.to_numeric(series, errors='coerce').fillna(0) > 0).astype(np.int8)


def _strip_ext(x) -> str:
    """'a/b/Bird_0046_18.jpg' -> 'a/b/Bird_0046_18'.

    `row.image_id[:-4]` (the version this replaces) silently corrupts the id
    for any extension that is not exactly 3 characters ('.jpeg') and raises
    TypeError when image_id parses as non-str.
    """
    return os.path.splitext(str(x))[0]


def group_index(target, place) -> np.ndarray:
    """g = 2*y + place, as int64. Works on scalars, Series and arrays."""
    t = np.asarray(target, dtype=np.int64)
    p = np.asarray(place, dtype=np.int64)
    return (2 * t + p).astype(np.int64)


def group_name(g: int) -> str:
    return GROUP_NAMES.get(int(g), str(int(g)))


# ══════════════════════════════════════════════════════════════════
#  Dataset (parity with the other modules; the training scripts use
#  data_loaders.BaselineDataset, which is dataset-agnostic)
# ══════════════════════════════════════════════════════════════════
class WaterBirds_Dataset(Dataset):
    def __init__(self, csv, mode, meta_features, dataset, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None and len(meta_features) > 0
        self.meta_features = meta_features
        self.desc_cols = meta_features
        self.transform = transform
        self.dataset = dataset
        self.desc_tau_dict = None

        # Materialise once as contiguous numpy. `self.csv.loc[index, cols]`
        # inside __getitem__ is ~100x slower and dominates the dataloader.
        self.image_ids = np.array([_strip_ext(x) for x in self.csv['image_id']])
        self.filepaths = self.csv['filepath'].to_numpy()
        self.targets = self.csv['target'].to_numpy(dtype=np.int64)
        self.groups = (self.csv['subgroup'].to_numpy(dtype=np.int64)
                       if 'subgroup' in self.csv.columns else None)
        if self.use_meta:
            self.desc = self.csv[self.desc_cols].to_numpy(dtype=np.float32)
            assert np.isfinite(self.desc).all(), 'NaN in concept matrix'
        else:
            self.desc = None

        print(f'[Waterbirds] mode: {self.mode}  n: {len(self)}  '
              f'use_meta: {self.use_meta}  '
              f'n_meta: {0 if self.desc is None else self.desc.shape[1]}')

    def __len__(self):
        return self.csv.shape[0]

    def copy(self):
        return self.csv.copy()

    def group_counts(self) -> dict:
        """Group sizes, for worst-group accuracy / reweighting."""
        if self.groups is None:
            return {}
        u, c = np.unique(self.groups, return_counts=True)
        return {group_name(int(g)): int(n) for g, n in zip(u, c)}

    def __getitem__(self, index):
        image = _load_image(self.filepaths[index])

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        image = torch.from_numpy(image)

        if self.use_meta:
            data = (image, torch.from_numpy(self.desc[index]))
        else:
            data = image

        return (
            self.image_ids[index],
            data,
            torch.tensor(self.targets[index], dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════
#  Concept columns
# ══════════════════════════════════════════════════════════════════
def _annotated_concepts(dfs: Sequence[pd.DataFrame]) -> List[str]:
    """`family::value` columns shared by every split — the CUB-style
    annotations when they have been joined onto the Waterbirds images."""
    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    return [c for c in dfs[0].columns
            if CONCEPT_SEP in str(c)
            and c in common
            and c not in _NON_ATTR_COLS
            and c not in PLACE_CONCEPTS]


def get_meta_data(df_train, df_test, *extra_dfs,
                  concepts: str = 'auto',
                  one_hot_place: bool = True):
    """
    Build the concept matrix a_i and return copies of every frame.

    `place` is expanded into binary indicators named with the project's
    `family::value` convention so that `concept_meta.build_family_index()`
    and `sci/coverage.py` see the same structure they see on CUB / Derm7pt:

        background::land   1 iff place == 0
        background::water  1 iff place == 1

    `one_hot_place=False` keeps only `background::water`.

    Returns (…frames…, meta_features, n_meta_features), matching the
    signature style of dataset_celebA.get_meta_data / dataset_cub.get_meta_data.
    """
    dfs = [df_train, df_test, *extra_dfs]

    annotated = _annotated_concepts(dfs)
    mode = str(concepts or 'auto').lower()
    if mode == 'auto':
        mode = 'both' if annotated else 'place'
    if mode not in ('place', 'annotated', 'both'):
        raise ValueError(f"concepts must be place|annotated|both|auto, "
                         f"got {concepts!r}")
    if mode in ('annotated', 'both') and not annotated:
        raise ValueError(
            f"concepts={mode!r} needs `family::value` concept columns in the "
            f"Waterbirds csv (none found). Join the CUB-200 attribute "
            f"annotations onto the Waterbirds images first — see "
            f"exp/expp21/build_waterbirds_csv.py --cub-attributes — or use "
            f"--waterbirds-concepts place.")

    place_cols = PLACE_CONCEPTS if one_hot_place else PLACE_CONCEPT_SINGLE

    out = []
    for d in dfs:
        if 'place' not in d.columns:
            raise KeyError(f"'place' column required. "
                           f"Got: {list(d.columns)[:10]}")
        d = d.copy()
        place = _to_int(d['place'], 'place')
        bad = set(np.unique(place)) - {PLACE_LAND, PLACE_WATER}
        if bad:
            raise ValueError(f'Unexpected place values {sorted(bad)}; '
                             f'expected 0/1')
        d['background::water'] = (place == PLACE_WATER).astype(np.int8)
        if one_hot_place:
            d['background::land'] = (place == PLACE_LAND).astype(np.int8)
        for c in annotated:
            d[c] = _to_bit(d[c])
        out.append(d)

    meta_features = ({'place': list(place_cols),
                      'annotated': list(annotated),
                      'both': list(place_cols) + list(annotated)}[mode])
    n_meta_features = len(meta_features)

    print(f'[Waterbirds] concept vocabulary = {mode!r}: '
          f'{n_meta_features} column(s)'
          + (f' -> {meta_features}' if n_meta_features <= 6
             else f' -> {meta_features[:4]} ... (+{n_meta_features - 4})'))
    if not extra_dfs:
        return out[0], out[1], meta_features, n_meta_features
    return (*out, meta_features, n_meta_features)


# ══════════════════════════════════════════════════════════════════
#  DataFrame construction
# ══════════════════════════════════════════════════════════════════
def _resolve_paths(image_ids: pd.Series, data_dir: str,
                   image_subdir: str = 'images') -> pd.Series:
    """Join image ids to a root, probing which layout this copy actually uses."""
    ids = image_ids.astype(str)
    probe = ids.head(8).tolist()
    candidates = [data_dir, os.path.join(data_dir, image_subdir)]
    for root in candidates:
        if any(os.path.isfile(os.path.join(root, p)) for p in probe):
            return ids.apply(lambda x: os.path.join(root, x))
    # nothing on disk yet (a manifest-only run, or a dry run): fall back to
    # the layout that exists as a directory, else the dataset root
    root = (os.path.join(data_dir, image_subdir)
            if os.path.isdir(os.path.join(data_dir, image_subdir))
            else data_dir)
    return ids.apply(lambda x: os.path.join(root, x))


def _prepare(df: pd.DataFrame, data_dir: str,
             image_subdir: str = 'images') -> pd.DataFrame:
    """Rename label -> target, add filepath, add subgroup."""
    df = df.copy()
    df = df.rename(columns={k: v for k, v in _RENAME.items()
                            if k in df.columns and v not in df.columns})

    for col in ('image_id', 'target'):
        if col not in df.columns:
            raise KeyError(f'Waterbirds csv must contain {col!r}. '
                           f'Got: {list(df.columns)[:10]}')

    df['target'] = _to_int(df['target'], 'target')

    df['filepath'] = _resolve_paths(df['image_id'], data_dir, image_subdir)

    # subgroup = 2*y + place, the standard Waterbirds group index. The version
    # this replaces referenced a `subgroup` column that was never created —
    # all the code that would have built it was commented out.
    if 'place' in df.columns:
        df['place'] = _to_int(df['place'], 'place')
        df['subgroup'] = group_index(df['target'], df['place'])

    # image_id is the immutable sample identifier every artefact is keyed by,
    # so it must be a string without the extension (§2 of the guideline).
    df['image_id'] = df['image_id'].map(_strip_ext)
    return df


def _split_column(df: pd.DataFrame) -> pd.Series:
    """Upstream metadata.csv encodes split as 0=train, 1=valid, 2=test."""
    if 'split' not in df.columns:
        raise KeyError('Single-file Waterbirds csv needs a `split` column')
    sp = df['split']
    if pd.api.types.is_numeric_dtype(sp):
        return sp.map({0: 'train', 1: 'valid', 2: 'test'})
    return sp.astype(str).str.strip().str.lower().replace(
        {'val': 'valid', 'validation': 'valid',
         'training': 'train', 'testing': 'test'})


def _load_all(data_dir: str, image_subdir: str = 'images') -> dict:
    """Return {split_name: DataFrame} from either csv layout."""
    paths = {k: os.path.join(data_dir, v) for k, v in _SPLIT_FILES.items()}
    if all(os.path.isfile(p) for p in paths.values()):
        return {k: _prepare(pd.read_csv(p), data_dir, image_subdir)
                for k, p in paths.items()}

    path = next((os.path.join(data_dir, f) for f in _SINGLE_FILES
                 if os.path.isfile(os.path.join(data_dir, f))), None)
    if path is None:
        missing = [v for k, v in _SPLIT_FILES.items()
                   if not os.path.isfile(paths[k])]
        raise FileNotFoundError(
            f'No Waterbirds csv found in {data_dir}. Missing {missing}, and '
            f'none of {_SINGLE_FILES} is present either. Build the split '
            f'files with:  python -m exp.expp21.build_waterbirds_csv '
            f'--data-dir {data_dir}')

    df = _prepare(pd.read_csv(path), data_dir, image_subdir)
    sp = _split_column(df)
    out = {name: df[sp == name].reset_index(drop=True)
           for name in ('train', 'valid', 'test')}
    missing = [k for k, v in out.items() if not len(v)]
    if missing:
        raise ValueError(f'Waterbirds: empty split(s) {missing} in {path}')
    return out


def describe_groups(df: pd.DataFrame, name: str = '') -> pd.DataFrame:
    """Group sizes + the label/place correlation, printed once per split."""
    if 'subgroup' not in df.columns:
        return pd.DataFrame()
    counts = (df.groupby('subgroup').size()
              .reindex(range(N_GROUPS), fill_value=0))
    tab = pd.DataFrame({
        'group': [group_name(g) for g in counts.index],
        'n': counts.to_numpy(dtype=int),
        'frac': (counts / max(1, len(df))).to_numpy(dtype=float),
    })
    if name:
        print(f'[Waterbirds] {name} groups:')
        print(tab.to_string(index=False))
    return tab


# ══════════════════════════════════════════════════════════════════
#  Public entry points
# ══════════════════════════════════════════════════════════════════
def get_waterbirds_df(data_dir: str, use_meta: bool,
                      concepts: str = 'auto',
                      one_hot_place: bool = True,
                      image_subdir: str = 'images',
                      verbose: bool = True):
    """
    Train + valid + test, matching `get_celeba_df` / `get_cub_df`.

    Returns
    -------
    df_train, df_valid, df_test : pd.DataFrame
    meta_features               : list[str] | None
    n_meta_features             : int
    mel_idx                     : int   positive class for AUC (waterbird)
    """
    d = _load_all(data_dir, image_subdir)
    df_train, df_valid, df_test = d['train'], d['valid'], d['test']

    if use_meta:
        # All three splits go through ONE call so they share one concept list
        # and one column order. The version this replaces built the concept
        # columns for valid in a separate inline loop with `.astype(int)`
        # while train/test got `.astype(np.int8)`.
        df_train, df_test, df_valid, meta_features, n_meta_features = \
            get_meta_data(df_train, df_test, df_valid,
                          concepts=concepts, one_hot_place=one_hot_place)
    else:
        meta_features, n_meta_features = None, 0

    if verbose:
        for nm, fr in (('train', df_train), ('valid', df_valid),
                       ('test', df_test)):
            describe_groups(fr, nm)

    return df_train, df_valid, df_test, meta_features, n_meta_features, MEL_IDX


# ── backwards-compatible aliases ─────────────────────────────────────────
def get_train_df(data_dir: str, use_meta: bool, concepts: str = 'auto',
                 one_hot_place: bool = True):
    """train + valid only (kept for the previous call signature)."""
    df_train, df_valid, _, meta_features, n_meta, mel_idx = get_waterbirds_df(
        data_dir, use_meta, concepts=concepts, one_hot_place=one_hot_place,
        verbose=False)
    return df_train, df_valid, meta_features, n_meta, mel_idx


def get_test_df(data_dir: str, use_meta: bool, concepts: str = 'auto',
                one_hot_place: bool = True):
    """train + valid + test (kept for the previous call signature)."""
    return get_waterbirds_df(data_dir, use_meta, concepts=concepts,
                             one_hot_place=one_hot_place, verbose=False)
