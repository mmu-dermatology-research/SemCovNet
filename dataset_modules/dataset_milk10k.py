"""
dataset_milk.py  —  MILK10k Derm + Clinic dataset modules
==========================================================

Two modalities share the same Dataset class but different get_*_df loaders:
    MILK10kDerm   : dermoscopy images
    MILK10kClinic : clinical photography images

CSV layout depends on ``num_classes``:

    num_classes == 2   (binary melanoma)
        subdir : train_valid_test/
        files  : derm_train_set.csv   / derm_valid_set.csv   / derm_test_set.csv
                 clinic_train_set.csv / clinic_valid_set.csv / clinic_test_set.csv

    num_classes  > 2   (multi-class)
        subdir : SCI_LTD_MCC/
        files  : derm_train_set_ltd.csv   / derm_valid_set_ltd.csv   / derm_test_set_ltd.csv
                 clinic_train_set_ltd.csv / clinic_valid_set_ltd.csv / clinic_test_set_ltd.csv

Key columns:
    lesion_id : lesion folder name
    isic_id   : image filename stem
    MEL       : binary melanoma target   (num_classes == 2, renamed to 'target')
    class     : multi-class target       (num_classes  > 2, renamed to 'target')
    site      : anatomical site (renamed to 'subgroup')
    MONET_*   : soft MONET concept scores in [0, 1]

Image path construction:
    data_dir/MILK10k_Training_Input/{lesion_id}/{isic_id}.jpg

Concept setup:
    probe_type = 'regression'   (MSELoss on MONET soft scores)
    One joint probe: Linear(D to K_MONET) predicts all MONET scores jointly.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple


# =============================================================
# Config
# =============================================================
GROUP_ATTR = 'site'

BINARY_csv_subdir     = 'train_valid_test'   # subdir containing the binary CSVs
MULTICLASS_csv_subdir = 'SCI_LTD_MCC'        # subdir containing the multi-class CSVs

BINARY_csv_suffix     = ''                   # derm_train_set.csv
MULTICLASS_csv_suffix = '_ltd'               # derm_train_set_ltd.csv


def _csv_layout(num_classes: int,
                csv_subdir: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve (csv_subdir, csv_suffix) from num_classes.

    An explicitly-passed ``csv_subdir`` wins; the suffix always follows
    num_classes (binary to '', multi-class to '_ltd').
    """
    if num_classes == 2:
        return (csv_subdir or BINARY_csv_subdir), BINARY_csv_suffix
    return (csv_subdir or MULTICLASS_csv_subdir), MULTICLASS_csv_suffix


def _csv_path(data_dir: str, csv_subdir: str,
              modality: str, split: str, suffix: str) -> str:
    """e.g. data_dir/SCI_LTD_MCC/derm_train_set_ltd.csv"""
    return os.path.join(data_dir, csv_subdir, f'{modality}_{split}_set{suffix}.csv')


# =============================================================
# Dataset class  (shared by Derm and Clinic)
# =============================================================
class MILK10k_Dataset(Dataset):
    """
    Mirrors WaterBirds_Dataset interface exactly:
        __getitem__ returns (image_id, data, target)
        data = image_tensor              if use_meta=False
        data = (image_tensor, desc)      if use_meta=True
        desc = MONET soft scores (K,)    float tensor in [0, 1]
    """

    def __init__(self, csv, mode, meta_features, dataset, transform=None):
        self.csv           = csv.reset_index(drop=True)
        self.mode          = mode
        self.use_meta      = meta_features is not None
        self.meta_features = meta_features
        self.transform     = transform
        self.dataset       = dataset
        self.desc_cols     = meta_features
        print(f"mode: {self.mode}  dataset: {self.dataset}  "
              f"use_meta: {self.use_meta}  "
              f"MONET concepts: {len(meta_features) if meta_features else 0}")

    def __len__(self):
        return self.csv.shape[0]

    def copy(self):
        return self.csv.copy()

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        if image is None:
            raise FileNotFoundError(f"Image not found: {row.filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res   = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)    # (3, H, W)

        if self.use_meta:
            desc_data = self.csv.loc[index, self.desc_cols].values.astype(np.float32)
            data = (
                torch.tensor(image).float(),
                torch.tensor(desc_data).float(),
            )
        else:
            data = torch.tensor(image).float()

        return (
            row.image_id,
            data,
            torch.tensor(int(self.csv.iloc[index].target)).long()
        )


# =============================================================
# Helpers
# =============================================================

def _build_filepath(row, data_dir: str, img_subdir: str) -> str:
    return os.path.join(data_dir, img_subdir,
                        str(row['lesion_id']), f"{row['isic_id']}.jpg")


def _load_and_rename(
    path:       str,
    data_dir:   str,
    num_classes: int,
    img_subdir: str = 'MILK10k_Training_Input',
) -> pd.DataFrame:
    """Load a MILK10k CSV and standardise column names + add filepath/image_id."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MILK10k CSV not found: {path}")

    df = pd.read_csv(path)

    if num_classes == 2:
        df = df.rename(columns={'MEL':  'target'})
    else:
        df = df.rename(columns={'class':  'target'})

    df = df.rename(columns={GROUP_ATTR: 'subgroup'})
    df['filepath'] = df.apply(
        lambda row: _build_filepath(row, data_dir, img_subdir), axis=1)
    df['image_id'] = df['isic_id'].astype(str)    # unique sample identifier
    df['subgroup'] = df['subgroup'].astype(str).fillna('unknown')
    df['target']   = df['target'].astype(int)
    return df


# Filled in by `encode_subgroups()`: the sorted category list, so code i
# always means SUBGROUP_CATEGORIES[i]. Read it with `subgroup_names()` when
# you want the anatomical site behind a group index in a table.
SUBGROUP_CATEGORIES: List[str] = []


def subgroup_names() -> List[str]:
    """The site names behind the integer `subgroup` codes, in code order."""
    return list(SUBGROUP_CATEGORIES)


def encode_subgroups(*dfs: Optional[pd.DataFrame], col: str = 'subgroup',
                     verbose: bool = True):
    """
    Turn a CATEGORICAL `subgroup` column into contiguous integer codes.

    MILK10k's subgroup is the anatomical `site` — 'head_neck_face',
    'lower_extremity', ... — and everything downstream expects s_i to be an
    integer:

        methods/sota/groups.py::group_arrays   df[col].to_numpy(np.int64)
        data_loaders.BaselineDataset           df[col].to_numpy("int64")

    which is where `ValueError: invalid literal for int() with base 10:
    'head_neck_face'` comes from.

    Returns (frames..., categories).
    """
    global SUBGROUP_CATEGORIES
    frames = list(dfs)
    present = [d for d in frames if d is not None and col in d.columns]
    if not present:
        return (*frames, [])

    if all(pd.api.types.is_numeric_dtype(d[col]) for d in present):
        out, seen = [], set()
        for d in frames:
            if d is None or col not in d.columns:
                out.append(d)
                continue
            d = d.copy()
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(-1).astype(np.int64)
            seen |= set(d[col].tolist())
            out.append(d)
        cats = [str(v) for v in sorted(seen)]
        SUBGROUP_CATEGORIES = cats
        if verbose:
            print(f"  [MILK] subgroup already numeric: {len(cats)} code(s) "
                  f"{cats[:8]}{' ...' if len(cats) > 8 else ''}")
        return (*out, cats)

    values = set()
    for d in present:
        values |= set(d[col].astype(str).str.strip().replace('', 'unknown'))
    cats = sorted(values)                       # <- the sort that fixes the code
    code = {c: i for i, c in enumerate(cats)}
    SUBGROUP_CATEGORIES = cats

    out, counts = [], {c: 0 for c in cats}
    for d in frames:
        if d is None or col not in d.columns:
            out.append(d)
            continue
        d = d.copy()
        raw = d[col].astype(str).str.strip().replace('', 'unknown')
        for c, n in raw.value_counts().items():
            counts[c] = counts.get(c, 0) + int(n)
        d[col] = raw.map(code).astype(np.int64)
        out.append(d)

    if verbose:
        print(f"  [MILK] subgroup '{col}' encoded from {len(cats)} sorted "
              f"categories (union over the frames given):")
        for i, c in enumerate(cats):
            print(f"           {i:>3d} = {c:<24s} n={counts.get(c, 0)}")
    return (*out, cats)


def get_meta_data(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    *extra_dfs: pd.DataFrame,
) -> Tuple:
    """
    Identify MONET_* columns as concept targets, and encode `subgroup`.

    MONET scores are soft values in [0, 1] — regression targets.
    probe_type = 'regression'  (MSELoss)
    """
    # ── categorical subgroup -> integer codes (sorted) ──────────────────
    *frames, _cats = encode_subgroups(df_train, df_test, *extra_dfs)
    df_train, df_test, *extra_out = frames

    meta_features   = [c for c in df_train.columns if c.startswith('MONET_')]
    n_meta_features = len(meta_features)

    if n_meta_features == 0:
        raise RuntimeError(
            "No MONET_* columns found. "
            "Load a descriptor CSV (e.g. derm_train_set_descriptors.csv) or "
            "ensure the standard CSV contains MONET columns."
        )

    print(f"  MONET concepts ({n_meta_features}): {meta_features[:5]} ...")
    print(f"  Score range (train): "
          f"[{df_train[meta_features].min().min():.3f}, "
          f"{df_train[meta_features].max().max():.3f}]")

    if not extra_dfs:
        return df_train, df_test, meta_features, n_meta_features
    return (df_train, df_test, *extra_out, meta_features, n_meta_features)


def _get_milk_df(
    modality:    str,
    data_dir:    str,
    use_meta:    bool,
    num_classes: int,
    img_subdir:  str = 'MILK10k_Training_Input',
    csv_subdir:  Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           Optional[List[str]], int, int]:
    """Shared loader for both modalities ('derm' | 'clinic')."""
    csv_subdir, suffix = _csv_layout(num_classes, csv_subdir)

    print(f"\n  [MILK-{modality}] num_classes={num_classes}  "
          f"csv_subdir='{csv_subdir}'  suffix='{suffix}'")

    df_train = _load_and_rename(
        _csv_path(data_dir, csv_subdir, modality, 'train', suffix),
        data_dir, num_classes, img_subdir)
    df_valid = _load_and_rename(
        _csv_path(data_dir, csv_subdir, modality, 'valid', suffix),
        data_dir, num_classes, img_subdir)
    df_test  = _load_and_rename(
        _csv_path(data_dir, csv_subdir, modality, 'test', suffix),
        data_dir, num_classes, img_subdir)

    print(f'  [MILK-{modality}] train: {len(df_train)}  '
          f'valid: {len(df_valid)}  test: {len(df_test)}  '
          f'classes: {sorted(df_train["target"].unique())}')

    if use_meta:
        # df_valid goes through the SAME call: it needs the same concept
        # column order and, above all, the same subgroup codes.
        df_train, df_test, df_valid, meta_features, n_meta = \
            get_meta_data(df_train, df_test, df_valid)
    else:
        # ERM runs pass --use-meta off, and they still need an integer
        # subgroup: `BaselineDataset` reads the column on every dataset.
        df_train, df_test, df_valid, _ = encode_subgroups(
            df_train, df_test, df_valid)
        meta_features = None
        n_meta        = 0

    print(f'  [MILK-{modality}] subgroup codes: '
          f'{ {i: c for i, c in enumerate(subgroup_names())} }')

    mel_idx = 1
    return df_train, df_valid, df_test, meta_features, n_meta, mel_idx


# =============================================================
# MILK10k Derm loaders
# =============================================================
def get_derm_df(
    data_dir:    str,
    use_meta:    bool,
    num_classes: int,
    img_subdir:  str = 'MILK10k_Training_Input',
    csv_subdir:  Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           Optional[List[str]], int, int]:
    """
    Returns (df_train, df_valid, df_test, meta_features, n_meta_features, mel_idx)

    num_classes == 2 to train_valid_test/derm_{train,valid,test}_set.csv
    num_classes  > 2 to SCI_LTD_MCC/derm_{train,valid,test}_set_ltd.csv
    """
    return _get_milk_df('derm', data_dir, use_meta, num_classes,
                        img_subdir, csv_subdir)


# =============================================================
# MILK10k Clinic loaders
# =============================================================
def get_clinic_df(
    data_dir:    str,
    use_meta:    bool,
    num_classes: int,
    img_subdir:  str = 'MILK10k_Training_Input',
    csv_subdir:  Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           Optional[List[str]], int, int]:
    """
    Returns (df_train, df_valid, df_test, meta_features, n_meta_features, mel_idx)

    num_classes == 2 to train_valid_test/clinic_{train,valid,test}_set.csv
    num_classes  > 2 to SCI_LTD_MCC/clinic_{train,valid,test}_set_ltd.csv
    """
    return _get_milk_df('clinic', data_dir, use_meta, num_classes,
                        img_subdir, csv_subdir)