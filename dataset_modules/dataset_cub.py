"""
CUB-200-2011 loader for concept-bottleneck / meta-feature models.

CSV schema
──────────
    image_id, image_name, class, class_label, split, is_training_image,
    bill_shape::curved_(up_or_down), bill_shape::dagger, ... (312 concepts)

Concept columns are identified by the '::' separator, matching the standard
CBM preprocessing of CUB's attribute annotations.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CONCEPT_SEP = '::'
N_CLASSES = 200
VALID_FRACTION = 0.2      # carved out of train when no valid split exists
SPLIT_SEED = 42

_NON_ATTR_COLS = {
    'target', 'image_id', 'image_name', 'filepath', 'class', 'class_label',
    'split', 'is_training_image', 'index', 'Unnamed: 0',
}

_SPLIT_FILES = {
    'train': 'cub_train_set.csv',
    'valid': 'cub_valid_set.csv',
    'test': 'cub_test_set.csv',
}
_SINGLE_FILES = ['cub_attribute_annotations.csv', 'cub_data.csv', 'cub_meta.csv',
                 'CUB_processed.csv', 'attributes.csv']


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _to_bit(series: pd.Series) -> pd.Series:
    """Concept columns are already 0/1 ints; convert defensively anyway."""
    if series.dtype == bool or str(series.dtype) == 'bool':
        return series.astype(np.int8)
    if series.dtype == object:
        return series.map(
            lambda v: 1 if str(v).strip().lower() in ('1', '1.0', 'true', 't', 'yes')
            else 0).astype(np.int8)
    return (pd.to_numeric(series, errors='coerce').fillna(0) > 0).astype(np.int8)


# ══════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════
class CUB_Dataset(Dataset):
    def __init__(self, csv, mode, meta_features, dataset, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None and len(meta_features) > 0
        self.meta_features = meta_features
        self.desc_cols = meta_features
        self.transform = transform
        self.dataset = dataset

        self.image_ids = self.csv['image_id'].to_numpy()
        self.filepaths = self.csv['filepath'].to_numpy()
        self.targets = self.csv['target'].to_numpy(dtype=np.int64)
        if self.use_meta:
            self.desc = self.csv[self.desc_cols].to_numpy(dtype=np.float32)
            assert np.isfinite(self.desc).all(), 'NaN in concept matrix'
        else:
            self.desc = None

        print(f'[CUB] mode: {self.mode}  n: {len(self)}  '
              f'use_meta: {self.use_meta}  n_meta: {0 if self.desc is None else self.desc.shape[1]}')

    def __len__(self):
        return self.csv.shape[0]

    def copy(self):
        return self.csv.copy()

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
def get_meta_data(df_train, df_test, *extra_dfs):
    """
    Concept columns = columns containing '::', shared by every split.
    """
    dfs = [df_train, df_test, *extra_dfs]

    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)

    meta_features = [c for c in dfs[0].columns
                     if CONCEPT_SEP in str(c)
                     and c in common
                     and c not in _NON_ATTR_COLS]

    out = []
    for d in dfs:
        d = d.copy()
        for c in meta_features:
            d[c] = _to_bit(d[c])
        out.append(d)

    n_meta_features = len(meta_features)
    if not extra_dfs:
        return out[0], out[1], meta_features, n_meta_features
    return (*out, meta_features, n_meta_features)


# ══════════════════════════════════════════════════════════════════
#  DataFrame construction
# ══════════════════════════════════════════════════════════════════
def _prepare(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    df = df.copy()
    if 'image_name' not in df.columns:
        raise KeyError(f'CUB csv must contain image_name. Got: {list(df.columns)[:8]}')

    # df['image_id'] = df['image_id'].apply(lambda x: str(x))
    df['image_id'] = df['image_id'].astype(str).str.zfill(6)
    # df['image_id'] = df['image_id'].apply(lambda x: str(x).zfill(6))
 
    df['filepath'] = df['image_name'].apply(
        lambda x: os.path.join(data_dir, 'images', str(x)))

    # CUB class ids are 1..200; CrossEntropyLoss needs 0..199.
    cls = pd.to_numeric(df['class'], errors='coerce')
    if cls.isna().any():
        raise ValueError('Non-numeric values in the class column')
    df['target'] = (cls - 1 if cls.min() >= 1 else cls).astype(np.int64)
    return df


def _split_column(df: pd.DataFrame) -> pd.Series:
    if 'split' in df.columns:
        return df['split'].astype(str).str.strip().str.lower()
    if 'is_training_image' in df.columns:
        return pd.to_numeric(df['is_training_image'], errors='coerce').map(
            {1: 'train', 0: 'test'})
    raise KeyError('CUB csv needs a `split` or `is_training_image` column')


def _stratified_carve(df: pd.DataFrame, frac: float, seed: int):
    """Split train → (train, valid), stratified by class, deterministic."""
    rng = np.random.RandomState(seed)
    valid_idx = []
    for _, g in df.groupby('target'):
        idx = np.array(g.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        valid_idx.extend(idx[:k])
    valid_idx = set(valid_idx)
    mask = df.index.isin(list(valid_idx))
    return (df[~mask].reset_index(drop=True),
            df[mask].reset_index(drop=True))


def _load_all(data_dir: str):
    """Return dict of split-name → DataFrame, from either csv layout."""
    per_split = {k: os.path.join(data_dir, v) for k, v in _SPLIT_FILES.items()}
    if os.path.isfile(per_split['train']) and os.path.isfile(per_split['test']):
        out = {k: _prepare(pd.read_csv(p), data_dir)
               for k, p in per_split.items() if os.path.isfile(p)}
    else:
        path = next((os.path.join(data_dir, f) for f in _SINGLE_FILES
                     if os.path.isfile(os.path.join(data_dir, f))), None)
        if path is None:
            raise FileNotFoundError(
                f'No CUB csv found in {data_dir}. Expected one of '
                f'{list(_SPLIT_FILES.values())} or {_SINGLE_FILES}')
        df = _prepare(pd.read_csv(path), data_dir)
        sp = _split_column(df)
        out = {}
        for name, keys in (('train', {'train', 'training'}),
                           ('valid', {'valid', 'val', 'validation'}),
                           ('test', {'test', 'testing'})):
            sub = df[sp.isin(keys)].reset_index(drop=True)
            if len(sub):
                out[name] = sub

    if 'train' not in out or 'test' not in out:
        raise ValueError(f'CUB: missing split(s). Found: {sorted(out)}')

    if 'valid' not in out:
        print(f'[CUB] No validation split found — carving {VALID_FRACTION:.0%} '
              f'out of train (stratified, seed={SPLIT_SEED}).')
        out['train'], out['valid'] = _stratified_carve(
            out['train'], VALID_FRACTION, SPLIT_SEED)

    return out

def get_cub_df(data_dir: str, use_meta: bool):
    """Train + valid + test. Mirrors get_test_df / get_derm_test_df."""
    d = _load_all(data_dir)
    df_train, df_valid, df_test = d['train'], d['valid'], d['test']

    if use_meta:
        df_train, df_test, df_valid, meta_features, n_meta = get_meta_data(
            df_train, df_test, df_valid)
    else:
        meta_features, n_meta = None, 0

    mel_idx = 0
    return df_train, df_valid, df_test, meta_features, n_meta, mel_idx
