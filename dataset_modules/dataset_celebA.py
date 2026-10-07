"""
CelebA loader for concept-bottleneck / meta-feature models.

CSV schema (celeba_train.csv / celeba_valid.csv / celeba_test.csv)
─────────────────────────────────────────────────────────────────
    image_id, celeb_id, <40 binary attribute columns as True/False>

`Blond_Hair` is renamed to `target` and becomes the task label; the
remaining attributes become the concept vector.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_ATTR = 'Blond_Hair'          # renamed to 'target'
GROUP_ATTR = 'Male'                 # spurious-correlation group in the CelebA benchmark

# ── Columns that are NOT binary concept attributes ─────────────────
_NON_ATTR_COLS = {
    'target', 'image_id', 'celeb_id', 'filepath',
    'split', 'subgroup', 'partition', 'index', 'Unnamed: 0',
    TARGET_ATTR,                    # guard: if the rename ever fails
}

# Concepts that trivially leak the Blond_Hair label. Off by default so the
# behaviour matches the original script, but exposed as a flag.
_LEAKY_ATTRS = {'Blond_Hair', 'Black_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald'}


# ══════════════════════════════════════════════════════════════════
#  Boolean → int helpers
# ══════════════════════════════════════════════════════════════════
def _str_to_bit(v) -> int:
    """'True'/'yes'/'1'/1.0 → 1 ; 'False'/'no'/'0'/'-1'/NaN → 0."""
    s = str(v).strip().lower()
    if s in ('true', 't', 'yes', 'y', '1', '1.0'):
        return 1
    if s in ('false', 'f', 'no', 'n', '0', '0.0', '-1', '-1.0', '', 'nan'):
        return 0
    try:
        return int(float(s) > 0)
    except ValueError:
        return 0


def _bool_col_to_int(series: pd.Series) -> pd.Series:
    """
    Convert a Series from True/False (any encoding) → int8 0/1.

    Handles all three encodings CelebA CSVs appear in:
      * python bool          (True / False)
      * string after CSV     ('True' / 'False')
      * original ±1 encoding (1 / -1)   ← the raw list_attr_celeba.txt format
    """
    if series.dtype == bool or str(series.dtype) == 'bool':
        return series.astype(np.int8)
    if series.dtype == object:
        return series.map(_str_to_bit).astype(np.int8)
    # numeric: >0 → 1. Correct for both {0,1} and {-1,+1}.
    return (pd.to_numeric(series, errors='coerce').fillna(0) > 0).astype(np.int8)


def _convert_all_bool_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = _bool_col_to_int(df[col])
    return df


def _load_image(path: str) -> np.ndarray:
    """cv2.imread returns None on a missing/corrupt file — fail loudly."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════
class CelebA_Dataset(Dataset):
    def __init__(self, csv, mode, meta_features, dataset, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None and len(meta_features) > 0
        self.meta_features = meta_features
        self.desc_cols = meta_features
        self.transform = transform
        self.dataset = dataset

        # Materialise once as contiguous numpy. Indexing a DataFrame inside
        # __getitem__ is ~100x slower and is a real bottleneck with num_workers.
        self.image_ids = self.csv['image_id'].to_numpy()
        self.filepaths = self.csv['filepath'].to_numpy()
        self.targets = self.csv['target'].to_numpy(dtype=np.int64)
        if self.use_meta:
            self.desc = self.csv[self.desc_cols].to_numpy(dtype=np.float32)
            assert np.isfinite(self.desc).all(), \
                'NaN/inf in concept matrix — a column was not converted to int'
        else:
            self.desc = None

        print(f'[CelebA] mode: {self.mode}  n: {len(self)}  '
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
#  DataFrame construction
# ══════════════════════════════════════════════════════════════════
def _attr_cols(dfs, drop_leaky: bool = False):
    """
    Concept columns.
    """
    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    cols = [c for c in dfs[0].columns
            if c in common and c not in _NON_ATTR_COLS]
    if drop_leaky:
        cols = [c for c in cols if c not in _LEAKY_ATTRS]
    return cols


def get_meta_data(df_train, df_test, *extra_dfs, drop_leaky: bool = False):
    """
    1. Identify binary attribute columns shared by every split.
    2. Convert True/False → int 0/1 for attributes AND target, in every split.
    3. Return cleaned frames + feature list.

    Returns (df_train, df_test, meta_features, n_meta) when called with two
    frames, and (df_train, df_test, *extra, meta_features, n_meta) otherwise.
    """
    dfs = [df_train, df_test, *extra_dfs]
    attr_cols = _attr_cols(dfs, drop_leaky=drop_leaky)

    out = []
    for d in dfs:
        d = _convert_all_bool_cols(d, attr_cols)
        d['target'] = _bool_col_to_int(d['target'])
        out.append(d)

    n_meta = len(attr_cols)
    if not extra_dfs:
        return out[0], out[1], attr_cols, n_meta
    return (*out, attr_cols, n_meta)


def _load_and_rename(path: str, data_dir: str) -> pd.DataFrame:
    """Load a CelebA split; add filepath and rename the label column."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing CelebA split csv: {path}')
    df = pd.read_csv(path)
    if TARGET_ATTR not in df.columns and 'target' not in df.columns:
        raise KeyError(f'{TARGET_ATTR!r} not found in {path}. '
                       f'Columns: {list(df.columns)[:10]} ...')
    df = df.rename(columns={TARGET_ATTR: 'target'})
    df['filepath'] = df['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', str(x)))
    return df


def _no_meta(dfs):
    """Target-only cleanup when use_meta is False."""
    return [d.assign(target=_bool_col_to_int(d['target'])) for d in dfs]


def get_celeba_df(data_dir: str, use_meta: bool, drop_leaky: bool = False):
    df_train = _load_and_rename(os.path.join(data_dir, 'celeba_train.csv'), data_dir)
    df_valid = _load_and_rename(os.path.join(data_dir, 'celeba_valid.csv'), data_dir)
    df_test = _load_and_rename(os.path.join(data_dir, 'celeba_test.csv'), data_dir)

    if use_meta:
        # All three splits go through ONE call, so they share one attribute
        # list and one column order. The original version processed valid
        # separately, which could produce a different column order.
        df_train, df_test, df_valid, meta_features, n_meta = get_meta_data(
            df_train, df_test, df_valid, drop_leaky=drop_leaky)
    else:
        df_train, df_valid, df_test = _no_meta([df_train, df_valid, df_test])
        meta_features, n_meta = None, 0

    mel_idx = 1
    return df_train, df_valid, df_test, meta_features, n_meta, mel_idx