"""
Derm7pt loader for concept-bottleneck / meta-feature models.

CSV schema (train_set.csv / valid_set.csv / test_set.csv)
────────────────────────────────────────────────────────
    case_num, diagnosis, seven_point_score, pigment_network, streaks,
    pigmentation, regression_structures, dots_and_globules, blue_whitish_veil,
    vascular_structures, level_of_diagnostic_difficulty, elevation, location,
    sex, management, clinic, derm, case_id, notes, label, class

Concept annotations: concept/derm7pt_gt_criteria_annotations.csv.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pathlib import Path


# The 7 textual criteria columns in the main CSV. Used as a fallback concept
# source (one-hot encoded) if the criteria annotation file is unavailable.
SEVEN_POINT_CRITERIA = [
    'pigment_network', 'streaks', 'pigmentation', 'regression_structures',
    'dots_and_globules', 'blue_whitish_veil', 'vascular_structures',
]

_NON_ATTR_COLS = {
    'target', 'image_id', 'filepath', 'derm', 'clinic', 'case_num', 'case_id',
    'notes', 'diagnosis', 'label', 'class', 'split', 'management', 'sex',
    'location', 'elevation', 'level_of_diagnostic_difficulty',
    'seven_point_score', 'modality', 'index', 'Unnamed: 0',
}

_CRITERIA_FILE = os.path.join('concept', 'derm7pt_gt_criteria_annotations.csv')


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════
class Derm7pt_Dataset(Dataset):
    def __init__(self, csv, mode, meta_features, dataset_name, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None and len(meta_features) > 0
        self.meta_features = meta_features
        self.desc_cols = meta_features
        self.transform = transform
        self.dataset = dataset_name

        self.image_ids = self.csv['image_id'].to_numpy()
        self.filepaths = self.csv['filepath'].to_numpy()
        self.targets = self.csv['target'].to_numpy(dtype=np.int64)
        if self.use_meta:
            self.desc = self.csv[self.desc_cols].to_numpy(dtype=np.float32)
            assert np.isfinite(self.desc).all(), \
                'NaN in concept matrix — an image_id failed to match the ' \
                'criteria annotations file'
        else:
            self.desc = None

        print(f'[Derm7pt] mode: {self.mode}  n: {len(self)}  '
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
def _is_binary(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return len(s) > 0 and set(np.unique(s.to_numpy())).issubset({0, 1})


def get_meta_data(df_train, df_test, *extra_dfs):
    """
    Concept columns = numeric {0,1} columns present in every split.
    """
    dfs = [df_train, df_test, *extra_dfs]

    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)

    meta_features = [
        c for c in dfs[0].columns
        if c in common
        and c not in _NON_ATTR_COLS
        and not c.endswith(('_x', '_y'))
        and all(_is_binary(d[c]) for d in dfs)
    ]

    out = []
    for d in dfs:
        d = d.copy()
        for c in meta_features:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0).astype(np.int8)
        out.append(d)

    n_meta_features = len(meta_features)
    if not extra_dfs:
        return out[0], out[1], meta_features, n_meta_features
    return (*out, meta_features, n_meta_features)


def _onehot_criteria(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Fallback concepts: one-hot encode the 7 textual criteria columns."""
    df = df.copy()
    for col, cats in categories.items():
        vals = df[col].astype(str).str.strip().str.lower()
        for cat in cats:
            df[f'{col}::{cat}'] = (vals == cat).astype(np.int8)
    return df


def _criteria_categories(dfs):
    cats = {}
    for col in SEVEN_POINT_CRITERIA:
        vals = pd.concat([d[col] for d in dfs if col in d.columns])
        if len(vals) == 0:
            continue
        cats[col] = sorted(vals.astype(str).str.strip().str.lower().unique())
    return cats


# ══════════════════════════════════════════════════════════════════
#  DataFrame construction
# ══════════════════════════════════════════════════════════════════
def _load_split(data_dir: str, filename: str, image_col: str,
                num_classes: int, class_map: dict = None) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing Derm7pt split csv: {path}')
    df = pd.read_csv(path)

    df['image_id'] = df[image_col].apply(lambda x: Path(str(x)).stem)
    df['filepath'] = df[image_col].apply(
        lambda x: os.path.join(data_dir, 'images', str(x)))

    if num_classes == 2:
        df['target'] = df['diagnosis'].astype(str).str.lower().apply(
            lambda x: 1 if 'melanoma' in x else 0).astype(np.int64)
    else:
        # 'class' in the CSV is not guaranteed to be 0-based or contiguous
        # (the samples show values 1, 2, 4). CrossEntropyLoss requires
        # labels in [0, num_classes). Remap with a shared, fixed mapping.
        if 'class' not in df.columns:
            raise KeyError(f"'class' column required for num_classes="
                           f"{num_classes} but not present in {path}")
        raw = pd.to_numeric(df['class'], errors='coerce').astype('Int64')
        df['target'] = raw.map(class_map).astype(np.int64)
        if df['target'].isna().any():
            raise ValueError(f'Unmapped class values in {path}')
    return df


def _build_class_map(data_dir: str, files) -> dict:
    """Contiguous 0-based mapping built from the union of all splits."""
    vals = set()
    for f in files:
        p = os.path.join(data_dir, f)
        if os.path.isfile(p):
            col = pd.read_csv(p, usecols=['class'])['class']
            vals |= set(pd.to_numeric(col, errors='coerce').dropna().astype(int))
    return {v: i for i, v in enumerate(sorted(vals))}


def _merge_criteria(data_dir: str, dfs):
    """
    Merge the concept annotations onto each split.
    """
    path = os.path.join(data_dir, _CRITERIA_FILE)
    if not os.path.isfile(path):
        print(f'[Derm7pt] {path} not found — falling back to one-hot '
              f'encoding of the textual 7-point criteria.')
        cats = _criteria_categories(dfs)
        return [_onehot_criteria(d, cats) for d in dfs], False

    crit = pd.read_csv(path)
    crit = crit.rename(columns={'ImageID': 'image_id', 'image_name': 'image_id'})
    if 'image_id' not in crit.columns:
        crit = crit.rename(columns={crit.columns[0]: 'image_id'})
    crit['image_id'] = crit['image_id'].apply(lambda x: Path(str(x)).stem)
    crit = crit.drop_duplicates(subset='image_id')

    overlap = (set(crit.columns) & set(dfs[0].columns)) - {'image_id'}
    if overlap:
        crit = crit.drop(columns=sorted(overlap))

    out = []
    for d in dfs:
        m = d.merge(crit, on='image_id', how='left')
        new_cols = [c for c in crit.columns if c != 'image_id']
        miss = m[new_cols].isna().all(axis=1).sum() if new_cols else 0
        if miss:
            print(f'[Derm7pt] WARNING: {miss}/{len(m)} rows had no concept '
                  f'annotation (image_id mismatch); filled with 0.')
        out.append(m)
    return out, True


def _finish(dfs, use_meta, data_dir):
    if not use_meta:
        return dfs, None, 0
    dfs, _ = _merge_criteria(data_dir, dfs)
    *dfs, meta_features, n_meta = get_meta_data(*dfs)
    if n_meta == 0:
        print('[Derm7pt] WARNING: no binary concept columns were found.')
    return list(dfs), meta_features, n_meta


def _mel_index(num_classes, class_map, data_dir, files):
    """Index of the melanoma class, used as the positive class for AUC."""
    if num_classes == 2:
        return 1
    for f in files:
        p = os.path.join(data_dir, f)
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p, usecols=['diagnosis', 'class'])
        mel = df[df['diagnosis'].astype(str).str.lower().str.contains('melanoma')]
        if len(mel):
            return int(class_map[int(mel['class'].iloc[0])])
    return 1


# ── derm (dermoscopic) images ──────────────────────────────────────
def get_derm_df(data_dir, use_meta, num_classes):
    return _get_df(data_dir, use_meta, num_classes, 'derm')

# ── clinic (clinical) images ───────────────────────────────────────
def get_clinic_df(data_dir, use_meta, num_classes):
    return _get_df(data_dir, use_meta, num_classes, 'clinic')

def _get_df(data_dir, use_meta, num_classes, image_col):
    """
    Single implementation behind the four public getters.
    """
    files = ['train_set.csv', 'valid_set.csv', 'test_set.csv']
    class_map = _build_class_map(data_dir, ['train_set.csv', 'valid_set.csv',
                                            'test_set.csv']) if num_classes != 2 else None

    dfs = [_load_split(data_dir, f, image_col, num_classes, class_map) for f in files]
    dfs, meta_features, n_meta = _finish(dfs, use_meta, data_dir)

    mel_idx = _mel_index(num_classes, class_map, data_dir, files)

    return dfs[0], dfs[1], dfs[2], meta_features, n_meta, mel_idx