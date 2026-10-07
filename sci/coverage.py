"""
sci/coverage.py
─────────────────────────────────────────────────────────────────────────────
Per-class concept coverage, computed on the TRAINING SET ONLY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_FAMILY_SEPS = ("::", ":")

# Columns that, if present, provide human-readable names for integer targets.
CLASS_NAME_COLS = (
    "class_name", "target_name", "label_name", "diagnosis",
    "class", "species", "y_name",
)


def infer_concept_family(concept_name: str) -> str:
    """
    Default family parser.
    """
    if '::' in concept_name:
        return concept_name.split('::', 1)[0]
    return concept_name


def parse_family(name: str, sep: Optional[str] = None):
    """
    'pigment_network::atypical' -> ('pigment_network', 'atypical')
    'Eyeglasses'                -> (None, 'Eyeglasses')

    Unlike `infer_concept_family`, an unseparated name yields family=None so
    that "this dataset has no semantic families" is representable.
    """
    seps = (sep,) if sep else DEFAULT_FAMILY_SEPS
    for s in seps:
        if s and s in name:
            fam, val = name.split(s, 1)
            return fam.strip(), val.strip()
    return None, name


def concept_metadata(concepts: Sequence[str], sep: Optional[str] = None) -> pd.DataFrame:
    """DataFrame: index | concept | family | value (column order preserved)."""
    if len(concepts):
        fams, vals = zip(*(parse_family(c, sep) for c in concepts))
    else:
        fams, vals = (), ()
    return pd.DataFrame({
        "index": np.arange(len(concepts)),
        "concept": list(concepts),
        "family": list(fams),
        "value": list(vals),
    })


def has_family_structure(meta: pd.DataFrame) -> bool:
    """True when at least one concept name carries an explicit family."""
    return not meta["family"].isna().all()


def order_by_family(meta: pd.DataFrame):
    """
    Column ordering that groups concepts by semantic family. 
    The SCI analysis itself always runs on the atomic concepts,
    never on family aggregates.

    Returns
    -------
    perm       : np.ndarray[int]            column permutation
    boundaries : list[(family, start, end)] half-open spans in permuted space
                 (empty when the dataset has no families)
    """
    if not has_family_structure(meta):
        return np.arange(len(meta)), []

    fam = meta["family"].fillna("(unassigned)")
    order_of_first = {f: i for i, f in enumerate(dict.fromkeys(fam))}
    key = fam.map(order_of_first).to_numpy()
    perm = np.lexsort((meta["index"].to_numpy(), key))

    boundaries, start = [], 0
    fam_sorted = fam.to_numpy()[perm]
    for i in range(1, len(fam_sorted) + 1):
        if i == len(fam_sorted) or fam_sorted[i] != fam_sorted[start]:
            boundaries.append((fam_sorted[start], start, i))
            start = i
    return perm, boundaries


# ══════════════════════════════════════════════════════════════════
#  Hard vs. soft concept detection
# ══════════════════════════════════════════════════════════════════
def _detect_is_soft(concept_mat: np.ndarray, tol: float = 1e-6) -> bool:
    """True if any observed value isn't (approximately) exactly 0 or 1."""
    vals = concept_mat[~np.isnan(concept_mat)]
    if vals.size == 0:
        return False
    uniq = np.unique(np.round(vals, 6))
    return not np.all(np.isin(uniq, [0.0, 1.0]))


def _check_range(concept_mat: np.ndarray) -> None:
    if concept_mat.size == 0 or np.all(np.isnan(concept_mat)):
        return
    lo, hi = float(np.nanmin(concept_mat)), float(np.nanmax(concept_mat))
    if lo < -1e-6 or hi > 1 + 1e-6:
        raise ValueError(
            f'Concept values must lie in [0, 1], found range [{lo}, {hi}]. '
            f'Check that meta_features really are concept columns.')


# ══════════════════════════════════════════════════════════════════
#  Matrix primitives  (Experiment E1 core)
# ══════════════════════════════════════════════════════════════════
@dataclass
class Evidence:
    """
    Everything needs from a training split. The image x_i is never read:
    as a dataset-level analysis, independent of DINOv3 / SemCovNet / ERM /
    optimizer / architecture.
    """
    A: np.ndarray            # [N, K] float64 evidence, NaN replaced by 0
    M: np.ndarray            # [N, K] float64 observed-label mask
    codes: np.ndarray        # [N]    int, class code in 0..T-1
    classes: np.ndarray      # [T]    original label values
    class_names: List[str]   # [T]    human-readable names
    concepts: List[str]      # [K]
    has_missing: bool
    is_soft: bool

    @property
    def N(self) -> int:
        return self.A.shape[0]

    @property
    def K(self) -> int:
        return self.A.shape[1]

    @property
    def T(self) -> int:
        return len(self.classes)


def build_evidence(df_train: pd.DataFrame,
                   meta_features: Iterable[str],
                   target_col: str = 'target',
                   is_soft: Optional[bool] = None) -> Evidence:
    """
    Extract (Y, A, M) from a training DataFrame produced by load_dataset().

    NaN in a concept column is treated as a missing label: it enters the mask
    M rather than being silently counted as a_{i,k}=0, so C_{y,k} divides by
    the number of *observed* labels.
    """
    meta_features = list(meta_features)
    if not meta_features:
        raise ValueError('meta_features is empty — nothing to compute coverage for.')
    if target_col not in df_train.columns:
        raise KeyError(f'{target_col!r} not found in df_train.')
    missing = [c for c in meta_features if c not in df_train.columns]
    if missing:
        raise KeyError(f'meta_features not found in df_train: {missing[:5]} ...')

    A_raw = df_train[meta_features].to_numpy(dtype=np.float64)
    _check_range(A_raw)
    if is_soft is None:
        is_soft = _detect_is_soft(A_raw)

    M = (~np.isnan(A_raw)).astype(np.float64)
    has_missing = bool((M == 0).any())
    A = np.nan_to_num(A_raw, nan=0.0)

    y_all = df_train[target_col].to_numpy()
    classes, codes = np.unique(y_all, return_inverse=True)

    class_names = [str(c) for c in classes]
    for col in CLASS_NAME_COLS:
        if col == target_col or col not in df_train.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_train[col]):
            continue
        lut = (df_train[[target_col, col]].dropna()
               .drop_duplicates(subset=[target_col])
               .set_index(target_col)[col].to_dict())
        if len(lut) == len(classes):
            class_names = [str(lut.get(c, c)) for c in classes]
            break

    return Evidence(A=A, M=M, codes=codes.astype(np.int64), classes=classes,
                    class_names=class_names, concepts=meta_features,
                    has_missing=has_missing, is_soft=bool(is_soft))


def group_sums(X: np.ndarray, codes: np.ndarray, T: int, presorted: bool = False):
    """
    Sum rows of X within each class code -> (sums [T, C], counts [T]).

    Empty classes yield a zero row and count 0 rather than raising, which
    matters for class-balanced subsampling and permutation replicates.
    """
    if presorted:
        Xs, cs = X, codes
    else:
        order = np.argsort(codes, kind='stable')
        Xs, cs = X[order], codes[order]

    counts = np.bincount(cs, minlength=T)
    out = np.zeros((T, X.shape[1]), dtype=np.float64)
    nz = np.flatnonzero(counts)
    if nz.size:
        starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
        out[nz] = np.add.reduceat(Xs, starts[nz], axis=0)
    return out, counts


def coverage_matrices(A: np.ndarray, M: Optional[np.ndarray], codes: np.ndarray,
                      T: int, use_mask: bool = False, presorted: bool = False):
    """
    The fundamental object of Experiment E1.

    Returns
    -------
    C      : [T, K] coverage     C_{y,k} = n_{y,k} / d_{y,k}  (NaN if d = 0)
    n      : [T, K] support      n_{y,k}
    d      : [T, K] denominator  d_{y,k} (= N_y when fully observed)
    counts : [T]    class counts N_y
    """
    K = A.shape[1]
    if use_mask and M is not None:
        Z = np.concatenate([A * M, M], axis=1)
        S, counts = group_sums(Z, codes, T, presorted)
        n, d = S[:, :K], S[:, K:]
    else:
        n, counts = group_sums(A, codes, T, presorted)
        d = np.repeat(counts[:, None].astype(np.float64), K, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)
    return C, n, d, counts


def global_prevalence(A: np.ndarray, M: Optional[np.ndarray],
                      use_mask: bool = False) -> np.ndarray:
    """
    pi_k over the whole training split — the control for the reviewer
    objection "some concepts are simply globally rare". Compare against
    D_{y,k} = C_{y,k} - pi_k to separate global rarity from class-conditioned
    semantic undercoverage.
    """
    if use_mask and M is not None:
        den = M.sum(axis=0)
        num = (A * M).sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return A.mean(axis=0)


# ══════════════════════════════════════════════════════════════════
#  Core computation
# ══════════════════════════════════════════════════════════════════
def compute_coverage(
    df_train: pd.DataFrame,
    meta_features: Iterable[str],
    target_col: str = 'target',
    subgroup_col: Optional[str] = None,
    family_fn: Callable[[str], str] = infer_concept_family,
    is_soft: Optional[bool] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Compute C_{y,k}, n_{y,k}, d_{y,k}, pi_k and D_{y,k} for every
    (class, concept) pair, using only the rows in `df_train`.
    """
    meta_features = list(meta_features)
    ev_full = build_evidence(df_train, meta_features, target_col, is_soft)

    def _rows_for(sub_df: pd.DataFrame, subgroup_val):
        if len(sub_df) == 0:
            return []
        ev = build_evidence(sub_df, meta_features, target_col, ev_full.is_soft)
        C, n, d, counts = coverage_matrices(ev.A, ev.M, ev.codes, ev.T,
                                            use_mask=ev.has_missing)
        pi = global_prevalence(ev.A, ev.M, use_mask=ev.has_missing)
        T, K = C.shape
        return pd.DataFrame({
            'class': np.repeat(ev.classes, K),
            'concept': np.tile(meta_features, T),
            'family': np.tile([family_fn(c) for c in meta_features], T),
            'subgroup': subgroup_val,
            'C': C.reshape(-1),
            'n': n.reshape(-1),
            'N_y': np.repeat(counts, K),
            'd': d.reshape(-1),
            'pi': np.tile(pi, T),
            'D': (C - pi[None, :]).reshape(-1),
            'is_soft': bool(ev.is_soft),
        })

    frames = [_rows_for(df_train, None)]

    if subgroup_col is not None:
        if subgroup_col not in df_train.columns:
            raise KeyError(f'{subgroup_col!r} not found in df_train.')
        for sg in sorted(pd.unique(df_train[subgroup_col].dropna()), key=str):
            frames.append(_rows_for(df_train[df_train[subgroup_col] == sg], sg))

    df_cov = pd.concat([f for f in frames if len(f)], ignore_index=True)

    if validate:
        _validate_coverage(df_cov)

    return df_cov


def _validate_coverage(df_cov: pd.DataFrame, tol: float = 1e-6) -> None:
    c = df_cov['C']
    bad_c = df_cov[(c < -tol) | (c > 1 + tol)]          # NaN compares False
    assert bad_c.empty, f'C_{{y,k}} outside [0,1] for {len(bad_c)} rows:\n{bad_c.head()}'

    bad_n = df_cov[(df_cov['n'] < -tol) | (df_cov['n'] > df_cov['d'] + tol)]
    assert bad_n.empty, f'n_{{y,k}} outside [0, d_{{y,k}}] for {len(bad_n)} rows:\n{bad_n.head()}'


# ══════════════════════════════════════════════════════════════════
#  Coverage distribution
# ══════════════════════════════════════════════════════════════════
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def coverage_distribution(values, quantiles: Sequence[float] = QUANTILES) -> dict:
    """
    Descriptive statistics of the flattened coverage distribution. Accepts a
    coverage matrix, a 1-D array, or the 'C' column of a long-form table.

    Deliberately no single scalar "SCI score": the distribution itself is the
    evidence, and a new arbitrary scalar would just replace CDI with another.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    n_total = v.size
    v = v[np.isfinite(v)]
    out = {
        'n_pairs': int(n_total),
        'n_valid': int(v.size),
        'mean': float(v.mean()) if v.size else float('nan'),
        'std': float(v.std()) if v.size else float('nan'),
        'min': float(v.min()) if v.size else float('nan'),
        'max': float(v.max()) if v.size else float('nan'),
        'frac_zero': float((v == 0).mean()) if v.size else float('nan'),
        'frac_lt_0.01': float((v < 0.01).mean()) if v.size else float('nan'),
        'frac_lt_0.05': float((v < 0.05).mean()) if v.size else float('nan'),
        'frac_lt_0.10': float((v < 0.10).mean()) if v.size else float('nan'),
    }
    for q in quantiles:
        out[f'Q{int(round(q * 100))}'] = float(np.quantile(v, q)) if v.size else float('nan')
    return out


# ══════════════════════════════════════════════════════════════════
#  Summary / persistence
# ══════════════════════════════════════════════════════════════════
def summarize_coverage(df_cov: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Print a summary of a coverage table and return the
    coverage-distribution statistics.
    """
    overall = df_cov[df_cov['subgroup'].isna()]
    n_classes = overall['class'].nunique()
    n_concepts = overall['concept'].nunique()
    n_families = overall['family'].nunique()
    is_soft = bool(overall['is_soft'].iloc[0]) if len(overall) else None

    print(f'classes            : {n_classes}')
    print(f'concepts (K)       : {n_concepts}')
    print(f'concept families   : {n_families}')
    print(f'soft concepts      : {is_soft}')
    print(f'C range            : [{overall["C"].min():.3f}, {overall["C"].max():.3f}]')

    stats = coverage_distribution(overall['C'])
    print(f'\nclass-concept pairs: {stats["n_pairs"]} (valid: {stats["n_valid"]})')
    print('coverage quantiles : ' + '  '.join(
        f'Q{q}={stats[f"Q{q}"]:.3f}' for q in (10, 25, 50, 75, 90)))
    print(f'zero coverage      : {stats["frac_zero"]:.1%}   '
          f'<0.01: {stats["frac_lt_0.01"]:.1%}   '
          f'<0.05: {stats["frac_lt_0.05"]:.1%}')

    fam_mean = (overall.groupby('family')['C'].mean()
                .sort_values(ascending=False))
    print(f'\nTop {top_n} families by mean coverage:')
    print(fam_mean.head(top_n).to_string())
    print(f'\nBottom {top_n} families by mean coverage:')
    print(fam_mean.tail(top_n).to_string())

    n_subgroups = df_cov['subgroup'].dropna().nunique()
    if n_subgroups:
        print(f'\nsubgroups present  : {n_subgroups} '
              f'({sorted(df_cov["subgroup"].dropna().unique(), key=str)})')

    return stats


def save_coverage(df_cov: pd.DataFrame, path: str) -> None:
    """Save to .csv or .parquet based on the file extension."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith('.parquet'):
        df_cov.to_parquet(path, index=False)
    else:
        df_cov.to_csv(path, index=False)
    print(f'[coverage] saved {len(df_cov)} rows -> {path}')


def load_coverage(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════
#  Standalone test (synthetic data, no real dataset required)
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        'target': rng.randint(0, 3, size=n),
        'subgroup': rng.choice(['A', 'B'], size=n),
        'concept::a': rng.randint(0, 2, size=n).astype(float),
        'concept::b': rng.randint(0, 2, size=n).astype(float),
        'MONET_score': rng.rand(n),
    })
    df.loc[:9, 'concept::b'] = np.nan            # missing-label mask
    meta_features = ['concept::a', 'concept::b', 'MONET_score']

    df_cov = compute_coverage(df, meta_features, subgroup_col='subgroup')
    summarize_coverage(df_cov)
    print('\nSample rows:')
    print(df_cov.head(10).to_string(index=False))

    ev = build_evidence(df, meta_features)
    C, nn, d, counts = coverage_matrices(ev.A, ev.M, ev.codes, ev.T,
                                         use_mask=ev.has_missing)
    overall = df_cov[df_cov['subgroup'].isna()]
    wide = overall.pivot(index='class', columns='concept', values='C')[meta_features]
    assert np.allclose(C, wide.to_numpy(), equal_nan=True)

    # masked coverage must equal pandas' NaN-skipping groupby mean
    ref = df.groupby('target')[meta_features].mean().to_numpy()
    assert np.allclose(C, ref, equal_nan=True)

    perm, bounds = order_by_family(concept_metadata(meta_features))
    print(f'\nlong-form == matrix-form: OK   families: {[b[0] for b in bounds]}')