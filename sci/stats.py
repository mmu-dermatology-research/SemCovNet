"""
sci/stats.py
─────────────────────────────────────────────────────────────────────────────
  > coverage quantiles + Tail/Middle/Head strata      (descriptive)
  > prevalence-preserving permutation null            (finite-sample control)
  > class-balanced subsampling sensitivity            (class-imbalance control)
  > within-class bootstrap confidence intervals       (estimator uncertainty)

"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .coverage import coverage_matrices, group_sums

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanvar_by_class(C: np.ndarray) -> np.ndarray:
    """V_k = Var_y(C_{y,k}) — the observed across-class coverage variation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanvar(C, axis=0)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    x, y = a[m], b[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with average ties (pandas rank, no scipy needed)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    return pearson(ra, rb)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ranked coverage distribution + Tail / Middle / Head
# ─────────────────────────────────────────────────────────────────────────────

def coverage_quantiles(C: np.ndarray, quantiles: Sequence[float] = QUANTILES) -> dict:
    v = C.reshape(-1)
    v = v[np.isfinite(v)]
    out = {
        "n_pairs": int(C.size),
        "n_valid": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "std": float(v.std()) if v.size else float("nan"),
        "min": float(v.min()) if v.size else float("nan"),
        "max": float(v.max()) if v.size else float("nan"),
        "frac_zero": float((v == 0).mean()) if v.size else float("nan"),
        "frac_lt_0.01": float((v < 0.01).mean()) if v.size else float("nan"),
        "frac_lt_0.05": float((v < 0.05).mean()) if v.size else float("nan"),
        "frac_lt_0.10": float((v < 0.10).mean()) if v.size else float("nan"),
    }
    for q in quantiles:
        out[f"Q{int(round(q*100))}"] = float(np.quantile(v, q)) if v.size else float("nan")
    return out


def assign_strata(C: np.ndarray, q25: float, q75: float) -> np.ndarray:
    """
    Descriptive training-coverage strata (NOT thresholds defining SCI):
        Tail   : C <= Q25
        Middle : Q25 < C < Q75
        Head   : C >= Q75
    """
    band = np.full(C.shape, "", dtype=object)
    finite = np.isfinite(C)
    band[finite & (C <= q25)] = "Tail"
    band[finite & (C > q25) & (C < q75)] = "Middle"
    band[finite & (C >= q75)] = "Head"
    band[~finite] = "Undefined"
    return band


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Prevalence-preserving permutation null
# ─────────────────────────────────────────────────────────────────────────────

def permutation_null(A, M, codes, T, B: int = 1000, seed: int = 0,
                     use_mask: bool = False, verbose: bool = True):
    """
    Shuffle Y while keeping A fixed. This preserves concept prevalence pi_k,
    the class counts N_y and N, but destroys any genuine class-concept
    association.

        V_k^obs   = Var_y(C_{y,k})
        p_k       = (1 + #{b : V_{k,b}^perm >= V_k^obs}) / (B + 1)

    Returns (DataFrame per concept, perm_matrix [B,K]).
    """
    C_obs, *_ = coverage_matrices(A, M, codes, T, use_mask=use_mask)
    V_obs = _nanvar_by_class(C_obs)

    rng = np.random.default_rng(seed)
    K = A.shape[1]
    perm_V = np.empty((B, K), dtype=np.float64)
    perm_codes = codes.copy()

    for b in range(B):
        rng.shuffle(perm_codes)
        C_p, *_ = coverage_matrices(A, M, perm_codes, T, use_mask=use_mask)
        perm_V[b] = _nanvar_by_class(C_p)
        if verbose and B >= 100 and (b + 1) % max(1, B // 5) == 0:
            print(f"    permutation {b+1}/{B}")

    ge = (perm_V >= V_obs[None, :]).sum(axis=0)
    p = (1.0 + ge) / (B + 1.0)
    q = benjamini_hochberg(p)

    df = pd.DataFrame({
        "V_obs": V_obs,
        "V_perm_mean": perm_V.mean(axis=0),
        "V_perm_q95": np.quantile(perm_V, 0.95, axis=0),
        "V_perm_max": perm_V.max(axis=0),
        "effect_ratio": V_obs / np.maximum(perm_V.mean(axis=0), 1e-12),
        "p_value": p,
        "q_value_bh": q,
        "n_permutations": B,
    })
    return df, perm_V


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Class-balanced subsampling sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def balance_sensitivity(A, M, codes, T, C_full, repeats: int = 20,
                        n_per_class: Optional[int] = None, seed: int = 0,
                        use_mask: bool = False, quantiles=QUANTILES,
                        verbose: bool = True):
    """
    Repeatedly subsample every class to the same size and recompute coverage.

    If rho(C_full, C_balanced) stays high, the observed semantic imbalance is
    not merely a by-product of unequal class counts.

    Returns (per-repeat DataFrame, mean balanced coverage matrix, summary dict).
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(codes, minlength=T)
    idx_by_class = [np.flatnonzero(codes == t) for t in range(T)]
    n_take = int(counts[counts > 0].min()) if n_per_class is None else int(n_per_class)

    q25 = float(np.nanquantile(C_full, 0.25))
    q75 = float(np.nanquantile(C_full, 0.75))
    band_full = assign_strata(C_full, q25, q75)

    rows, acc = [], np.zeros_like(C_full)
    for r in range(repeats):
        sel = np.concatenate([
            (ix if ix.size <= n_take else rng.choice(ix, size=n_take, replace=False))
            for ix in idx_by_class if ix.size > 0
        ])
        sub_codes = codes[sel]
        C_b, *_ = coverage_matrices(A[sel], None if M is None else M[sel],
                                    sub_codes, T, use_mask=use_mask)
        acc += np.nan_to_num(C_b)

        a, b = C_full.reshape(-1), C_b.reshape(-1)
        qb25 = float(np.nanquantile(C_b, 0.25))
        qb75 = float(np.nanquantile(C_b, 0.75))
        band_b = assign_strata(C_b, qb25, qb75)
        both = (band_full != "Undefined") & (band_b != "Undefined")

        row = {
            "repeat": r,
            "n_per_class": n_take,
            "spearman": spearman(a, b),
            "pearson": pearson(a, b),
            "mean_abs_diff": float(np.nanmean(np.abs(a - b))),
            "band_agreement": float((band_full[both] == band_b[both]).mean()),
            "tail_recall": float(
                (band_b[(band_full == "Tail") & both] == "Tail").mean()
            ) if ((band_full == "Tail") & both).any() else float("nan"),
        }
        for q in quantiles:
            row[f"Q{int(round(q*100))}"] = float(np.nanquantile(C_b, q))
        rows.append(row)
        if verbose and (r + 1) % max(1, repeats // 4) == 0:
            print(f"    balance repeat {r+1}/{repeats}")

    df = pd.DataFrame(rows)
    C_bal_mean = acc / repeats
    summary = {
        "n_per_class": n_take,
        "repeats": repeats,
        "spearman_mean": float(df["spearman"].mean()),
        "spearman_std": float(df["spearman"].std(ddof=0)),
        "pearson_mean": float(df["pearson"].mean()),
        "band_agreement_mean": float(df["band_agreement"].mean()),
        "Q25_mean": float(df["Q25"].mean()),
        "Q50_mean": float(df["Q50"].mean()),
        "Q75_mean": float(df["Q75"].mean()),
    }
    return df, C_bal_mean, summary


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Within-class bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_coverage(A, M, codes, T, B: int = 1000, seed: int = 0,
                       use_mask: bool = False, cell_index: Optional[np.ndarray] = None,
                       quantiles=QUANTILES, alpha: float = 0.05,
                       verbose: bool = True):
    """
    Resample within each class with replacement and recompute C.

    cell_index : flat indices (into C.reshape(-1)) of the class-concept pairs
                 to keep per-cell CIs for. None = all cells (fine for CelebA /
                 Derm7pt; for CUB pass a subset, 200x312x1000 floats is 250 MB).

    Returns (per-cell CI DataFrame, dataset-level quantile CI DataFrame).
    """
    rng = np.random.default_rng(seed)
    N, K = A.shape

    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=T)
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
    row_class = codes[order]
    row_starts = starts[row_class].astype(np.int64)
    row_counts = counts[row_class].astype(np.int64)

    flat_len = T * K
    if cell_index is None:
        cell_index = np.arange(flat_len)
    cell_index = np.asarray(cell_index, dtype=np.int64)

    cells = np.empty((B, cell_index.size), dtype=np.float32)
    qkeys = [f"Q{int(round(q*100))}" for q in quantiles]
    qsamples = np.empty((B, len(quantiles)), dtype=np.float64)
    zero_frac = np.empty(B, dtype=np.float64)

    for b in range(B):
        u = rng.random(N)
        pick = order[row_starts + (u * row_counts).astype(np.int64)]
        C_b, *_ = coverage_matrices(A[pick], None if M is None else M[pick],
                                    row_class, T, use_mask=use_mask, presorted=True)
        flat = C_b.reshape(-1)
        cells[b] = flat[cell_index]
        valid = flat[np.isfinite(flat)]
        qsamples[b] = [np.quantile(valid, q) for q in quantiles]
        zero_frac[b] = float((valid == 0).mean())
        if verbose and B >= 100 and (b + 1) % max(1, B // 5) == 0:
            print(f"    bootstrap {b+1}/{B}")

    lo = np.nanquantile(cells, alpha / 2, axis=0)
    hi = np.nanquantile(cells, 1 - alpha / 2, axis=0)
    se = np.nanstd(cells, axis=0)

    C_obs, *_ = coverage_matrices(A, M, codes, T, use_mask=use_mask)
    cell_df = pd.DataFrame({
        "flat_index": cell_index,
        "class_idx": cell_index // K,
        "concept_idx": cell_index % K,
        "coverage": C_obs.reshape(-1)[cell_index],
        "ci_lo": lo, "ci_hi": hi, "boot_se": se,
        "ci_width": hi - lo,
        "n_bootstrap": B,
    })

    qdf = pd.DataFrame({
        "statistic": qkeys + ["frac_zero"],
        "estimate": [float(np.quantile(C_obs.reshape(-1)[np.isfinite(C_obs.reshape(-1))], q))
                     for q in quantiles]
                    + [float((C_obs.reshape(-1)[np.isfinite(C_obs.reshape(-1))] == 0).mean())],
        "ci_lo": list(np.quantile(qsamples, alpha / 2, axis=0))
                 + [float(np.quantile(zero_frac, alpha / 2))],
        "ci_hi": list(np.quantile(qsamples, 1 - alpha / 2, axis=0))
                 + [float(np.quantile(zero_frac, 1 - alpha / 2))],
        "n_bootstrap": B,
    })
    return cell_df, qdf


def select_illustrative_cells(C: np.ndarray, n_per_band: int = 6, seed: int = 0):
    """Pick a spread of Tail / Middle / Head cells for CI visualisation."""
    rng = np.random.default_rng(seed)
    flat = C.reshape(-1)
    finite = np.flatnonzero(np.isfinite(flat))
    if finite.size == 0:
        return np.array([], dtype=np.int64)
    q25, q75 = np.nanquantile(flat, 0.25), np.nanquantile(flat, 0.75)
    bands = {
        "Tail": finite[flat[finite] <= q25],
        "Middle": finite[(flat[finite] > q25) & (flat[finite] < q75)],
        "Head": finite[flat[finite] >= q75],
    }
    out = []
    for ix in bands.values():
        if ix.size:
            out.append(rng.choice(ix, size=min(n_per_band, ix.size), replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=np.int64)