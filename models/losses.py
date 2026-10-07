"""
models/losses.py
─────────────────────────────────────────────────────────────────────────────
semantic concept supervision.

    L = L_task + lambda_concept * L_concept

with,

    L_concept = sum_{i,k} M_{i,k} l_{i,k} / sum_{i,k} M_{i,k}

i.e. always an *observation-weighted mean*, never a sum. K ranges from 7
(MILK10k) to 312 (CUB); summing would make one lambda_concept mean four
completely different things across datasets.

Three supervision modes
───────────────────────────────────
    'bce'        CelebA, CUB          a_{i,k} in {0,1}      masked BCE
    'soft_bce'   MILK10k              a_{i,k} in [0,1]      soft-target BCE
                                      (MONET scores are never thresholded in
                                       the primary protocol)
    'family_ce'  Derm7pt              one-hot within each clinical criterion
                                      -> CE over the family's logits,
                                      averaged across observed families;
                                      any concept outside a multi-member
                                      family falls back to masked BCE.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CONCEPT_LOSS_MODES = ("bce", "soft_bce", "family_ce", "mse", "none")


def mask_from_targets(a: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                                    torch.Tensor]:
    """(targets_with_nan, optional mask) -> (clean_targets, float mask)."""
    finite = torch.isfinite(a)
    m = finite.float() if mask is None else (mask.float() * finite.float())
    return torch.nan_to_num(a, nan=0.0), m


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor,
                           pos_weight: Optional[torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (sum of masked element losses, number of observed labels).
    Works unchanged for hard {0,1} and soft [0,1] targets.
    """
    l = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=pos_weight.to(logits.dtype) if pos_weight is not None else None)
    return (l * mask).sum(), mask.sum()


def masked_mse(logits: torch.Tensor, targets: torch.Tensor,
               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    l = (torch.sigmoid(logits) - targets) ** 2
    return (l * mask).sum(), mask.sum()


def family_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor,
                         families: Sequence[Sequence[int]]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CE within each family of mutually exclusive concept values.

    A (sample, family) pair counts as observed when every member is observed
    and the family's targets form a valid distribution (sum > 0); the target
    distribution is normalised, so both one-hot and soft annotations work.

    Returns (sum of per-(sample, family) CE, number of observed pairs).
    """
    total = logits.new_zeros(())
    n_obs = logits.new_zeros(())
    for idx in families:
        if len(idx) < 2:
            continue
        cols = torch.as_tensor(list(idx), device=logits.device, dtype=torch.long)
        lg = logits.index_select(1, cols)              # [B, |F|]
        tg = targets.index_select(1, cols)
        mk = mask.index_select(1, cols)

        observed = (mk.min(dim=1).values > 0) & (tg.sum(dim=1) > 0)
        if not bool(observed.any()):
            continue
        p = tg / tg.sum(dim=1, keepdim=True).clamp_min(1e-8)
        ce = -(p * F.log_softmax(lg, dim=1)).sum(dim=1)  # [B]
        ce = ce * observed.float()
        total = total + ce.sum()
        n_obs = n_obs + observed.float().sum()
    return total, n_obs


class ConceptLoss(nn.Module):
    """
    concept objective.

    Parameters
    ----------
    mode      : 'bce' | 'soft_bce' | 'family_ce' | 'mse' | 'none'
    families  : list of index lists (only used by 'family_ce')
    pos_weight: optional [K] positive weighting for very rare binary concepts.
                Off by default — reweighting concepts by frequency is exactly
                the kind of coverage-aware mechanism must NOT contain.
    """

    def __init__(self, mode: str = "bce",
                 families: Optional[Sequence[Sequence[int]]] = None,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        mode = (mode or "none").lower()
        if mode not in CONCEPT_LOSS_MODES:
            raise ValueError(f"Unknown concept loss mode '{mode}'. "
                             f"Choose from {CONCEPT_LOSS_MODES}")
        self.mode = mode
        self.families: List[List[int]] = [list(f) for f in (families or [])]
        self._family_members = sorted({k for f in self.families
                                       for k in f if len(f) >= 2})
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: Optional[torch.Tensor],
                targets: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "none" or logits is None or targets is None \
                or targets.numel() == 0:
            return logits.new_zeros(()) if logits is not None \
                else torch.zeros((), requires_grad=False)

        targets, m = mask_from_targets(targets.float(), mask)
        targets = targets.to(logits.dtype)
        m = m.to(logits.dtype)

        if self.mode in ("bce", "soft_bce"):
            s, n = masked_bce_with_logits(logits, targets, m, self.pos_weight)
        elif self.mode == "mse":
            s, n = masked_mse(logits, targets, m)
        else:                                   # family_ce (+ BCE leftovers)
            s, n = family_cross_entropy(logits, targets, m, self.families)
            leftover = [k for k in range(logits.shape[1])
                        if k not in self._family_members]
            if leftover:
                cols = torch.as_tensor(leftover, device=logits.device,
                                       dtype=torch.long)
                s2, n2 = masked_bce_with_logits(
                    logits.index_select(1, cols),
                    targets.index_select(1, cols),
                    m.index_select(1, cols))
                s, n = s + s2, n + n2

        return s / n.clamp_min(1.0)


def build_concept_loss(spec) -> ConceptLoss:
    """`spec` is a concept_meta.ConceptSpec (duck-typed here to avoid a cycle)."""
    return ConceptLoss(mode=getattr(spec, "loss_mode", "bce"),
                       families=getattr(spec, "family_indices", None))
