"""
models/ccss/shared_structure.py
─────────────────────────────────────────────────────────────────────────────
Shared Semantic Representation Decomposition.

    m_{y,k} = mu_0 + mu~_y + mu~_k                       (corrected core)

    mu~_y = mu_y - (1/T) sum_{y'} mu_{y'}
    mu~_k = mu_k - (1/K) sum_{k'} mu_{k'}

Three learnable parameter blocks:

    mu_0            in R^D          global semantic representation
    M^class         in R^{T x D}    one vector per class
    M^concept       in R^{K x D}    one vector per atomic concept

CCSS mixes m_{y,k} with the empirical prototype z_bar_{y,k}:

    t_hat = m + alpha (z_bar - m)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

SHARED_MODES = ("concept", "class_concept", "global_class_concept")
SHARED_NORMS = ("layernorm", "l2", "none")


@dataclass
class SharedStructureConfig:
    """`mode` is the ablation switch."""
    mode: str = "global_class_concept"
    normalize: str = "none"                # corrected primary path (was 'layernorm')
    init_std: float = 0.02
    zero_init_global: bool = True          # mu_0 initialised near zero
    layernorm_affine: bool = False         # pure normalisation by default
    center: bool = True                    # identifiability centering (corrected)


class SharedSemanticStructure(nn.Module):
    """
    Parameters
    ----------
    n_classes  : T
    n_concepts : K
    dim        : D (256 recommended)
    cfg        : SharedStructureConfig

    Shapes
    ------
    m_all()             -> [T, K, D]
    m_batch(y)          -> [B, K, D]     y: LongTensor[B]
    m_pairs(y, k)       -> [N, D]
    """

    def __init__(self, n_classes: int, n_concepts: int, dim: int = 256,
                 cfg: Optional[SharedStructureConfig] = None):
        super().__init__()
        cfg = cfg or SharedStructureConfig()
        if cfg.mode not in SHARED_MODES:
            raise ValueError(f"Unknown shared-structure mode '{cfg.mode}'. "
                             f"Choose from {SHARED_MODES}")
        if cfg.normalize not in SHARED_NORMS:
            raise ValueError(f"Unknown normalize '{cfg.normalize}'. "
                             f"Choose from {SHARED_NORMS}")
        self.cfg = cfg
        self.T, self.K, self.D = int(n_classes), int(n_concepts), int(dim)

        self.use_global = cfg.mode == "global_class_concept"
        self.use_class = cfg.mode in ("class_concept", "global_class_concept")

        # mu_0 : global semantic representation, initialised near zero
        if self.use_global:
            self.mu_0 = nn.Parameter(torch.zeros(self.D))
            if not cfg.zero_init_global:
                nn.init.trunc_normal_(self.mu_0, std=cfg.init_std)
        else:
            self.register_parameter("mu_0", None)

        # M^class : [T, D]
        if self.use_class:
            self.mu_class = nn.Parameter(torch.empty(self.T, self.D))
            nn.init.trunc_normal_(self.mu_class, std=cfg.init_std)
        else:
            self.register_parameter("mu_class", None)

        # M^concept : [K, D] — always present
        self.mu_concept = nn.Parameter(torch.empty(self.K, self.D))
        nn.init.trunc_normal_(self.mu_concept, std=cfg.init_std)

        self.norm = (nn.LayerNorm(self.D, elementwise_affine=cfg.layernorm_affine)
                     if cfg.normalize == "layernorm" else None)

    # ── construction ────────────────────────────────────────────────────
    def _normalize(self, m: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalize == "layernorm":
            return self.norm(m)
        if self.cfg.normalize == "l2":
            return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return m

    # ── identifiability centering ───────────────────────────────────────
    # mu~_y = mu_y - mean_{y'} mu_{y'},  mu~_k = mu_k - mean_{k'} mu_{k'}.
    # Both means are over the FULL class / concept axis, never over the
    # subset an individual call happens to index, so m_batch(y) and
    # m_pairs(y, k) agree with m_all()[y, k] exactly.
    def _mu_class_c(self) -> torch.Tensor:
        """[T, D] class effects, centered when cfg.center."""
        mu = self.mu_class
        return mu - mu.mean(dim=0, keepdim=True) if self.cfg.center else mu

    def _mu_concept_c(self) -> torch.Tensor:
        """[K, D] concept effects, centered when cfg.center."""
        mu = self.mu_concept
        return mu - mu.mean(dim=0, keepdim=True) if self.cfg.center else mu

    def m_all(self) -> torch.Tensor:
        """[T, K, D] — every shared class-concept prior. Used once per epoch
        for the sigma_k^2 / tau^2 refresh, not in the inner training loop."""
        m = self._mu_concept_c().unsqueeze(0).expand(self.T, self.K, self.D)
        if self.use_class:
            m = m + self._mu_class_c().unsqueeze(1)
        if self.use_global:
            m = m + self.mu_0.view(1, 1, self.D)
        return self._normalize(m)

    def m_batch(self, y_idx: torch.Tensor) -> torch.Tensor:
        """
        [B, K, D] for the classes in a minibatch — the only shared-structure
        call on the hot path. Gradients flow to mu_0 / mu_y / mu_k from here.
        """
        m = self._mu_concept_c().unsqueeze(0)                 # [1, K, D]
        if self.use_class:
            m = m + self._mu_class_c().index_select(0, y_idx).unsqueeze(1)
        else:
            m = m.expand(y_idx.shape[0], self.K, self.D)
        if self.use_global:
            m = m + self.mu_0.view(1, 1, self.D)
        return self._normalize(m.contiguous())

    def m_pairs(self, y_idx: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
        """[N, D] for an arbitrary list of (y, k) pairs."""
        m = self._mu_concept_c().index_select(0, k_idx)
        if self.use_class:
            m = m + self._mu_class_c().index_select(0, y_idx)
        if self.use_global:
            m = m + self.mu_0.view(1, self.D)
        return self._normalize(m)

    def forward(self, y_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.m_all() if y_idx is None else self.m_batch(y_idx)

    # ── diagnostics ───────────────────────────────────────────
    @torch.no_grad()
    def concept_similarity(self) -> torch.Tensor:
        """[K, K] cosine similarity between the mu_k."""
        e = self.mu_concept
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return e @ e.t()

    @torch.no_grad()
    def class_similarity(self) -> Optional[torch.Tensor]:
        if not self.use_class:
            return None
        e = self.mu_class
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return e @ e.t()

    @torch.no_grad()
    def nearest_concepts(self, k: int, topn: int = 5):
        """Indices + similarities of the `topn` concepts closest to concept k
        (concept retrieval / nearest-embedding diagnostic)."""
        sim = self.concept_similarity()[k].clone()
        sim[k] = -2.0
        v, i = torch.topk(sim, min(topn, self.K - 1))
        return i.tolist(), v.tolist()

    @torch.no_grad()
    def collapse_report(self) -> Dict[str, float]:
        """
        The go/no-go check. `offdiag_mean` near 1.0 means the concept
        embeddings have collapsed onto one vector (failure mode) — inspect
        concept supervision and token diversity before adding a regulariser.
        """
        sim = self.concept_similarity()
        off = sim[~torch.eye(self.K, dtype=torch.bool, device=sim.device)]
        out = {
            "mu_concept_norm": float(self.mu_concept.norm(dim=-1).mean()),
            "mu_concept_offdiag_cos_mean": float(off.mean()),
            "mu_concept_offdiag_cos_std": float(off.std()) if off.numel() > 1
            else float("nan"),
            "mu_concept_offdiag_cos_max": float(off.max()) if off.numel()
            else float("nan"),
        }
        # after centering, "collapsed" means the deviations themselves vanish,
        # so the norm of mu~_k is the quantity to watch alongside the cosine
        out["mu_concept_centered_norm"] = float(
            self._mu_concept_c().norm(dim=-1).mean())
        if self.use_class:
            out["mu_class_norm"] = float(self.mu_class.norm(dim=-1).mean())
            out["mu_class_centered_norm"] = float(
                self._mu_class_c().norm(dim=-1).mean())
        if self.use_global:
            out["mu_0_norm"] = float(self.mu_0.norm())
        return out

    def extra_repr(self) -> str:
        return (f"mode={self.cfg.mode}, normalize={self.cfg.normalize}, "
                f"center={self.cfg.center}, "
                f"T={self.T}, K={self.K}, D={self.D}")
