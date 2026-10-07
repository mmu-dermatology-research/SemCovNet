"""
models/heads.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResidualProjection(nn.Module):
    """z_i^res = LN(W_r g_i) in R^{d_sem}, from g_i = [f_cls; mean patch]."""

    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.proj(g)))


class SharedConceptHead(nn.Module):
    """
    One scalar per concept token.

    Shared (default): a single (w_c, b_c) applied to every z_{i,k}. Concept
    identity is already carried by the query, so per-concept heads mostly add
    capacity differences between concepts — kept available as an ablation via
    `per_concept=True`.
    """

    def __init__(self, dim: int = 256, n_concepts: int = 1,
                 per_concept: bool = False, hidden: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        self.per_concept = per_concept
        self.K = int(n_concepts)

        self.pre = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout)) \
            if dropout > 0 else nn.LayerNorm(dim)

        if hidden and hidden > 0:
            self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
            dim = hidden
        else:
            self.trunk = nn.Identity()

        if per_concept:
            self.weight = nn.Parameter(torch.zeros(self.K, dim))
            self.bias = nn.Parameter(torch.zeros(self.K))
            nn.init.trunc_normal_(self.weight, std=0.02)
        else:
            self.linear = nn.Linear(dim, 1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: [B, K, d] -> concept logits R_i: [B, K]."""
        h = self.trunk(self.pre(Z))
        if self.per_concept:
            return torch.einsum("bkd,kd->bk", h, self.weight) + self.bias
        return self.linear(h).squeeze(-1)


class SemanticAttentionPool(nn.Module):
    """
    z_i^sem = AttnPool(q_sem, Z_i) with one learnable pooling query.

    Attention pooling rather than a mean because different concepts matter
    differently per image and per task. Deliberately *not* coverage-weighted:
    coverage enters only via CCSS later.
    """

    def __init__(self, dim: int = 256, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.q_sem = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.q_sem, std=0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, Z: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Z: [B,K,d] -> z_sem: [B,d] (+ [B,K] pooling weights)."""
        B = Z.shape[0]
        q = self.q_sem.expand(B, -1, -1)
        out, w = self.attn(q, Z, Z, need_weights=return_attention,
                           average_attn_weights=True)
        z_sem = self.norm(out.squeeze(1))
        return z_sem, (w.squeeze(1) if (return_attention and w is not None)
                       else None)


class MeanSemanticPool(nn.Module):
    """Ablation pooler: plain mean over concept tokens."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, Z: torch.Tensor, return_attention: bool = False):
        return self.norm(Z.mean(dim=1)), None


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if layer_norm else nn.Identity()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(h))


class MLPClassifier(nn.Module):
    """LN -> Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def build_classifier(kind: str, in_dim: int, out_dim: int,
                     hidden: int = 512, dropout: float = 0.1) -> nn.Module:
    kind = (kind or "linear").lower()
    if kind == "linear":
        return LinearClassifier(in_dim, out_dim)
    if kind == "mlp":
        return MLPClassifier(in_dim, out_dim, hidden, dropout)
    raise ValueError(f"Unknown classifier '{kind}' (linear | mlp)")
