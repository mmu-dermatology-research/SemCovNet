"""
models/concept_router.py
─────────────────────────────────────────────────────────────────────────────
Visual Concept Router.

Attention maps
──────────────
`forward(..., return_attention=True)` returns head-averaged A_i in
R^{K x P}; `attention_to_maps()` reshapes P -> (h, w) for visualisation and
for checking query collapse. Localisation is *not* a training objective at
this stage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: queries [B,K,d] attend to patches [B,P,d].

    Uses the fused SDPA kernel when attention weights are not requested and
    falls back to explicit softmax when they are, so returning attention maps
    costs memory only when you ask for it (relevant for CUB: K=312, P=196,
    8 heads).
    """

    def __init__(self, dim: int = 256, n_heads: int = 8, dropout: float = 0.0,
                 qkv_bias: bool = True):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.dim, self.n_heads = dim, n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = dropout
        self.proj_drop = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # [B, N, d] -> [B, H, N, hd]
        return x.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, queries: torch.Tensor, patches: torch.Tensor,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = queries.shape[0]
        q = self._shape(self.q_proj(queries), B)      # [B,H,K,hd]
        k = self._shape(self.k_proj(patches), B)      # [B,H,P,hd]
        v = self._shape(self.v_proj(patches), B)

        attn = None
        if return_attention:
            scores = (q @ k.transpose(-2, -1)) * self.scale      # [B,H,K,P]
            probs = scores.softmax(dim=-1)
            out = probs @ v
            attn = probs.mean(dim=1)                             # [B,K,P]
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop if self.training else 0.0)

        out = out.transpose(1, 2).reshape(B, -1, self.dim)       # [B,K,d]
        return self.proj_drop(self.out_proj(out)), attn


class ConceptRouterBlock(nn.Module):
    """
    Transformer-decoder style block:

        Q -> CrossAttention(Q, F) -> Residual+Norm -> FFN -> Residual+Norm

    `pre_norm=False` reproduces the post-norm formulation written in the
    plan; `pre_norm=True` is the usually-more-stable variant if you later see
    training instability with a fine-tuned backbone.
    """

    def __init__(self, dim: int = 256, n_heads: int = 8, ffn_ratio: int = 4,
                 dropout: float = 0.0, pre_norm: bool = False):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = CrossAttention(dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, patches: torch.Tensor,
                return_attention: bool = False):
        if self.pre_norm:
            h, attn = self.attn(self.norm1(q), patches, return_attention)
            q = q + h
            q = q + self.ffn(self.norm2(q))
        else:
            h, attn = self.attn(q, patches, return_attention)
            q = self.norm1(q + h)
            q = self.norm2(q + self.ffn(q))
        return q, attn


class VisualConceptRouter(nn.Module):
    """
    module.

    Parameters
    ----------
    in_dim      : encoder token width (768 for DINOv3 ViT-B/16)
    dim         : semantic dimension d_sem (256 recommended)
    n_concepts  : K, dataset specific
    n_heads     : 8
    n_layers    : 1 (start here)
    ffn_ratio   : 2 or 4
    query_init  : 'normal' | 'orthogonal'

    Returns
    -------
    Z    : [B, K, d]
    attn : [B, K, P] head-averaged attention of the last block, or None
    """

    def __init__(self, in_dim: int = 768, dim: int = 256, n_concepts: int = 1,
                 n_heads: int = 8, n_layers: int = 1, ffn_ratio: int = 4,
                 dropout: float = 0.0, pre_norm: bool = False,
                 query_init: str = "normal"):
        super().__init__()
        self.in_dim, self.dim, self.K = in_dim, dim, int(n_concepts)

        # W_v : 768 -> 256 patch projection
        self.patch_proj = nn.Sequential(nn.Linear(in_dim, dim),
                                        nn.LayerNorm(dim))

        # Q = [q_1 .. q_K] : the dataset-specific query bank
        self.queries = nn.Parameter(torch.empty(self.K, dim))
        if query_init == "orthogonal" and self.K <= dim:
            nn.init.orthogonal_(self.queries)
        else:
            nn.init.trunc_normal_(self.queries, std=0.02)

        self.blocks = nn.ModuleList([
            ConceptRouterBlock(dim, n_heads, ffn_ratio, dropout, pre_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    @torch.no_grad()
    def init_queries_from(self, embeddings: torch.Tensor) -> None:
        """
        Optional appendix ablation: initialise q_k from text embeddings
        (e.g. CLIP concept-name features), projected/truncated to `dim`.
        """
        if embeddings.shape[0] != self.K:
            raise ValueError(f"expected {self.K} embeddings, "
                             f"got {embeddings.shape[0]}")
        e = embeddings.float()
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
        if e.shape[1] >= self.dim:
            self.queries.copy_(e[:, :self.dim])
        else:
            self.queries[:, :e.shape[1]].copy_(e)

    def forward(self, patch_tokens: torch.Tensor,
                return_attention: bool = False):
        if patch_tokens is None:
            raise ValueError(
                "VisualConceptRouter needs dense patch tokens; the selected "
                "encoder returned none (supports_patch_tokens=False).")
        B = patch_tokens.shape[0]
        F_tilde = self.patch_proj(patch_tokens)                  # [B,P,d]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)          # [B,K,d]

        attn = None
        for blk in self.blocks:
            q, a = blk(q, F_tilde, return_attention)
            if a is not None:
                attn = a
        return self.norm(q), attn


def attention_to_maps(attn: torch.Tensor,
                      grid_size: Tuple[int, int]) -> torch.Tensor:
    """[B,K,P] -> [B,K,h,w] (196 -> 14x14) for qualitative localisation."""
    h, w = grid_size
    B, K, P = attn.shape
    if h * w != P:
        raise ValueError(f"grid {h}x{w} does not match {P} patches")
    return attn.view(B, K, h, w)


@torch.no_grad()
def query_similarity(router: VisualConceptRouter) -> torch.Tensor:
    """
    Cosine similarity between concept queries — the cheap collapse check.
    A near-all-ones matrix means the K queries have collapsed onto one
    concept.
    """
    q = router.queries
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    return q @ q.t()
