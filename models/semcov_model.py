"""
models/semcov_model.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .concept_router import VisualConceptRouter, attention_to_maps
from .encoders.base_encoder import BaseVisualEncoder
from .heads import (MeanSemanticPool, ResidualProjection, SemanticAttentionPool,
                    SharedConceptHead, build_classifier)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SemCovConfig:
    # branches
    use_router: bool = True
    use_concept_loss: bool = True
    use_residual: bool = True
    use_semantic: bool = True

    # dimensions
    d_sem: int = 256                 # shared latent width

    # router
    router_layers: int = 1
    router_heads: int = 8
    router_ffn_ratio: int = 4
    router_dropout: float = 0.0
    router_pre_norm: bool = False
    query_init: str = "normal"

    # heads
    concept_head_per_concept: bool = False   # shared scalar head by default
    concept_head_hidden: int = 0
    semantic_pool: str = "attn"              # 'attn' | 'mean'
    semantic_pool_heads: int = 8
    classifier: str = "linear"               # 'linear' | 'mlp'
    classifier_hidden: int = 512
    dropout: float = 0.1

    # loss weighting (used by the train script, kept here for checkpointing)
    lambda_concept: float = 1.0

    # what forward() returns by default
    return_patch_tokens: bool = False
    return_attention: bool = False
    return_tokens: bool = False


MODEL_CONFIGS: Dict[str, Dict] = {
    "dino_erm": dict(use_router=False, use_concept_loss=False,
                     use_residual=True, use_semantic=False),
    "dino_concept_only": dict(use_router=True, use_concept_loss=True,
                              use_residual=False, use_semantic=True),
    "dino_concept_residual": dict(use_router=True, use_concept_loss=True,
                                  use_residual=True, use_semantic=True),
    "dino_concept_no_auxloss": dict(use_router=True, use_concept_loss=False,
                                    use_residual=True, use_semantic=True),
}

# Aliases used in the experimental checkpoint.
MODEL_ALIASES = {"B0": "dino_erm",
                 "B1": "dino_concept_no_auxloss",
                 "B2": "dino_concept_residual"}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class SemCovPhase7(nn.Module):
    """
    Parameters
    ----------
    encoder     : any BaseVisualEncoder (DINOv3Encoder in practice)
    n_classes   : task output dimension
    n_concepts  : K, dataset specific (0 disables the semantic branch)
    cfg         : SemCovConfig
    """

    def __init__(self, encoder: BaseVisualEncoder, n_classes: int,
                 n_concepts: int = 0, cfg: Optional[SemCovConfig] = None):
        super().__init__()
        self.cfg = cfg or SemCovConfig()
        self.encoder = encoder
        self.n_classes = int(n_classes)
        self.K = int(n_concepts)

        c = self.cfg
        if c.use_router and self.K <= 0:
            raise ValueError("use_router=True requires n_concepts > 0 "
                             "(run the loaders with --use-meta).")
        if c.use_router and not getattr(encoder, "supports_patch_tokens", False):
            raise ValueError(
                f"{type(encoder).__name__} does not expose patch tokens, so "
                f"the Visual Concept Router cannot run. Use variant='dinov3'.")
        if not (c.use_residual or c.use_semantic):
            raise ValueError("At least one of residual / semantic must be on.")

        d = c.d_sem
        joint_dim = 0

        # ── residual branch ───────────────────────────
        if c.use_residual:
            self.residual_proj = ResidualProjection(encoder.global_dim, d,
                                                    dropout=0.0)
            joint_dim += d
        else:
            self.residual_proj = None

        # ── semantic branch ────────────────────────────────
        if c.use_router:
            self.router = VisualConceptRouter(
                in_dim=encoder.embed_dim, dim=d, n_concepts=self.K,
                n_heads=c.router_heads, n_layers=c.router_layers,
                ffn_ratio=c.router_ffn_ratio, dropout=c.router_dropout,
                pre_norm=c.router_pre_norm, query_init=c.query_init)
            self.concept_head = (
                SharedConceptHead(d, self.K,
                                  per_concept=c.concept_head_per_concept,
                                  hidden=c.concept_head_hidden)
                if c.use_concept_loss else None)
            if c.use_semantic:
                self.semantic_pool = (
                    SemanticAttentionPool(d, c.semantic_pool_heads)
                    if c.semantic_pool == "attn" else MeanSemanticPool(d))
                joint_dim += d
            else:
                self.semantic_pool = None
        else:
            self.router = None
            self.concept_head = None
            self.semantic_pool = None

        self.joint_dim = joint_dim
        self.classifier = build_classifier(c.classifier, joint_dim, n_classes,
                                           c.classifier_hidden, c.dropout)

        # remember which encoder params were trainable, for warm-up toggling
        self._encoder_trainable = {n: p.requires_grad
                                   for n, p in self.encoder.named_parameters()}
        print(f"[model] joint_dim={joint_dim} (residual={c.use_residual}, "
              f"semantic={c.use_semantic})  K={self.K}  classes={n_classes}  "
              f"trainable(non-encoder)={self.n_trainable_head()/1e6:.2f}M")

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                return_patch_tokens: Optional[bool] = None,
                return_attention: Optional[bool] = None,
                return_tokens: Optional[bool] = None) -> Dict[str, object]:
        """
        Returns a dict ("forward outputs"):

            task_logits       [B, C]         always
            concept_logits    [B, K]         when a concept head exists
            concept_tokens    [B, K, d]      when return_tokens
            concept_attention [B, K, P]      when return_attention
            semantic_feature  [B, d]         when the semantic branch is on
            residual_feature  [B, d]         when the residual branch is on
            joint_feature     [B, joint_dim] always
            patch_tokens      [B, P, D]      when return_patch_tokens
            cls_token         [B, D]         always
            grid_size         (h, w)

        Never returns only the task logits — the probes, SCI representation
        analysis and attention visualisations in later all read
        from here.
        """
        c = self.cfg
        want_patches = c.return_patch_tokens if return_patch_tokens is None \
            else return_patch_tokens
        want_attn = c.return_attention if return_attention is None \
            else return_attention
        want_tokens = c.return_tokens if return_tokens is None \
            else return_tokens

        enc = self.encoder(x)

        out: Dict[str, object] = {
            "cls_token": enc.cls_token,
            "grid_size": enc.grid_size,
            "patch_tokens": enc.patch_tokens if want_patches else None,
            "concept_logits": None,
            "concept_tokens": None,
            "concept_attention": None,
            "semantic_feature": None,
            "residual_feature": None,
        }

        parts: List[torch.Tensor] = []

        if self.residual_proj is not None:
            z_res = self.residual_proj(enc.global_feature)
            out["residual_feature"] = z_res
            parts.append(z_res)

        if self.router is not None:
            Z, attn = self.router(enc.patch_tokens, return_attention=want_attn)
            if want_tokens:
                out["concept_tokens"] = Z
            if want_attn:
                out["concept_attention"] = attn
            if self.concept_head is not None:
                out["concept_logits"] = self.concept_head(Z)
            if self.semantic_pool is not None:
                z_sem, pool_w = self.semantic_pool(Z,
                                                   return_attention=want_attn)
                out["semantic_feature"] = z_sem
                if want_attn and pool_w is not None:
                    out["semantic_pool_weights"] = pool_w
                parts.append(z_sem)

        h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        out["joint_feature"] = h
        out["task_logits"] = self.classifier(h)
        return out

    # ── convenience ─────────────────────────────────────────────────────
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["task_logits"]

    def concept_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        o = self.forward(x, return_attention=True)
        return attention_to_maps(o["concept_attention"], o["grid_size"])

    # ── optimizer bookkeeping (DINOv3) ─────────────────────────
    def head_parameters(self) -> List[nn.Parameter]:
        enc_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in enc_ids]

    def n_trainable_head(self) -> int:
        return sum(p.numel() for p in self.head_parameters())

    def param_groups(self, lr_semcov: float = 3e-4,
                     lr_lora: Optional[float] = None,
                     lr_dino: Optional[float] = None,
                     weight_decay: float = 0.05) -> List[dict]:
        """
        Group 1: new SemCovNet parameters      eta_SemCov
        Group 2: LoRA parameters               ~ eta_SemCov
        Group 3: unfrozen DINOv3 blocks        eta_DINO << eta_SemCov

        Defaults follow the plan: LoRA at the SemCovNet rate, fully unfrozen
        DINOv3 blocks at 0.05x-0.1x of it.
        """
        groups = [{"name": "semcov", "params": self.head_parameters(),
                   "lr": lr_semcov, "weight_decay": weight_decay}]
        enc_groups = self.encoder.param_groups() \
            if hasattr(self.encoder, "param_groups") else {}
        if enc_groups.get("lora"):
            groups.append({"name": "lora", "params": enc_groups["lora"],
                           "lr": lr_lora if lr_lora is not None else lr_semcov,
                           "weight_decay": 0.0})
        if enc_groups.get("dino"):
            groups.append({"name": "dino", "params": enc_groups["dino"],
                           "lr": lr_dino if lr_dino is not None
                           else 0.05 * lr_semcov,
                           "weight_decay": weight_decay})
        for g in groups:
            print(f"[optim] group '{g['name']}': "
                  f"{sum(p.numel() for p in g['params'])/1e6:.2f}M params, "
                  f"lr={g['lr']:.2e}")
        return [g for g in groups if len(g["params"]) > 0]

    def set_encoder_trainable(self, flag: bool) -> None:
        """
        Two-stage schedule (DINOv3): semantic warm-up with the
        encoder frozen, then enable LoRA / last-N adaptation.
        """
        for n, p in self.encoder.named_parameters():
            p.requires_grad = bool(flag) and self._encoder_trainable.get(n, False)


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_name: str, encoder: BaseVisualEncoder, n_classes: int,
                n_concepts: int = 0, **cfg_overrides) -> SemCovPhase7:
    """
    build_model('dino_concept_residual', encoder, n_classes=2, n_concepts=39,
                d_sem=256, classifier='linear', lambda_concept=1.0)
    """
    name = MODEL_ALIASES.get(model_name, model_name)
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from {list(MODEL_CONFIGS)} "
                         f"(aliases {list(MODEL_ALIASES)})")
    cfg = SemCovConfig(**MODEL_CONFIGS[name])
    for k, v in cfg_overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown SemCovConfig field: {k}")
        setattr(cfg, k, v)
    print(f"[model] building '{name}'")
    return SemCovPhase7(encoder, n_classes, n_concepts, cfg)
