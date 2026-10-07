"""
models/encoders/dinov3_encoder.py
─────────────────────────────────────────────────────────────────────────────
DINOv3 ViT-B/16 (LVD-1689M) wrapper — the visual encoder.

    hub id : facebook/dinov3-vitb16-pretrain-lvd1689m
    tokens : 1 CLS + 4 register + 196 patch   (at 224x224, patch 16)
    dim    : 768,  depth 12,  heads 12

Requires: pip install "transformers>=4.56" (DINOv3 support) and access to
the HF hub (or a local snapshot path passed as `weights`).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base_encoder import BaseVisualEncoder, EncoderOutput
from .lora import inject_lora, lora_parameters, resolve_lora_targets

token = "<HF hub token for DINOv3 LVD-1689M>"  # HF hub token for DINOv3 LVD-1689M
DINOV3_VITB16_LVD = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# DINOv3 uses the standard ImageNet statistics.
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)

_PRECISION = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


# ─────────────────────────────────────────────────────────────────────────────
# Config  (DINOv3 plan §21)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoRAConfig:
    enabled: bool = False
    last_n_blocks: int = 3          # blocks 9, 10, 11 for a 12-block ViT-B
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_attention: bool = True
    target_mlp: bool = True


@dataclass
class DINOv3Config:
    encoder_name: str = "dinov3_vitb16"
    weights: str = DINOV3_VITB16_LVD      # hub id or local snapshot dir
    image_size: int = 224

    # 'cls' | 'patch' | 'cls_patch' | 'intermediate'
    feature_mode: str = "cls_patch"

    # 'frozen' | 'lora' | 'partial' | 'full'  (+ aliases, see _normalize_mode)
    train_mode: str = "frozen"
    unfreeze_last_n: int = 3              # used by train_mode='partial'
    train_final_norm: bool = True         # train `norm` alongside late blocks

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    return_cls: bool = True
    return_patch_tokens: bool = True
    return_intermediate: bool = False
    intermediate_layers: Sequence[int] = (8, 9, 10, 11)

    # 'cls' | 'mean_patch' | 'cls_mean'   (§15, cls_mean recommended)
    global_pool: str = "cls_mean"

    precision: str = "bf16"               # autocast dtype for the backbone
    cast_frozen_weights: bool = True      # store frozen weights in `precision`
    gradient_checkpointing: bool = False

    def resolved_train_mode(self) -> str:
        return _normalize_mode(self.train_mode, self.lora)


def _normalize_mode(mode: str, lora_cfg: Optional[LoRAConfig] = None) -> str:
    m = (mode or "frozen").lower().strip()
    if m in ("frozen", "freeze", "none"):
        return "frozen"
    if m in ("lora", "lora_last3", "lora3", "lora_last4"):
        if lora_cfg is not None:
            lora_cfg.enabled = True
            if m == "lora_last4":
                lora_cfg.last_n_blocks = 4
        return "lora"
    if m in ("partial", "last3", "full_last3", "last4", "full_last4",
             "unfreeze_last_n"):
        return "partial"
    if m in ("full", "full_ft", "finetune"):
        return "full"
    raise ValueError(
        f"Unknown train_mode '{mode}'. Choose from: frozen | lora | partial | "
        f"full (aliases: lora_last3, full_last3, last4, full_ft)")


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────
class DINOv3Encoder(BaseVisualEncoder):
    supports_patch_tokens = True

    def __init__(self, cfg: Optional[DINOv3Config] = None,
                 hf_model: Optional[nn.Module] = None, **overrides):
        """
        `hf_model` lets you inject an already-constructed DINOv3ViTModel
        (shared weights across runs, a local snapshot, or a randomly
        initialised model for offline shape tests). When it is None the
        weights are fetched with AutoModel.from_pretrained(cfg.weights).
        """
        super().__init__()
        cfg = cfg or DINOv3Config()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"Unknown DINOv3Config field: {k}")
            setattr(cfg, k, v)
        self.cfg = cfg

        if hf_model is not None:
            self.backbone = hf_model
        else:
            try:
                from transformers import AutoModel
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "DINOv3 needs `transformers>=4.56`: "
                    "pip install -U transformers") from e
            self.backbone = AutoModel.from_pretrained(cfg.weights, token=token)

        hf = self.backbone.config
        self.embed_dim = int(getattr(hf, "hidden_size", 768))
        self.patch_size = _as_int(getattr(hf, "patch_size", 16))
        self.n_register = int(getattr(hf, "num_register_tokens", 0) or 0)
        self.n_prefix = 1 + self.n_register            # CLS + registers
        self.depth = int(getattr(hf, "num_hidden_layers", 12))

        if cfg.global_pool not in ("cls", "mean_patch", "cls_mean"):
            raise ValueError(f"Unknown global_pool '{cfg.global_pool}'")
        self.global_dim = self.embed_dim * (2 if cfg.global_pool == "cls_mean"
                                            else 1)

        self.autocast_dtype = _PRECISION.get(cfg.precision.lower())
        if self.autocast_dtype is None:
            raise ValueError(f"Unknown precision '{cfg.precision}'")

        self.train_mode = cfg.resolved_train_mode()
        self._apply_train_mode()

        if cfg.gradient_checkpointing and hasattr(self.backbone,
                                                  "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        print(f"[DINOv3] {cfg.weights}\n"
              f"         mode={self.train_mode}  dim={self.embed_dim}  "
              f"depth={self.depth}  registers={self.n_register}  "
              f"global_pool={cfg.global_pool}  precision={cfg.precision}\n"
              f"         trainable {self.n_trainable()/1e6:.2f}M / "
              f"{self.n_total()/1e6:.2f}M params")

    # ── block access ────────────────────────────────────────────────────
    def _blocks(self) -> nn.ModuleList:
        """
        Locate the transformer block list across transformers versions:
            v5: backbone.model.layer     (DINOv3ViTEncoder.layer)
            others: backbone.encoder.layer / backbone.layer / backbone.blocks
        """
        for path in ("model.layer", "encoder.layer", "layer", "blocks",
                     "encoder.layers", "model.layers"):
            obj = self.backbone
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and isinstance(obj, (nn.ModuleList, nn.Sequential)):
                return obj
        raise RuntimeError(
            "Could not locate DINOv3 transformer blocks on the HF model. "
            "Inspect `model` and extend DINOv3Encoder._blocks().")

    def _final_norm(self) -> Optional[nn.Module]:
        return getattr(self.backbone, "norm", None)

    # ── freezing / adaptation ───────────────────────────────────────────
    def _apply_train_mode(self) -> None:
        cfg = self.cfg
        mode = self.train_mode
        blocks = self._blocks()
        n_blocks = len(blocks)

        # start from fully frozen in every mode except 'full'
        for p in self.backbone.parameters():
            p.requires_grad = (mode == "full")

        self.lora_block_ids: List[int] = []
        self.trainable_block_ids: List[int] = []

        if mode == "frozen":
            if cfg.cast_frozen_weights and self.autocast_dtype != torch.float32:
                self.backbone.to(dtype=self.autocast_dtype)

        elif mode == "lora":
            n_last = max(1, min(int(cfg.lora.last_n_blocks), n_blocks))
            self.lora_block_ids = list(range(n_blocks - n_last, n_blocks))
            targets = resolve_lora_targets(cfg.lora.target_attention,
                                           cfg.lora.target_mlp)
            n_wrapped = 0
            for i in self.lora_block_ids:
                n_wrapped += inject_lora(blocks[i], targets,
                                         r=cfg.lora.rank, alpha=cfg.lora.alpha,
                                         dropout=cfg.lora.dropout)
            for p in lora_parameters(self.backbone):
                p.requires_grad = True
            print(f"[DINOv3] LoRA r={cfg.lora.rank} alpha={cfg.lora.alpha} "
                  f"-> blocks {self.lora_block_ids}, {n_wrapped} linear layers, "
                  f"targets={targets}")

        elif mode == "partial":
            n_last = max(0, min(int(cfg.unfreeze_last_n), n_blocks))
            self.trainable_block_ids = list(range(n_blocks - n_last, n_blocks))
            for i in self.trainable_block_ids:
                for p in blocks[i].parameters():
                    p.requires_grad = True
            if cfg.train_final_norm and self._final_norm() is not None:
                for p in self._final_norm().parameters():
                    p.requires_grad = True
            print(f"[DINOv3] full fine-tuning of blocks "
                  f"{self.trainable_block_ids}")

        # 'full' -> everything already requires_grad
        if mode != "frozen":
            # keep master weights in fp32; precision is handled by autocast
            self.backbone.to(dtype=torch.float32)

    def train(self, mode: bool = True):
        """A frozen backbone stays in eval() so dropout/droppath never fire."""
        super().train(mode)
        if self.train_mode == "frozen":
            self.backbone.eval()
        return self

    def param_groups(self) -> dict:
        lora_ps = lora_parameters(self.backbone)
        lora_ids = {id(p) for p in lora_ps}
        other = [p for p in self.backbone.parameters()
                 if p.requires_grad and id(p) not in lora_ids]
        groups = {}
        if lora_ps:
            groups["lora"] = lora_ps
        if other:
            groups["dino"] = other
        return groups

    # ── forward ─────────────────────────────────────────────────────────
    def _autocast(self):
        if self.autocast_dtype == torch.float32:
            return contextlib.nullcontext()
        if not torch.cuda.is_available():
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:
        cfg = self.cfg
        B, _, H, W = pixel_values.shape
        gh, gw = H // self.patch_size, W // self.patch_size

        want_hidden = cfg.return_intermediate or cfg.feature_mode == "intermediate"

        # Build the autograd graph when EITHER the backbone has trainable
        # parameters OR the caller handed us an input that requires grad.
        #
        # The second clause matters for input-gradient methods (the
        # memorization analysis accumulates ||d l_i / d x_i||^2). Without it,
        # a frozen backbone — including a LoRA run during its frozen warm-up
        # stage, where every adapter still has requires_grad=False — severs
        # the graph between `pixel_values` and the loss, and
        # `torch.autograd.grad(loss, images)` fails with "One of the
        # differentiated Tensors appears to not have been used in the graph".
        # Skipping the graph when nothing needs it is the right optimisation;
        # it was just testing the wrong condition.
        grad_on = self.has_trainable_params or pixel_values.requires_grad

        with torch.set_grad_enabled(grad_on and torch.is_grad_enabled()), self._autocast():
            if self.train_mode == "frozen" and cfg.cast_frozen_weights:
                pixel_values = pixel_values.to(
                    next(self.backbone.parameters()).dtype)
            out = self.backbone(pixel_values=pixel_values,
                                output_hidden_states=want_hidden)

        seq = out.last_hidden_state.float()          # [B, 1+R+P, D]
        cls = seq[:, 0]                              # [B, D]
        patches = seq[:, self.n_prefix:]             # [B, P, D]  registers dropped

        P = patches.shape[1]
        if gh * gw != P:
            # non-square / unexpected resolution — fall back to a square guess
            side = int(round(P ** 0.5))
            if side * side == P:
                gh, gw = side, side
            else:
                raise RuntimeError(
                    f"{P} patch tokens do not match grid {gh}x{gw}; check "
                    f"image_size ({H}x{W}) and patch_size ({self.patch_size})")

        mean_patch = patches.mean(dim=1)             # [B, D]

        if cfg.global_pool == "cls":
            g = cls
        elif cfg.global_pool == "mean_patch":
            g = mean_patch
        else:
            g = torch.cat([cls, mean_patch], dim=-1)  # [B, 2D]

        inter = None
        if want_hidden and getattr(out, "hidden_states", None) is not None:
            hs = out.hidden_states
            inter = [hs[i].float()[:, self.n_prefix:]
                     for i in cfg.intermediate_layers if i < len(hs)]

        return EncoderOutput(
            patch_tokens=patches if cfg.return_patch_tokens else None,
            cls_token=cls if cfg.return_cls else None,
            mean_patch=mean_patch,
            global_feature=g,
            grid_size=(gh, gw),
            intermediate=inter,
        )


def _as_int(v) -> int:
    if isinstance(v, (list, tuple)):
        return int(v[0])
    return int(v)
