"""
models/encoders/load_encoder.py
─────────────────────────────────────────────────────────────────────────────
One entry point for every backbone:

        encoder, preprocess, embed_dim = load_encoder("CelebA")

The per-dataset variant
map, the global override switches and `_load_encoder` keep the same shape,
with the "......implement appropriate model loading logic here....." part
filled in.

  Instantiated inside train_semcov.py / predict_semcov.py via:
      from models.encoders import load_encoder
      encoder, preprocess, embed_dim = load_encoder(args.dataset, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base_encoder import BaseVisualEncoder, EncoderOutput
from .dinov3_encoder import (DINOV3_VITB16_LVD, DINOv3Config, DINOv3Encoder,
                             LoRAConfig)

try:                                   # keeps this file importable standalone
    from data_loaders import DATASET_CHOICES
except Exception:                      # pragma: no cover
    DATASET_CHOICES = ["CelebA", "CUB", "Derm7ptDerm", "Derm7ptClinic",
                       "MILK10kDerm", "MILK10kClinic"]


# ─────────────────────────────────────────────────────────────────────────────
# Global encoder variant switch
# ─────────────────────────────────────────────────────────────────────────────
# None -> no predefined encoder, use the per-dataset map below.
ENCODER_VARIANT: Optional[str] = None

# Set to e.g. "dinov3" to force one backbone for ALL datasets at once.
FORCE_GLOBAL_VARIANT: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Variant specs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EncoderSpec:
    kind: str            # 'dinov3' | 'open_clip'
    model_name: str      # hub id / open_clip model name
    pretrained: Optional[str]
    embed_dim: int
    supports_patch_tokens: bool


_VARIANT_SPECS: dict[str, EncoderSpec] = {
    "dinov3": EncoderSpec(
        kind="dinov3",
        model_name=DINOV3_VITB16_LVD,
        pretrained=None,                 # weights come with the hub id
        embed_dim=768,
        supports_patch_tokens=True,
    ),
    # DermLIP / PanDerm — requires the Derm1M open_clip fork
    # (imported automatically from DERMLIP_PROJECT_ROOT when selected)
    "dermlip": EncoderSpec(
        kind="open_clip",
        model_name="hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        pretrained=None,
        embed_dim=512,
        supports_patch_tokens=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset variant map
# ─────────────────────────────────────────────────────────────────────────────
_DATASET_VARIANT: dict[str, str] = {
    "CelebA": "dinov3",
    "CUB": "dinov3",
    "Derm7ptDerm": "dinov3",
    "Derm7ptClinic": "dinov3",
    "MILK10kDerm": "dinov3",
    "MILK10kClinic": "dinov3",
    # the Waterbirds stress test uses the SAME backbone as every
    # other dataset — whole point is that the encoder is not a variable.
    "Waterbirds": "dinov3",
}


def _resolve_variant(dataset_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        chosen = variant
    elif FORCE_GLOBAL_VARIANT is not None:
        chosen = FORCE_GLOBAL_VARIANT
    elif ENCODER_VARIANT is not None:
        chosen = ENCODER_VARIANT
    else:
        chosen = _DATASET_VARIANT.get(dataset_name)
        if chosen is None:
            raise ValueError(
                f"No encoder variant assigned for dataset '{dataset_name}'. "
                f"Add it to _DATASET_VARIANT or pass variant=... explicitly.")
    if chosen not in _VARIANT_SPECS:
        raise ValueError(f"Unknown encoder variant '{chosen}'. "
                         f"Choose from: {list(_VARIANT_SPECS)}")
    return chosen


def _build_registry(variant: Optional[str] = None) -> dict[str, EncoderSpec]:
    registry = {d: _VARIANT_SPECS[_resolve_variant(d, variant)]
                for d in DATASET_CHOICES}
    return registry


_ENCODER_REGISTRY: dict[str, EncoderSpec] = _build_registry()


# ─────────────────────────────────────────────────────────────────────────────
# open_clip fallback encoder (global features only)
# ─────────────────────────────────────────────────────────────────────────────
class OpenCLIPGlobalEncoder(BaseVisualEncoder):
    """
    Thin open_clip wrapper exposing the shared EncoderOutput contract.
    `patch_tokens` is None: this backbone can serve the ERM baseline
    and probes, but not the Visual Concept Router.
    """
    supports_patch_tokens = False

    def __init__(self, model_name: str, pretrained: Optional[str] = None,
                 embed_dim: int = 512, freeze: bool = True):
        super().__init__()
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.clip = model
        self.preprocess = preprocess
        self.embed_dim = embed_dim
        self.global_dim = embed_dim
        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.has_trainable_params:
            self.clip.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:
        with torch.set_grad_enabled(self.has_trainable_params
                                    and torch.is_grad_enabled()):
            feats = self.clip.encode_image(pixel_values).float()
        if feats.ndim == 3:                      # [B, T, D] -> CLS
            feats = feats[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return EncoderOutput(patch_tokens=None, cls_token=feats,
                             mean_patch=None, global_feature=feats,
                             grid_size=None)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def _load_encoder(dataset_name: str,
                  variant: Optional[str] = None,
                  image_size: int = 224,
                  train_mode: str = "frozen",
                  global_pool: str = "cls_mean",
                  precision: str = "bf16",
                  unfreeze_last_n: int = 3,
                  lora_rank: int = 8,
                  lora_alpha: int = 16,
                  lora_dropout: float = 0.05,
                  lora_last_n: int = 3,
                  lora_target_attention: bool = True,
                  lora_target_mlp: bool = True,
                  gradient_checkpointing: bool = False,
                  return_intermediate: bool = False,
                  weights: Optional[str] = None,
                  ) -> Tuple[BaseVisualEncoder, "callable", int]:
    """
    Returns (encoder, preprocess_fn, embed_dim).

    encoder(x: Tensor[B,3,H,W]) -> EncoderOutput with
        patch_tokens [B,P,768], cls_token [B,768], mean_patch [B,768],
        global_feature [B,768 or 1536], grid_size (14,14) at 224px.

    preprocess_fn is an albumentations transform pair builder compatible with
    data_loaders.BaselineDataset (normalise only, no ToTensorV2 — the dataset
    does the HWC->CHW transpose itself).
    """
    if dataset_name not in DATASET_CHOICES and dataset_name not in _DATASET_VARIANT:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Choose from: {DATASET_CHOICES}")

    chosen = _resolve_variant(dataset_name, variant)
    spec = _VARIANT_SPECS[chosen]

    from ..transforms import get_encoder_transforms   # lazy: avoids cycles

    if spec.kind == "dinov3":
        cfg = DINOv3Config(
            weights=weights or spec.model_name,
            image_size=image_size,
            train_mode=train_mode,
            unfreeze_last_n=unfreeze_last_n,
            global_pool=global_pool,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
            return_intermediate=return_intermediate,
            lora=LoRAConfig(
                enabled=("lora" in str(train_mode).lower()),
                last_n_blocks=lora_last_n,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_attention=lora_target_attention,
                target_mlp=lora_target_mlp,
            ),
        )
        encoder = DINOv3Encoder(cfg)
        preprocess = get_encoder_transforms(image_size, backbone="dinov3")
        return encoder, preprocess, encoder.embed_dim

    if spec.kind == "open_clip":
        encoder = OpenCLIPGlobalEncoder(spec.model_name, spec.pretrained,
                                        spec.embed_dim,
                                        freeze=(train_mode == "frozen"))
        preprocess = get_encoder_transforms(image_size, backbone="clip")
        return encoder, preprocess, encoder.embed_dim

    raise ValueError(f"Unhandled encoder kind '{spec.kind}'")


# Public alias — `load_encoder(config)` in the plan's wording.
def load_encoder(dataset_name: str, **kwargs):
    return _load_encoder(dataset_name, **kwargs)


def encoder_registry() -> dict[str, EncoderSpec]:
    return dict(_ENCODER_REGISTRY)
