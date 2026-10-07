"""
models/encoders/base_encoder.py
─────────────────────────────────────────────────────────────────────────────
Design principle (DINOv3):

    SemCovNet must not know whether the encoder is frozen, LoRA-adapted,
    partially fine-tuned or fully fine-tuned.

So every encoder returns the same object:

    EncoderOutput(
        patch_tokens   : Tensor[B, P, D]   dense spatial tokens (no CLS,
                                           no register tokens) 
        cls_token      : Tensor[B, D]      normalised CLS      
        mean_patch     : Tensor[B, D]      (1/P) sum_p F_{i,p}
        global_feature : Tensor[B, G]      per `global_pool`:
                                             'cls'        -> G = D
                                             'mean_patch' -> G = D
                                             'cls_mean'   -> G = 2D
        grid_size      : (h, w)            P = h * w  (14x14 at 224/16)
        intermediate   : list[Tensor] | None
    )

`patch_tokens` may be None for backbones that cannot produce dense tokens
(e.g. an open_clip global-only encoder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    patch_tokens: Optional[torch.Tensor]          # [B, P, D]
    cls_token: torch.Tensor                       # [B, D]
    mean_patch: Optional[torch.Tensor] = None     # [B, D]
    global_feature: Optional[torch.Tensor] = None # [B, G]
    grid_size: Optional[Tuple[int, int]] = None   # (h, w)
    intermediate: Optional[List[torch.Tensor]] = None

    def as_dict(self) -> dict:
        return {
            "patch_tokens": self.patch_tokens,
            "cls_token": self.cls_token,
            "mean_patch": self.mean_patch,
            "global_feature": self.global_feature,
            "grid_size": self.grid_size,
            "intermediate": self.intermediate,
        }


class BaseVisualEncoder(nn.Module):
    """
    Interface implemented by DINOv3Encoder (and any future CLIP / ConvNeXt
    encoder).

    Subclasses must set
        embed_dim   : D, the token dimension
        global_dim  : G, the width of `global_feature`
    and implement `forward(pixel_values) -> EncoderOutput`.
    """

    embed_dim: int = 0
    global_dim: int = 0
    supports_patch_tokens: bool = False

    # ── parameter bookkeeping ────────────────────────────────────────────
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def has_trainable_params(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self) -> dict:
        """
        Split trainable encoder parameters into named optimizer groups
        (DINOv3). Default: everything trainable is 'backbone'.
        """
        return {"backbone": self.trainable_parameters()}

    def describe(self) -> str:
        return (f"{self.__class__.__name__}(embed_dim={self.embed_dim}, "
                f"global_dim={self.global_dim}, "
                f"trainable={self.n_trainable()/1e6:.2f}M / "
                f"{self.n_total()/1e6:.2f}M)")

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:  # pragma: no cover
        raise NotImplementedError
