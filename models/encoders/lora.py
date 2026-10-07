"""
models/encoders/lora.py
─────────────────────────────────────────────────────────────────────────────
Dependency-free LoRA implementation for nn.Linear layers.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)          # B = 0 -> identity at init

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.lora_dropout(x)
        h = torch.nn.functional.linear(h, self.lora_A.to(h.dtype))
        h = torch.nn.functional.linear(h, self.lora_B.to(h.dtype))
        return out + self.scaling * h

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.2f}"

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold BA into the base weight and return a plain nn.Linear."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.data += delta.to(self.base.weight.dtype)
        return self.base


# ─────────────────────────────────────────────────────────────────────────────
# Target resolution / injection
# ─────────────────────────────────────────────────────────────────────────────

# Canonical (plan) name -> HF DINOv3ViT submodule names
_TARGET_ALIASES = {
    "attention.qkv": ("q_proj", "k_proj", "v_proj"),
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "attention.proj": ("o_proj",),
    "proj": ("o_proj",),
    "feed_forward.fc1": ("up_proj",),
    "fc1": ("up_proj",),
    "feed_forward.fc2": ("down_proj",),
    "fc2": ("down_proj",),
}

ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("up_proj", "down_proj", "gate_proj")


def resolve_lora_targets(target_attention: bool = True,
                         target_mlp: bool = True,
                         extra: Sequence[str] = ()) -> Tuple[str, ...]:
    """Build the leaf-module-name set to wrap."""
    names: List[str] = []
    if target_attention:
        names += list(ATTENTION_TARGETS)
    if target_mlp:
        names += list(MLP_TARGETS)
    for e in extra:
        names += list(_TARGET_ALIASES.get(e, (e,)))
    # de-duplicate, keep order
    return tuple(dict.fromkeys(names))


def inject_lora(module: nn.Module, targets: Iterable[str], r: int = 8,
                alpha: int = 16, dropout: float = 0.05) -> int:
    """
    Recursively replace every nn.Linear whose *attribute name* is in
    `targets` with a LoRALinear. Returns the number of wrapped layers.
    """
    targets = set(targets)
    n_wrapped = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha,
                                             dropout=dropout))
            n_wrapped += 1
        elif isinstance(child, LoRALinear):
            continue
        else:
            n_wrapped += inject_lora(child, targets, r, alpha, dropout)
    return n_wrapped


def lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Every LoRA A/B parameter under `module` (for its own optimizer group)."""
    out: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            out += [m.lora_A, m.lora_B]
    return out


def mark_only_lora_trainable(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    for p in lora_parameters(module):
        p.requires_grad = True
