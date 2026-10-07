"""
models/ccss/losses.py
─────────────────────────────────────────────────────────────────────────────
using CCSS to train the visual representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CCSS_LOSS_MODES = ("align", "contrastive", "none")


def _unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _weights(a: torch.Tensor, min_weight: float = 0.0) -> torch.Tensor:
    w = torch.nan_to_num(a.float(), nan=0.0).clamp(0.0, 1.0)
    if min_weight > 0:
        w = torch.where(w >= min_weight, w, torch.zeros_like(w))
    return w


@dataclass
class CCSSLossConfig:
    mode: str = "align"
    temperature: float = 0.1          # tau_c of §13.3, contrastive only
    min_weight: float = 0.0           # ignore evidence below this
    neg_classes: int = 0              # 0 = all classes; >0 = sampled negatives
    eps: float = 1e-6


class CCSSLoss(nn.Module):
    """
    Parameters
    ----------
    cfg : CCSSLossConfig

    forward(Z, theta, a, y_idx=None, theta_all=None, valid_classes=None)
        Z         : [B, K, D]   concept tokens (gradient path)
        theta     : [B, K, D]   theta_hat_{y_i,k}  (positive targets)
        a         : [B, K]      semantic evidence a_{i,k}
        y_idx     : [B]         class indices          (contrastive only)
        theta_all : [T', K, D]  class-conditioned prototypes (contrastive)
        valid_classes : [T']    class ids matching theta_all's first axis
                                (contrastive; identity when None)

    Returns (loss, logs).
    """

    def __init__(self, cfg: Optional[CCSSLossConfig] = None):
        super().__init__()
        cfg = cfg or CCSSLossConfig()
        if cfg.mode not in CCSS_LOSS_MODES:
            raise ValueError(f"Unknown CCSS loss mode '{cfg.mode}'. "
                             f"Choose from {CCSS_LOSS_MODES}")
        self.cfg = cfg

    # ── §13.2 ───────────────────────────────────────────────────────────
    def align(self, Z: torch.Tensor, theta: torch.Tensor,
              a: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = self.cfg.eps
        w = _weights(a, self.cfg.min_weight).to(Z.dtype)
        cos = (_unit(Z, eps) * _unit(theta, eps)).sum(-1)          # [B, K]
        denom = w.sum().clamp_min(eps)
        loss = (w * (1.0 - cos)).sum() / denom
        with torch.no_grad():
            logs = {"ccss_cos_mean": float((w * cos).sum() / denom),
                    "ccss_weight_sum": float(w.sum())}
        return loss, logs

    # ── §13.3 ───────────────────────────────────────────────────────────
    def contrastive(self, Z: torch.Tensor, theta_all: torch.Tensor,
                    a: torch.Tensor, y_idx: torch.Tensor,
                    valid_classes: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps, tau_c = self.cfg.eps, max(self.cfg.temperature, 1e-3)
        B, K, _ = Z.shape
        Tn = theta_all.shape[0]

        # position of each sample's own class inside theta_all
        if valid_classes is None:
            pos = y_idx.to(torch.long)
        else:
            lut = torch.full((int(valid_classes.max()) + 1,), -1,
                             device=Z.device, dtype=torch.long)
            lut[valid_classes.to(torch.long)] = torch.arange(
                Tn, device=Z.device, dtype=torch.long)
            pos = lut[y_idx.to(torch.long)]
        keep = pos >= 0
        if not bool(keep.any()):
            z = Z.sum() * 0.0
            return z, {"ccss_pos_sim": float("nan")}

        sim = torch.einsum("bkd,tkd->bkt", _unit(Z, eps),
                           _unit(theta_all, eps)) / tau_c          # [B, K, T']
        logp = F.log_softmax(sim, dim=-1)
        idx = pos.clamp_min(0).view(B, 1, 1).expand(B, K, 1)
        pos_logp = logp.gather(-1, idx).squeeze(-1)                # [B, K]

        w = _weights(a, self.cfg.min_weight).to(Z.dtype) * keep.unsqueeze(1)
        denom = w.sum().clamp_min(eps)
        loss = -(w * pos_logp).sum() / denom
        with torch.no_grad():
            pos_sim = sim.gather(-1, idx).squeeze(-1) * tau_c
            logs = {"ccss_pos_sim": float((w * pos_sim).sum() / denom),
                    "ccss_weight_sum": float(w.sum()),
                    "ccss_n_negatives": float(Tn - 1)}
        return loss, logs

    # ── dispatch ────────────────────────────────────────────────────────
    def forward(self, Z: Optional[torch.Tensor],
                theta: Optional[torch.Tensor],
                a: Optional[torch.Tensor],
                y_idx: Optional[torch.Tensor] = None,
                theta_all: Optional[torch.Tensor] = None,
                valid_classes: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.cfg.mode == "none" or Z is None or a is None or a.numel() == 0:
            zero = (Z.sum() * 0.0) if Z is not None else torch.zeros(())
            return zero, {}
        if self.cfg.mode == "contrastive":
            if theta_all is None or y_idx is None:
                raise ValueError("contrastive CCSS needs theta_all and y_idx")
            return self.contrastive(Z, theta_all, a, y_idx, valid_classes)
        if theta is None:
            raise ValueError("align CCSS needs theta")
        return self.align(Z, theta, a)

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"mode={c.mode}, temperature={c.temperature}, "
                f"neg_classes={c.neg_classes or 'all'}")


def build_ccss_loss(mode: str = "align", temperature: float = 0.1,
                    min_weight: float = 0.0, neg_classes: int = 0) -> CCSSLoss:
    return CCSSLoss(CCSSLossConfig(mode=mode, temperature=temperature,
                                   min_weight=min_weight,
                                   neg_classes=neg_classes))
