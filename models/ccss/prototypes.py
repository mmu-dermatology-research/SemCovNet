"""
models/ccss/prototypes.py
─────────────────────────────────────────────────────────────────────────────
specialization:
    alpha_{y,k}     = n_eff tau^2 / (n_eff tau^2 + sigma_k^2)
    theta_hat_{y,k} = m_{y,k} + alpha_{y,k} (z_bar_{y,k} - m_{y,k})
where
    n_eff = 0   ->  alpha = 0        theta_hat = m      
    n_eff small ->  alpha small      lean on shared structure
    n_eff large ->  alpha -> 1       specialize

No Tail/Middle/Head band is used anywhere in this file; the 25/50/25 coverage groups
are evaluation-only.

Representation space:
m, z_bar, sigma_k^2, tau^2 and alpha are all estimated and mixed in the
*native* semantic space. Normalization happens exactly once, inside
`CCSSLoss.align`.

Alternative calibration modes live in `alpha_mode`:

    'ccss'   alpha = n tau^2 / (n tau^2 + sigma_k^2)    the proposed mechanism
    'ccss_hetero'                                       ablation
             alpha = n tau_k^2 / (n tau_k^2 + sigma^2_{y,k})
             the richer, over-parameterised pair-specific estimator that the
             primary rule replaces — kept so the paper can show that the
             stable pooled estimator beats it, not just the heuristics.
    'zero'   alpha = 0        shared prior only                          (B2)
    'one'    alpha = 1        fixed pair specialization, uncalibrated    (B3)
    'gate'   alpha = C/(C+l)  the earlier heuristic coverage gate        (B4)

    'freq'    alpha = n_raw / max(n_raw)                       B0
              raw sample count only. n_raw counts a_{i,k} > 0, so on soft
              evidence (MILK10k) it deliberately differs from n_eff: ten
              images with a = 0.1 count the same as ten with a = 0.9.
    'cov'     alpha = C_{y,k}                                  B1
              class-conditioned prevalence only. Ignores how many images
              produced it and how consistent their representations are.
    'eff'     alpha = n_eff / (n_eff + kappa)                  B2
              soft/weighted support only, no uncertainty term.
    'n_sigma' alpha = n_eff / (n_eff + sigma_k^2)              B3
              support and observation noise, but no tau^2, so a concept that
              legitimately varies across classes is treated the same as one
              that does not.
    'ccss'    the full rule                                    B4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

ALPHA_MODES = ("ccss",            # corrected primary: pooled sigma_k^2, global tau^2
               "ccss_hetero",     # ablation: pair-specific sigma^2_{y,k}, tau_k^2
               "zero", "one", "gate",
               # specialization-rule ablation (B0-B3)
               "freq", "cov", "eff", "n_sigma")
TOKEN_NORMS = ("l2", "layernorm", "none")
SUPPORT_MODES = ("sum", "ess")


def unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class PrototypeBankConfig:
    ema: float = 0.95                 # beta 0.99 is the slower option
    eps: float = 1e-6
    sigma_floor: float = 1e-6         # numerical safeguards, not SCI
    tau_floor: float = 1e-6
    token_norm: str = "none"          # corrected primary (was 'l2'); see docstring
    theta_normalize: str = "none"     # corrected primary (was 'unit')
    support: str = "sum"              # 'sum' (primary) | 'ess' (ablation)
    alpha_mode: str = "ccss"
    gate_lambda: float = 0.1          # only for alpha_mode='gate' (B4)
    alpha_kappa: float = 1.0          # only for alpha_mode='eff' (B2)
    min_weight: float = 0.0           # ignore evidence below this when updating
    n_eff_scale: float = 1.0          # see `calibrate_support_scale`


class ClassConceptPrototypeBank(nn.Module):
    """
    Dataset-level statistics for every (class, concept) pair.

    Buffers (all FP32, kept out of autocast for stability):

        z_bar   [T, K, D]   EMA empirical prototype
        q_mean  [T, K]      EMA of E_w[||z||^2] (second moment)
        seen    [T, K]      number of EMA updates a pair has received
        n_eff   [T, K]      effective semantic support
        sigma2_concept [K]  PRIMARY pooled observation variance (per dim)
        tau2_global    []   PRIMARY global specialization variance (per dim)
        sigma2  [T, K]      pair-specific within variance (intermediate for the
                            pooling, and the 'ccss_hetero' ablation)
        tau2    [K]         concept-specific between-class variability
                            ('ccss_hetero' ablation)
        alpha   [T, K]      specialization coefficient in [0, 1]
        n_raw   [T, K]      raw count of a_{i,k} > 0 (alpha_mode='freq')
        coverage[T, K]      C_{y,k}, stored for analysis and for alpha_mode='gate'
        w_sum / w2_sum      support accumulators (only used by commit_support)

    Memory: T x K x D floats. CUB (200 x 312 x 256) is 64 MB — the largest
    case in the study and comfortably inside an A100 40 GB budget.
    """

    def __init__(self, n_classes: int, n_concepts: int, dim: int = 256,
                 cfg: Optional[PrototypeBankConfig] = None):
        super().__init__()
        cfg = cfg or PrototypeBankConfig()
        for name, val, allowed in (("alpha_mode", cfg.alpha_mode, ALPHA_MODES),
                                   ("token_norm", cfg.token_norm, TOKEN_NORMS),
                                   ("support", cfg.support, SUPPORT_MODES)):
            if val not in allowed:
                raise ValueError(f"Unknown {name} '{val}'. Choose from {allowed}")
        self.cfg = cfg
        self.T, self.K, self.D = int(n_classes), int(n_concepts), int(dim)

        z = torch.zeros
        self.register_buffer("z_bar", z(self.T, self.K, self.D))
        self.register_buffer("q_mean", z(self.T, self.K))
        self.register_buffer("seen", z(self.T, self.K))
        self.register_buffer("n_eff", z(self.T, self.K))
        # primary (corrected) statistics
        self.register_buffer("sigma2_concept",
                             torch.full((self.K,), float(cfg.sigma_floor)))
        self.register_buffer("tau2_global",
                             torch.tensor(float(cfg.tau_floor)))
        # pair-specific / concept-specific statistics: intermediate for the
        # pooling above, and the estimators of the 'ccss_hetero' ablation
        self.register_buffer("sigma2", torch.full((self.T, self.K),
                                                  float(cfg.sigma_floor)))
        self.register_buffer("tau2", torch.full((self.K,), float(cfg.tau_floor)))
        self.register_buffer("alpha", z(self.T, self.K))
        self.register_buffer("coverage", torch.full((self.T, self.K),
                                                    float("nan")))
        self.register_buffer("n_raw", z(self.T, self.K))
        self.register_buffer("w_sum", z(self.T, self.K))
        self.register_buffer("w2_sum", z(self.T, self.K))
        self.register_buffer("support_committed", torch.zeros((), dtype=torch.bool))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))
        # transient exact-accumulation scratch (Stage B); never checkpointed
        self._acc_S = self._acc_W = self._acc_Q = None

    # ── helpers ─────────────────────────────────────────────────────────
    @property
    def valid(self) -> torch.Tensor:
        """Pairs with evidence AND at least one prototype observation."""
        return (self.n_eff > 0) & (self.seen > 0)

    def _prepare_tokens(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z.detach().float()
        if self.cfg.token_norm == "l2":
            return unit(Z, self.cfg.eps)
        if self.cfg.token_norm == "layernorm":
            return torch.nn.functional.layer_norm(Z, (Z.shape[-1],))
        return Z

    @staticmethod
    def _weights(a: torch.Tensor, min_weight: float = 0.0) -> torch.Tensor:
        """a_{i,k} -> w_{i,k}. NaN (missing label) contributes zero weight."""
        w = torch.nan_to_num(a.detach().float(), nan=0.0).clamp_(0.0, 1.0)
        if min_weight > 0:
            w = torch.where(w >= min_weight, w, torch.zeros_like(w))
        return w

    # ── prototype update ──────────────────────────────────────
    @torch.no_grad()
    def update(self, Z: torch.Tensor, y_idx: torch.Tensor,
               a: torch.Tensor, ema: Optional[float] = None) -> None:
        """
        One EMA step from a minibatch.

        Z     : [B, K, D] concept tokens (detached internally)
        y_idx : [B]       class indices
        a     : [B, K]    semantic evidence
        """
        if Z is None or a is None or a.numel() == 0:
            return
        beta = self.cfg.ema if ema is None else float(ema)
        eps = self.cfg.eps

        Zn = self._prepare_tokens(Z)
        w = self._weights(a, self.cfg.min_weight).to(Zn.dtype)
        y_idx = y_idx.detach().to(torch.long)

        # scatter-add into [T, K, *] accumulators
        S = torch.zeros(self.T, self.K, self.D, device=Zn.device, dtype=Zn.dtype)
        S.index_add_(0, y_idx, w.unsqueeze(-1) * Zn)
        W = torch.zeros(self.T, self.K, device=Zn.device, dtype=Zn.dtype)
        W.index_add_(0, y_idx, w)
        Q = torch.zeros(self.T, self.K, device=Zn.device, dtype=Zn.dtype)
        Q.index_add_(0, y_idx, w * (Zn * Zn).sum(-1))

        present = W > eps
        if not bool(present.any()):
            return
        denom = W.clamp_min(eps)
        z_batch = S / denom.unsqueeze(-1)                 # [T, K, D]
        q_batch = Q / denom                               # [T, K]

        fresh = present & (self.seen == 0)                # first observation
        warm = present & (self.seen > 0)

        p3 = present.unsqueeze(-1)
        f3, w3 = fresh.unsqueeze(-1), warm.unsqueeze(-1)
        self.z_bar = torch.where(
            f3, z_batch.to(self.z_bar.dtype),
            torch.where(w3, (beta * self.z_bar
                             + (1.0 - beta) * z_batch).to(self.z_bar.dtype),
                        self.z_bar))
        self.q_mean = torch.where(
            fresh, q_batch.to(self.q_mean.dtype),
            torch.where(warm, (beta * self.q_mean
                               + (1.0 - beta) * q_batch).to(self.q_mean.dtype),
                        self.q_mean))
        self.seen = self.seen + present.to(self.seen.dtype)
        self.initialised.fill_(True)
        del S, Q, z_batch, p3

    # ── Stage B: exact accumulation over a full pass ─────────
    # The EMA above is what runs during training. Stage B instead
    # wants the statistics *initialised*, and an exact weighted pass over the
    # training split beats an EMA transient: tail pairs get their one true value
    # rather than a value that is still 95% zeros. The accumulators are plain
    # attributes, not buffers, so they never enter a checkpoint.
    @torch.no_grad()
    def begin_accumulation(self) -> None:
        dev = self.z_bar.device
        self._acc_S = torch.zeros(self.T, self.K, self.D, device=dev,
                                  dtype=torch.float32)
        self._acc_W = torch.zeros(self.T, self.K, device=dev,
                                  dtype=torch.float32)
        self._acc_Q = torch.zeros(self.T, self.K, device=dev,
                                  dtype=torch.float32)

    @torch.no_grad()
    def accumulate(self, Z: torch.Tensor, y_idx: torch.Tensor,
                   a: torch.Tensor) -> None:
        if getattr(self, "_acc_S", None) is None:
            self.begin_accumulation()
        Zn = self._prepare_tokens(Z)
        w = self._weights(a, self.cfg.min_weight).to(Zn.dtype)
        y_idx = y_idx.detach().to(torch.long)
        self._acc_S.index_add_(0, y_idx, (w.unsqueeze(-1) * Zn).float())
        self._acc_W.index_add_(0, y_idx, w.float())
        self._acc_Q.index_add_(0, y_idx, (w * (Zn * Zn).sum(-1)).float())

    @torch.no_grad()
    def commit_accumulation(self) -> None:
        if getattr(self, "_acc_S", None) is None:
            return
        W = self._acc_W
        present = W > self.cfg.eps
        denom = W.clamp_min(self.cfg.eps)
        self.z_bar.copy_(torch.where(present.unsqueeze(-1),
                                     self._acc_S / denom.unsqueeze(-1),
                                     self.z_bar))
        self.q_mean.copy_(torch.where(present, self._acc_Q / denom,
                                      self.q_mean))
        self.seen.copy_(torch.where(present, self.seen.clamp_min(1.0),
                                    self.seen))
        self.initialised.fill_(True)
        self._acc_S = self._acc_W = self._acc_Q = None

    # ── effective support ───────────────────────────────────
    @torch.no_grad()
    def accumulate_support(self, y_idx: torch.Tensor, a: torch.Tensor) -> None:
        """Accumulate sum w and sum w^2 over a full pass (see commit_support)."""
        w = self._weights(a, self.cfg.min_weight)
        y_idx = y_idx.detach().to(torch.long)
        self.w_sum.index_add_(0, y_idx, w.to(self.w_sum.dtype))
        self.w2_sum.index_add_(0, y_idx, (w * w).to(self.w2_sum.dtype))

    @torch.no_grad()
    def commit_support(self, reset: bool = True) -> None:
        """Turn the accumulators from one full epoch into n_eff."""
        if self.cfg.support == "ess":
            n = (self.w_sum ** 2) / (self.w2_sum + self.cfg.eps)
        else:
            n = self.w_sum.clone()
        self.n_eff.copy_(n)
        self.support_committed.fill_(True)
        if reset:
            self.w_sum.zero_()
            self.w2_sum.zero_()

    @torch.no_grad()
    def set_raw_count(self, n_raw) -> None:
        """
        Install n_{y,k} = #{i : y_i = y, a_{i,k} > 0} (B0).
        """
        n = torch.as_tensor(n_raw, dtype=self.n_raw.dtype,
                            device=self.n_raw.device)
        if n.shape != self.n_raw.shape:
            raise ValueError(f"n_raw shape {tuple(n.shape)} != "
                             f"{tuple(self.n_raw.shape)}")
        self.n_raw.copy_(torch.nan_to_num(n, nan=0.0).clamp_min(0.0))

    @torch.no_grad()
    def set_support(self, n_eff) -> None:
        """
        Install an exactly-computed support matrix [T, K].

        Preferred path: n_eff is a function of (y_i, a_i) only, so
        `metrics.class_concept_support(df_train, ...)` gives the exact value
        without a forward pass and without any EMA transient.
        """
        n = torch.as_tensor(n_eff, dtype=self.n_eff.dtype,
                            device=self.n_eff.device)
        if tuple(n.shape) != (self.T, self.K):
            raise ValueError(f"support matrix must be [{self.T}, {self.K}], "
                             f"got {tuple(n.shape)}")
        self.n_eff.copy_(torch.nan_to_num(n, nan=0.0).clamp_min(0.0))
        self.support_committed.fill_(True)

    @torch.no_grad()
    def set_coverage(self, C) -> None:
        """Store C_{y,k} for analysis and for alpha_mode='gate'."""
        c = torch.as_tensor(C, dtype=self.coverage.dtype,
                            device=self.coverage.device)
        if tuple(c.shape) != (self.T, self.K):
            raise ValueError(f"coverage matrix must be [{self.T}, {self.K}], "
                             f"got {tuple(c.shape)}")
        self.coverage.copy_(c)

    # ── uncertainty + alpha ─────────────────────────
    @torch.no_grad()
    def refresh(self, shared=None) -> Dict[str, float]:
        """
        Recompute the variance statistics and alpha (once per epoch, or
        every N iterations — never every minibatch, that is what destabilises
        the DINO <-> prototype feedback loop).

        Order of operations (corrected core):

            1. s^2_{y,k}        pair-level within variance   (intermediate)
            2. sigma_k^2        pooled across classes        (PRIMARY)
            3. r_{y,k}          = z_bar - m
            4. tau^2            pooled over all valid pairs  (PRIMARY)
            4b. tau_k^2         concept-specific             ('ccss_hetero')
            5. alpha            compute_alpha()

        `shared` is the SharedSemanticStructure; it is needed for
        r_{y,k} = z_bar - m and therefore for tau^2. Without it steps 3-4b are
        skipped and the previous tau values are kept.
        """
        cfg = self.cfg
        v = self.valid                                              # [T, K]
        # support weights: n_eff over valid pairs, 0 elsewhere
        wts = torch.where(v, self.n_eff, torch.zeros_like(self.n_eff))

        # ── 1. pair-level within variance (intermediate statistic) ──────
        # s^2_{y,k} = (E_w[||z||^2] - ||z_bar||^2) / D       
        sig = (self.q_mean - (self.z_bar * self.z_bar).sum(-1)) / float(self.D)
        sig = torch.nan_to_num(sig, nan=cfg.sigma_floor)
        sig = sig.clamp_min(cfg.sigma_floor)                      
        sig = torch.where(v, sig, torch.full_like(sig, cfg.sigma_floor))
        self.sigma2.copy_(sig)

        # ── 2. PRIMARY: pooled concept observation variance ─────────────
        #   sigma_k^2 = sum_y n_eff_{y,k} s^2_{y,k} / sum_y n_eff_{y,k}
        # A singleton pair contributes almost nothing to the pool (its weight
        # is 1) but *inherits* the pooled value, which is the whole point:
        # its own s^2 ~ 0 no longer buys it alpha ~ 1.
        w_k = wts.sum(0)                                            # [K]
        sig_k = (wts * sig).sum(0) / w_k.clamp_min(cfg.eps)         # [K]
        # a concept with no valid observation at all keeps the floor
        sig_k = torch.where(w_k > 0, sig_k, torch.full_like(sig_k, cfg.sigma_floor))
        self.sigma2_concept.copy_(torch.nan_to_num(sig_k, nan=cfg.sigma_floor)
                                  .clamp_min(cfg.sigma_floor))

        # ── 3-4. residuals and specialization variance ──────────────────
        if shared is not None:
            m = shared.m_all().detach().float()                     # [T, K, D]
            zb, mm = self.z_bar, m
            if cfg.theta_normalize == "unit":                       # ablation only
                zb, mm = unit(zb, cfg.eps), unit(mm, cfg.eps)
            r = zb - mm                                             # [T, K, D]

            # 4. PRIMARY: one global tau^2 over ALL valid class-concept pairs
            #   r_bar  = sum_{y,k} w r / sum w
            #   tau^2  = sum_{y,k} w ||r - r_bar||^2 / (D sum w)
            w_tot = wts.sum()
            if float(w_tot) > 0:
                w3 = wts.unsqueeze(-1)
                r_bar = (w3 * r).sum(dim=(0, 1)) / w_tot.clamp_min(cfg.eps)  # [D]
                dev_g = ((r - r_bar.view(1, 1, self.D)) ** 2).sum(-1)        # [T,K]
                tau_g = (wts * dev_g).sum() / (float(self.D)
                                               * w_tot.clamp_min(cfg.eps))
                tau_g = torch.nan_to_num(tau_g, nan=cfg.tau_floor)
            else:
                tau_g = torch.tensor(cfg.tau_floor, device=r.device)
                dev_g = None
            self.tau2_global.copy_(tau_g.clamp_min(cfg.tau_floor))

            # 4b. ablation: concept-specific between-class variability
            rho = wts / w_k.clamp_min(cfg.eps).unsqueeze(0)
            rho = torch.where(w_k.unsqueeze(0) > 0, rho, torch.zeros_like(rho))
            r_bar_k = (rho.unsqueeze(-1) * r).sum(0)                # [K, D]
            dev = ((r - r_bar_k.unsqueeze(0)) ** 2).sum(-1)         # [T, K]
            tau = (rho * dev).sum(0) / float(self.D)                # [K]
            # a concept observed in <2 classes has no between-class evidence;
            # fall back to the pooled global value rather than to the floor,
            # which is what made tau_k^2 unusable on binary-class datasets
            n_classes_k = (wts > 0).sum(0)
            tau = torch.where(n_classes_k >= 2, tau,
                              torch.full_like(tau, float(self.tau2_global)))
            self.tau2.copy_(torch.nan_to_num(tau, nan=cfg.tau_floor)
                            .clamp_min(cfg.tau_floor))            
            del m, r, dev, dev_g

        self.alpha.copy_(self.compute_alpha())
        return self.mechanism_report()

    @torch.no_grad()
    def compute_alpha(self) -> torch.Tensor:
        """the zero-support safeguard."""
        cfg = self.cfg
        if cfg.alpha_mode == "zero":
            a = torch.zeros_like(self.alpha)
        elif cfg.alpha_mode == "one":
            a = torch.ones_like(self.alpha)
        elif cfg.alpha_mode == "gate":
            c = torch.nan_to_num(self.coverage, nan=0.0)
            a = c / (c + cfg.gate_lambda)
        elif cfg.alpha_mode == "freq":
            # B0 — raw sample count only, normalised by the largest count so
            # alpha stays in [0, 1] without introducing a second scale knob
            n = torch.nan_to_num(self.n_raw, nan=0.0)
            if float(n.max()) <= 0:
                # nobody called set_raw_count(); fall back to n_eff and say so
                n = torch.nan_to_num(self.n_eff, nan=0.0)
            a = n / (n.max() + cfg.eps)
        elif cfg.alpha_mode == "cov":
            # B1 — class-conditioned prevalence only
            a = torch.nan_to_num(self.coverage, nan=0.0)
        elif cfg.alpha_mode == "eff":
            # B2 — soft support only, no uncertainty term
            n = cfg.n_eff_scale * self.n_eff
            a = n / (n + cfg.alpha_kappa).clamp_min(cfg.eps)
        elif cfg.alpha_mode == "n_sigma":
            # B3 — support and observation noise, no tau^2
            n = cfg.n_eff_scale * self.n_eff
            a = n / (n + self.sigma2_concept.unsqueeze(0)).clamp_min(cfg.eps)
        elif cfg.alpha_mode == "ccss_hetero":
            # ablation — the over-parameterised pair-specific estimator:
            #   alpha = n tau_k^2 / (n tau_k^2 + sigma^2_{y,k})
            num = (cfg.n_eff_scale * self.n_eff) * self.tau2.unsqueeze(0)
            a = num / (num + self.sigma2).clamp_min(cfg.eps)
        else:
            # PRIMARY — pooled observation variance, global specialization
            # variance:  alpha = n tau^2 / (n tau^2 + sigma_k^2)
            #   tau^2         scalar,  broadcast over [T, K]
            #   sigma_k^2     [K],     broadcast over classes
            #   n_eff_{y,k}   [T, K]
            num = (cfg.n_eff_scale * self.n_eff) * self.tau2_global
            a = num / (num + self.sigma2_concept.unsqueeze(0)).clamp_min(cfg.eps)
        # zero support -> share; unseen pair -> share
        a = torch.where(self.valid, a, torch.zeros_like(a))
        return a.clamp_(0.0, 1.0)

    @torch.no_grad()
    def calibrate_support_scale(self, target_median_alpha: float = 0.5
                                ) -> float:
        """The primary configuration is

            calibrate_alpha = 0        n_eff_scale = 1
        """
        v = self.valid
        if not bool(v.any()):
            return float(self.cfg.n_eff_scale)
        ratio = (self.n_eff * self.tau2_global
                 / self.sigma2_concept.unsqueeze(0)
                 .clamp_min(self.cfg.eps))[v]
        med = float(torch.quantile(ratio.clamp_min(self.cfg.eps), 0.5))
        if med <= 0:
            return float(self.cfg.n_eff_scale)
        t = min(max(target_median_alpha, 1e-3), 1 - 1e-3)
        self.cfg.n_eff_scale = float((t / (1.0 - t)) / med)
        self.alpha.copy_(self.compute_alpha())
        print(f"[ccss] n_eff_scale calibrated to {self.cfg.n_eff_scale:.4g} "
              f"(median alpha -> {target_median_alpha:g})")
        return self.cfg.n_eff_scale

    # ── the calibrated semantic target ──────────────────────
    def theta(self, y_idx: torch.Tensor, shared) -> torch.Tensor:
        """
        t_hat_{y_i,k} = m + alpha (z_bar - m)  for a minibatch -> [B, K, D].
        """
        m = shared.m_batch(y_idx)                                   # [B, K, D]
        zb = self.z_bar.index_select(0, y_idx).detach()
        a = self.alpha.index_select(0, y_idx).detach().unsqueeze(-1)
        if self.cfg.theta_normalize == "unit":                      # ablation
            m, zb = unit(m, self.cfg.eps), unit(zb, self.cfg.eps)
        return m + a * (zb - m)

    def theta_all(self, shared, classes: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """
        theta_hat for every class (or a subset) -> [T', K, D].
        Used by the class-contrastive variant; for CUB this is a
        200 x 312 x 256 tensor, hence the optional class subset.
        """
        if classes is None:
            m = shared.m_all()
            zb, a = self.z_bar.detach(), self.alpha.detach().unsqueeze(-1)
        else:
            classes = classes.to(torch.long)
            m = shared.m_batch(classes)
            zb = self.z_bar.index_select(0, classes).detach()
            a = self.alpha.index_select(0, classes).detach().unsqueeze(-1)
        if self.cfg.theta_normalize == "unit":                      # ablation
            m, zb = unit(m, self.cfg.eps), unit(zb, self.cfg.eps)
        return m + a * (zb - m)

    # ── reporting ──────────────────────────────────────────────
    @torch.no_grad()
    def mechanism_report(self, bands: Optional[torch.Tensor] = None
                         ) -> Dict[str, float]:
        """
        CCSS mechanism metrics. `bands` is an optional [T, K] int tensor with
        0=tail, 1=middle, 2=head from TRAINING coverage — evaluation-only
        strata, never consumed by the mechanism itself.
        """
        v = self.valid
        n_valid = int(v.sum())
        out: Dict[str, float] = {
            "ccss_pairs_valid": float(n_valid),
            "ccss_pairs_total": float(self.T * self.K),
            "ccss_pairs_zero_support": float(int((self.n_eff <= 0).sum())),
            # raw count and effective support coincide for binary
            # evidence and diverge for soft evidence; logging both makes the
            # B0-vs-B4 distinction checkable from the training log alone.
            "n_raw_mean": float(self.n_raw.mean()),
            "n_eff_mean": float(self.n_eff.mean()),
        }
        if n_valid == 0:
            return out

        a, s, t = self.alpha[v], self.sigma2[v], self.tau2
        out.update({
            "alpha_mean": float(a.mean()),
            "alpha_std": float(a.std()) if a.numel() > 1 else 0.0,
            "alpha_q10": float(torch.quantile(a, 0.10)),
            "alpha_q50": float(torch.quantile(a, 0.50)),
            "alpha_q90": float(torch.quantile(a, 0.90)),
            "alpha_frac_below_0.05": float((a < 0.05).float().mean()),
            "alpha_frac_above_0.95": float((a > 0.95).float().mean()),
            # primary (corrected) statistics
            "sigma2_concept_mean": float(self.sigma2_concept.mean()),
            "sigma2_concept_q50": float(torch.quantile(self.sigma2_concept, 0.50)),
            "tau2_global": float(self.tau2_global),
            # pair-specific / concept-specific ('ccss_hetero') statistics
            "sigma2_mean": float(s.mean()),
            "sigma2_q50": float(torch.quantile(s, 0.50)),
            "tau2_mean": float(t.mean()),
            "tau2_q50": float(torch.quantile(t, 0.50)),
            "n_eff_mean": float(self.n_eff[v].mean()),
        })
        # tail sanity check: a pair with support <= 3 must NOT be sitting
        # at alpha ~ 1. This is the metric that caught the singleton failure,
        # so it is logged every refresh rather than only in the notebook.
        tail = v & (self.n_eff <= 3.0)
        if bool(tail.any()):
            at = self.alpha[tail]
            out["alpha_tail_support_mean"] = float(at.mean())
            out["alpha_tail_support_frac_above_0.9"] = float(
                (at > 0.9).float().mean())
            out["n_pairs_support_le3"] = float(int(tail.sum()))
        if bands is not None:
            bands = bands.to(self.alpha.device)
            for idx, name in ((0, "tail"), (1, "middle"), (2, "head")):
                m = v & (bands == idx)
                if bool(m.any()):
                    out[f"alpha_{name}"] = float(self.alpha[m].mean())
                    out[f"sigma2_{name}"] = float(self.sigma2[m].mean())
                    out[f"n_eff_{name}"] = float(self.n_eff[m].mean())
            if "alpha_head" in out and "alpha_tail" in out:
                out["alpha_head_tail_gap"] = out["alpha_head"] - out["alpha_tail"]
        return out

    @torch.no_grad()
    def support_alpha_diagnostic(self, n_bins: int = 8) -> Dict[str, float]:
        """
        "specialization vs support" and "tail sanity" diagnostics, in a
        form that goes straight into a log or a plot.

        Returns mean alpha for the singleton-ish supports {0,1,2,3} explicitly
        plus `n_bins` log-spaced support quantile bins, and the Spearman-ish
        monotonicity check (rank correlation of n_eff with alpha over valid
        pairs). Expected: alpha increases with n_eff, with no bin at ~1 for
        support <= 3.
        """
        v = self.valid
        out: Dict[str, float] = {}
        if not bool(v.any()):
            return out
        n, a = self.n_eff[v], self.alpha[v]

        for k in (1, 2, 3):
            sel = (n >= k) & (n < k + 1)
            if bool(sel.any()):
                out[f"alpha_at_n{k}"] = float(a[sel].mean())
                out[f"count_at_n{k}"] = float(int(sel.sum()))
        zero = self.valid.logical_not() & (self.n_eff <= 0)
        out["alpha_at_n0"] = float(self.alpha[zero].mean()) if bool(zero.any()) \
            else 0.0

        # quantile bins of support -> mean alpha (the monotone trend to plot)
        qs = torch.linspace(0, 1, n_bins + 1, device=n.device)
        edges = torch.quantile(n, qs)
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            sel = (n >= lo) & (n <= hi) if b == n_bins - 1 else (n >= lo) & (n < hi)
            if bool(sel.any()):
                out[f"alpha_bin{b}_mean"] = float(a[sel].mean())
                out[f"alpha_bin{b}_n_eff_mean"] = float(n[sel].mean())

        # rank correlation between support and specialization
        if n.numel() > 2:
            rn = n.argsort().argsort().float()
            ra = a.argsort().argsort().float()
            rn = rn - rn.mean()
            ra = ra - ra.mean()
            denom = (rn.norm() * ra.norm()).clamp_min(self.cfg.eps)
            out["alpha_support_rank_corr"] = float((rn * ra).sum() / denom)
        return out

    @torch.no_grad()
    def export(self) -> Dict[str, torch.Tensor]:
        """CPU tensors."""
        return {k: getattr(self, k).detach().cpu().clone()
                for k in ("z_bar", "sigma2_concept", "tau2_global",
                          "sigma2", "tau2", "alpha", "n_eff",
                          "coverage", "seen", "q_mean", "n_raw")}

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"T={self.T}, K={self.K}, D={self.D}, ema={c.ema}, "
                f"alpha_mode={c.alpha_mode}, support={c.support}, "
                f"token_norm={c.token_norm}, "
                f"theta_normalize={c.theta_normalize}")


# ─────────────────────────────────────────────────────────────────────────────
# Prototype alignment diagnostic ("prototype alignment")
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def prototype_alignment(bank: ClassConceptPrototypeBank, shared
                        ) -> Dict[str, float]:
    """
    cos(z_bar_{y,k}, m_{y,k}) averaged over valid pairs, plus the mean
    distance the calibrated target actually travels away from the shared
    prior. If `theta_shift` is ~0 everywhere, CCSS is not specializing at all
    and alpha has saturated at zero.
    """
    v = bank.valid
    if not bool(v.any()):
        return {"prototype_cos_mean": float("nan"), "theta_shift_mean": 0.0}
    eps = bank.cfg.eps
    m_raw = shared.m_all().detach().float()
    zb_raw = bank.z_bar
    # cosine is scale-free: always computed on unit vectors
    cos = (unit(m_raw, eps) * unit(zb_raw, eps)).sum(-1)

    m, zb = (unit(m_raw, eps), unit(zb_raw, eps)) \
        if bank.cfg.theta_normalize == "unit" else (m_raw, zb_raw)
    a = bank.alpha.unsqueeze(-1)
    theta = m + a * (zb - m)
    shift = (theta - m).norm(dim=-1)
    span = (zb - m).norm(dim=-1).clamp_min(eps)
    return {"prototype_cos_mean": float(cos[v].mean()),
            "theta_shift_mean": float(shift[v].mean()),
            "theta_shift_max": float(shift[v].max()),
            "theta_shift_rel_mean": float((shift[v] / span[v]).mean()),
            "m_norm_mean": float(m_raw.norm(dim=-1)[v].mean()),
            "z_bar_norm_mean": float(zb_raw.norm(dim=-1)[v].mean())}
