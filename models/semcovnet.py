"""
models/semcovnet.py
─────────────────────────────────────────────────────────────────────────────
SemCovNet Framework
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ccss.diagnostics import (gradient_flow_check, parameter_report,
                               print_parameter_report)
from .ccss.losses import CCSSLoss, CCSSLossConfig
from .ccss.prototypes import (ClassConceptPrototypeBank, PrototypeBankConfig,
                              prototype_alignment)
from .ccss.shared_structure import (SharedSemanticStructure,
                                    SharedStructureConfig)
from .encoders.base_encoder import BaseVisualEncoder
from .semcov_model import (MODEL_ALIASES, MODEL_CONFIGS, SemCovConfig,
                           SemCovPhase7)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SemCovNetConfig(SemCovConfig):
    """ configuration + mechanism."""

    # ──  shared semantic structure ───────────────────────────────
    use_shared: bool = True
    shared_mode: str = "global_class_concept"   # ablation
    shared_normalize: str = "none"              # corrected primary: mix in the
                                                # native space ('layernorm' is
                                                # now an ablation)
    shared_center: bool = True                  # identifiability centering of
                                                # mu_y and mu_k

    # ── prototype bank / calibration ───────────────────────
    use_ccss: bool = True
    # ccss | ccss_hetero | zero | one | gate
    #      (+  freq | cov | eff | n_sigma)
    alpha_mode: str = "ccss"
    proto_ema: float = 0.95                     # beta
    proto_token_norm: str = "none"              # corrected primary (was 'l2')
    proto_theta_normalize: str = "none"         # corrected primary (was 'unit')
    support_mode: str = "sum"                   # primary 'sum' | ablation 'ess'
    sigma_floor: float = 1e-6                   # numerical safeguards
    tau_floor: float = 1e-6
    gate_lambda: float = 0.1                    # B4 heuristic gate only
    alpha_kappa: float = 1.0                    # B2 (alpha_mode='eff')
    evidence_min_weight: float = 0.0
    n_eff_scale: float = 1.0                    # 1.0
    calibrate_alpha: float = 0.0                # 0 = primary; >0 is a diagnostic
                                                # appendix experiment only

    # ──  CCSS objective ─────────────────────────────────────────
    ccss_loss: str = "align"                    # align | contrastive | none
    ccss_temperature: float = 0.1               # tau_c, contrastive only
    ccss_neg_classes: int = 0                   # 0 = all classes
    lambda_ccss: float = 0.25                   # search {0.1,.25,.5,1}


SEMCOVNET_CONFIGS: Dict[str, Dict] = {
    # concept supervision only — the M1 / M3 control
    "semcov_concept": dict(use_router=True, use_concept_loss=True,
                           use_residual=True, use_semantic=True,
                           use_shared=False, use_ccss=False,
                           ccss_loss="none", lambda_ccss=0.0),
    # B2 — shared semantic structure, no specialization
    "semcov_shared": dict(use_router=True, use_concept_loss=True,
                          use_residual=True, use_semantic=True,
                          use_shared=True, use_ccss=True, alpha_mode="zero"),
    # B3 — fixed class-concept specialization, no statistical calibration
    "semcov_fixed": dict(use_router=True, use_concept_loss=True,
                         use_residual=True, use_semantic=True,
                         use_shared=True, use_ccss=True, alpha_mode="one"),
    # B4 — the earlier heuristic coverage gate g(C) = C / (C + lambda)
    "semcov_gate": dict(use_router=True, use_concept_loss=True,
                        use_residual=True, use_semantic=True,
                        use_shared=True, use_ccss=True, alpha_mode="gate"),
    # B5 / B6 / B7 — the proposed model (adaptation is a separate flag)
    # alpha = n_eff tau^2 / (n_eff tau^2 + sigma_k^2): pooled concept
    # observation variance, one global specialization variance
    "semcovnet": dict(use_router=True, use_concept_loss=True,
                      use_residual=True, use_semantic=True,
                      use_shared=True, use_ccss=True, alpha_mode="ccss"),
    # ablation — the richer pair-specific/heteroscedastic estimator
    # alpha = n_eff tau_k^2 / (n_eff tau_k^2 + sigma^2_{y,k}).
    # Identical in every other respect to `semcovnet`, so the comparison
    # isolates the estimator rather than the architecture.
    "semcov_hetero": dict(use_router=True, use_concept_loss=True,
                          use_residual=True, use_semantic=True,
                          use_shared=True, use_ccss=True,
                          alpha_mode="ccss_hetero"),

    # ──  what information should control specialization? ────────
    # Identical architecture, identical losses, identical schedule. The ONLY
    # difference between these and "semcovnet" is how alpha_{y,k} is formed,
    # which is what makes the comparison a statement about the rule rather
    # than about capacity.
    "semcov_freq": dict(use_router=True, use_concept_loss=True,      # B0
                        use_residual=True, use_semantic=True,
                        use_shared=True, use_ccss=True, alpha_mode="freq"),
    "semcov_cov": dict(use_router=True, use_concept_loss=True,       # B1
                       use_residual=True, use_semantic=True,
                       use_shared=True, use_ccss=True, alpha_mode="cov"),
    "semcov_eff": dict(use_router=True, use_concept_loss=True,       # B2
                       use_residual=True, use_semantic=True,
                       use_shared=True, use_ccss=True, alpha_mode="eff"),
    "semcov_nsigma": dict(use_router=True, use_concept_loss=True,    # B3
                          use_residual=True, use_semantic=True,
                          use_shared=True, use_ccss=True,
                          alpha_mode="n_sigma"),
}

SEMCOVNET_ALIASES = {
    "B3_fixed": "semcov_fixed",
    "B4_gate": "semcov_gate",
    "B5_hetero": "semcov_hetero", "hetero": "semcov_hetero",
    "B5": "semcovnet", "B6": "semcovnet", "B7": "semcovnet",
    "B0_freq": "semcov_freq", "B1_cov": "semcov_cov",
    "B2_eff": "semcov_eff", "B3_nsigma": "semcov_nsigma",
    "M1": "semcov_concept", "M3": "semcov_concept",
    "M2": "semcovnet", "M4": "semcovnet", "M5": "semcovnet",
}

ALL_MODEL_CONFIGS = {**MODEL_CONFIGS, **SEMCOVNET_CONFIGS}
ALL_MODEL_ALIASES = {**MODEL_ALIASES, **SEMCOVNET_ALIASES}


# ─────────────────────────────────────────────────────────────────────────────
# Training schedule
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CCSSSchedule:
    """
    Epoch -> (CCSS on?, encoder adapting?, initialise statistics?).

    Two-stage (default)

        A  epochs 1 .. warmup            frozen DINO, L_task + lambda_c L_concept
        B  once, after epoch `warmup`    initialise z_bar, sigma^2, tau^2, alpha
        C  epochs warmup+1 .. end        + lambda_CCSS L_CCSS, DINO adapting

    Three-stage (For Derm7pt)

        A  epochs 1 .. warmup            frozen DINO, no CCSS
        B' epochs warmup+1 .. adapt_at   frozen DINO, CCSS on (targets settle)
        C  epochs adapt_at+1 .. end      LoRA-last3 enabled, CCSS on

    "Do not activate DINO adaptation at epoch 0." Warm-up is 5-10% of
    training (3-5 epochs out of 50).
    """
    warmup_epochs: int = 3          # end of Stage A
    encoder_start_epoch: int = 0    # first epoch with DINO adaptation (0 = warmup)
    total_epochs: int = 20
    three_stage: bool = False
    ccss_stat_epochs: int = 2       # Stage B' length when three_stage
    enabled: bool = True            # False for the coverage-unaware
                                    # models, which have no CCSS to stage

    def __post_init__(self):
        if self.encoder_start_epoch <= 0:
            self.encoder_start_epoch = (
                self.warmup_epochs + self.ccss_stat_epochs if self.three_stage
                else self.warmup_epochs)

    def ccss_on(self, epoch: int) -> bool:
        """`epoch` is 1-based."""
        return self.enabled and epoch > self.warmup_epochs

    def encoder_on(self, epoch: int) -> bool:
        return epoch > self.encoder_start_epoch

    def init_statistics_before(self, epoch: int) -> bool:
        """Stage B fires once, immediately before the first CCSS epoch."""
        return self.enabled and epoch == self.warmup_epochs + 1

    def stage(self, epoch: int) -> str:
        if not self.enabled:
            # coverage-unaware run: the only thing that stages is the encoder
            return "C" if self.encoder_on(epoch) else "A"
        if not self.ccss_on(epoch):
            return "A"
        return "C" if self.encoder_on(epoch) else "B"

    def describe(self) -> str:
        if not self.enabled:
            return (f"[schedule] coverage-unaware run: encoder adaptation "
                    f"from epoch {self.encoder_start_epoch + 1}")
        return (f"[schedule] warm-up {self.warmup_epochs}ep -> CCSS from ep "
                f"{self.warmup_epochs + 1} -> encoder adaptation from ep "
                f"{self.encoder_start_epoch + 1} "
                f"({'three' if self.three_stage else 'two'}-stage, "
                f"{self.total_epochs} total)")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class SemCovNet(SemCovPhase7):
    """
    Parameters
    ----------
    encoder    : BaseVisualEncoder (DINOv3Encoder in practice)
    n_classes  : T
    n_concepts : K
    cfg        : SemCovNetConfig
    """

    def __init__(self, encoder: BaseVisualEncoder, n_classes: int,
                 n_concepts: int = 0, cfg: Optional[SemCovNetConfig] = None):
        cfg = cfg or SemCovNetConfig()
        # the CCSS mechanism lives on concept tokens, so the router is required
        if (cfg.use_shared or cfg.use_ccss) and not cfg.use_router:
            raise ValueError("SemCovNet needs the Visual Concept Router "
                             "(use_router=True) to produce z_{i,k}.")
        super().__init__(encoder, n_classes, n_concepts, cfg)
        self.cfg: SemCovNetConfig = cfg

        d = cfg.d_sem
        self.T, self.K = int(n_classes), int(n_concepts)

        self.shared: Optional[SharedSemanticStructure] = (
            SharedSemanticStructure(
                self.T, self.K, d,
                SharedStructureConfig(mode=cfg.shared_mode,
                                      normalize=cfg.shared_normalize,
                                      center=cfg.shared_center))
            if cfg.use_shared and self.K > 0 else None)

        self.bank: Optional[ClassConceptPrototypeBank] = (
            ClassConceptPrototypeBank(
                self.T, self.K, d,
                PrototypeBankConfig(ema=cfg.proto_ema,
                                    sigma_floor=cfg.sigma_floor,
                                    tau_floor=cfg.tau_floor,
                                    token_norm=cfg.proto_token_norm,
                                    theta_normalize=cfg.proto_theta_normalize,
                                    support=cfg.support_mode,
                                    alpha_mode=cfg.alpha_mode,
                                    gate_lambda=cfg.gate_lambda,
                                    alpha_kappa=cfg.alpha_kappa,
                                    min_weight=cfg.evidence_min_weight,
                                    n_eff_scale=cfg.n_eff_scale))
            if cfg.use_ccss and self.K > 0 else None)

        self.ccss_criterion = CCSSLoss(CCSSLossConfig(
            mode=(cfg.ccss_loss if cfg.use_ccss else "none"),
            temperature=cfg.ccss_temperature,
            min_weight=cfg.evidence_min_weight,
            neg_classes=cfg.ccss_neg_classes))

        # stage flag: CCSS is dark until the schedule turns it on
        self.register_buffer("_ccss_active", torch.zeros((), dtype=torch.bool))
        # evaluation-only coverage bands, kept for logging
        self.register_buffer("coverage_bands",
                             torch.full((self.T, max(self.K, 1)), -1,
                                        dtype=torch.long))

        if self.shared is not None:
            print(f"[semcovnet] shared structure: {self.shared.extra_repr()}")
        if self.bank is not None:
            print(f"[semcovnet] prototype bank : {self.bank.extra_repr()}")
            print(f"[semcovnet] CCSS loss      : "
                  f"{self.ccss_criterion.extra_repr()}  "
                  f"lambda_CCSS={cfg.lambda_ccss}")

    # ── stage control ───────────────────────────────────────────────────
    @property
    def ccss_available(self) -> bool:
        return self.bank is not None and self.shared is not None

    @property
    def ccss_active(self) -> bool:
        return bool(self._ccss_active) and self.ccss_available

    def set_ccss_active(self, flag: bool) -> None:
        if flag and not self.ccss_available:
            raise RuntimeError("this configuration has no CCSS mechanism "
                               "(use_shared / use_ccss are off)")
        self._ccss_active.fill_(bool(flag))

    @property
    def needs_tokens(self) -> bool:
        """Whether forward() has to materialise Z (train-time only)."""
        return self.ccss_available and (self.ccss_active or self.training)

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, **kw) -> Dict[str, object]:
        """
        Identical contract to SemCovPhase7.forward(); concept tokens are
        materialised automatically whenever the CCSS machinery needs them.
        """
        if kw.get("return_tokens") is None and self.needs_tokens:
            kw["return_tokens"] = True
        return super().forward(x, **kw)

    # ──  statistics maintenance ─────────────────────────────
    @torch.no_grad()
    def update_statistics(self, Z: Optional[torch.Tensor],
                          y: torch.Tensor, a: Optional[torch.Tensor]) -> None:
        """
        Per-minibatch EMA update with detached tokens.
        Call AFTER optimizer.step() so the prototypes reflect the parameters
        that produced them.
        """
        if self.bank is None or Z is None or a is None or a.numel() == 0:
            return
        self.bank.update(Z, y, a)

    @torch.no_grad()
    def refresh_statistics(self, diagnostics: bool = False) -> Dict[str, float]:
        """
        Recompute sigma_k^2, tau^2, alpha and therefore theta_hat.
        once per epoch (or every N iterations), never per minibatch.

        `diagnostics=True` additionally returns the specialization-vs-
        support curve and tail sanity numbers (more keys; intended for the
        once-per-stage call, not every epoch).
        """
        if not self.ccss_available:
            return {}
        logs = self.bank.refresh(self.shared)
        bands = self.coverage_bands
        if int(bands.max()) >= 0:
            logs = self.bank.mechanism_report(bands)
        logs.update(prototype_alignment(self.bank, self.shared))
        if diagnostics:
            logs.update(self.bank.support_alpha_diagnostic())
        return logs

    @torch.no_grad()
    def set_support(self, n_eff) -> None:
        """Install the exact n^eff_{y,k} matrix."""
        if self.bank is not None:
            self.bank.set_support(n_eff)
            
    @torch.no_grad()
    def set_raw_count(self, n_raw) -> None:
        """
        Install n_{y,k} = #{i : y_i = y, a_{i,k} > 0}.

        Forwards to the bank, like `set_support` and `set_coverage`. A no-op
        for models without a CCSS bank, so the caller does not have to know
        which variant it is holding.
        """
        if self.bank is not None:
            self.bank.set_raw_count(n_raw)

    @torch.no_grad()
    def set_coverage(self, C, bands=None) -> None:
        """
        Store C_{y,k} (analysis artefact, and the input to alpha_mode
        'gate'). `bands` is the evaluation-only 0/1/2 Tail/Middle/Head map,
        used for logging alpha_tail vs alpha_head, never by the mechanism.
        """
        if self.bank is not None:
            self.bank.set_coverage(C)
        if bands is not None:
            b = torch.as_tensor(bands, dtype=torch.long,
                                device=self.coverage_bands.device)
            self.coverage_bands.copy_(b)

    @torch.no_grad()
    def initialise_statistics(self, loader, device, autocast=None,
                              max_batches: int = 0, verbose: bool = True
                              ) -> Dict[str, float]:
        """
        Stage - initialise the semantic statistics from TRAINING
        data only, with an exact weighted pass rather than an EMA transient:

            1. extract concept tokens
            2. z_bar_{y,k}      exact weighted mean
            3. sigma_k^2        pooled concept observation variance
            4. tau^2            global specialization variance
            5. n^eff_{y,k}      (already exact via set_support)
            6. alpha_{y,k}
            7. theta_hat_{y,k}

        "No test/validation information may enter these statistics."
        """
        if not self.ccss_available:
            return {}
        was_training = self.training
        self.eval()
        self.bank.begin_accumulation()
        need_support = not bool(self.bank.support_committed)

        n_batches = 0
        for batch in loader:
            _, data, target = batch
            images = data["image"].to(device, non_blocking=True)
            a = data["desc_soft"]
            if a is None or a.numel() == 0:
                continue
            a = a.to(device, non_blocking=True)
            y = target.to(device, non_blocking=True)
            ctx = autocast() if callable(autocast) else _null_ctx()
            with ctx:
                out = self.forward(images, return_tokens=True)
            Z = out.get("concept_tokens")
            if Z is None:
                continue
            self.bank.accumulate(Z.float(), y, a)
            if need_support:
                self.bank.accumulate_support(y, a)
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break

        self.bank.commit_accumulation()
        if need_support:
            self.bank.commit_support()
        self.refresh_statistics()
        if (self.cfg.calibrate_alpha > 0
                and self.cfg.alpha_mode in ("ccss", "ccss_hetero")):
            # failure mode safeguard: DIAGNOSTIC / appendix only, off in the
            # primary configuration. Opt-in and applied exactly once.
            self.cfg.n_eff_scale = self.bank.calibrate_support_scale(
                self.cfg.calibrate_alpha)
        logs = self.refresh_statistics(diagnostics=True)
        if was_training:
            self.train()
        if verbose:
            v = int(self.bank.valid.sum())
            nan = float("nan")
            print(f"[stage B] statistics initialised from {n_batches} training "
                  f"batches: {v}/{self.T * self.K} valid class-concept pairs, "
                  f"mean alpha={logs.get('alpha_mean', nan):.4f}, "
                  f"mean sigma_k^2={logs.get('sigma2_concept_mean', nan):.3e}, "
                  f"tau^2={logs.get('tau2_global', nan):.3e}")
            # tail sanity: a singleton pair sitting at alpha ~ 1 is a
            # failure of the estimator, not a property of the data.
            tail_a = logs.get("alpha_tail_support_mean")
            if tail_a is not None:
                frac = logs.get("alpha_tail_support_frac_above_0.9", 0.0)
                print(f"[stage B] tail check (n_eff <= 3): "
                      f"{int(logs.get('n_pairs_support_le3', 0))} pairs, "
                      f"mean alpha={tail_a:.4f}, frac(alpha>0.9)={frac:.3f}")
                if frac > 0.5:
                    print("[stage B] WARNING: most low-support pairs are near "
                          "full specialization — inspect sigma_k^2 / tau^2 "
                          "before launching experiments (tail sanity).")
            rc = logs.get("alpha_support_rank_corr")
            if rc is not None:
                print(f"[stage B] alpha-vs-support rank corr = {rc:+.3f} "
                      f"(expected clearly positive)")
        return logs

    # ──  the CCSS term ─────────────────────────────────────────
    def ccss_objective(self, out: Dict[str, object], y: torch.Tensor,
                       a: Optional[torch.Tensor]
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_CCSS for one minibatch.

        Returns a zero (grad-free) tensor whenever CCSS is inactive, so the
        training loop needs no branching and warm-up epochs cost nothing.
        """
        Z = out.get("concept_tokens")
        if not self.ccss_active or Z is None or a is None or a.numel() == 0:
            dev = y.device if torch.is_tensor(y) else None
            return torch.zeros((), device=dev), {}

        y = y.to(torch.long)
        if self.cfg.ccss_loss == "contrastive":
            classes = self._negative_classes(y)
            theta_all = self.bank.theta_all(self.shared, classes)
            return self.ccss_criterion(Z, None, a, y_idx=y,
                                       theta_all=theta_all,
                                       valid_classes=classes)
        theta = self.bank.theta(y, self.shared)          # [B, K, D]
        return self.ccss_criterion(Z, theta, a)

    def _negative_classes(self, y: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Class subset for contrastive CCSS. All classes by default; with
        `ccss_neg_classes > 0` the batch's own classes are always kept and the
        rest are sampled, which keeps CUB's [B,K,T] similarity tensor small.
        """
        n = int(self.cfg.ccss_neg_classes)
        if n <= 0 or n >= self.T:
            return None
        own = torch.unique(y)
        pool = torch.randperm(self.T, device=y.device)[:n]
        return torch.unique(torch.cat([own, pool]))

    def total_objective(self, out: Dict[str, object], y: torch.Tensor,
                        a: Optional[torch.Tensor], task_criterion,
                        concept_criterion=None
                        ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        the objective:

            L = L_task + lambda_concept L_concept + lambda_CCSS L_CCSS

        Kept here so training and any future script cannot drift apart on
        what "the SemCovNet loss" means.
        """
        cfg = self.cfg
        l_task = task_criterion(out["task_logits"].float(), y)

        if (concept_criterion is not None
                and out.get("concept_logits") is not None and a is not None
                and a.numel()):
            l_con = concept_criterion(out["concept_logits"].float(), a)
        else:
            l_con = torch.zeros((), device=l_task.device)

        l_ccss, logs = self.ccss_objective(out, y, a)
        total = (l_task
                 + cfg.lambda_concept * l_con
                 + cfg.lambda_ccss * l_ccss)
        logs.update({"loss": float(total.detach()),
                     "task_loss": float(l_task.detach()),
                     "concept_loss": float(l_con.detach()),
                     "ccss_loss": float(l_ccss.detach())})
        return total, logs

    # ── optimiser groups ────────────────────────────────────────
    def param_groups(self, lr_semcov: float = 3e-4,
                     lr_lora: Optional[float] = None,
                     lr_dino: Optional[float] = None,
                     weight_decay: float = 0.05,
                     no_decay_norm_bias: bool = True) -> List[dict]:
        """
        Group:  new SemCovNet parameters (incl. mu_0, mu_y, mu_k)  1e-4..3e-4
        Group: bias / LayerNorm parameters                        no decay
        Group:  LoRA                                               5e-5..2e-4
        Group:  fully unfrozen DINO blocks                         0.05-0.1x

        For explicit grouping so the weight-decay choices are
        reproducible; forbids using one learning rate for pretrained
        DINO weights and freshly initialised modules.
        """
        enc_ids = {id(p) for p in self.encoder.parameters()}
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad or id(p) in enc_ids:
                continue
            if no_decay_norm_bias and (p.ndim <= 1 or n.endswith(".bias")):
                no_decay.append(p)
            else:
                decay.append(p)

        groups = [{"name": "semcov", "params": decay, "lr": lr_semcov,
                   "weight_decay": weight_decay}]
        if no_decay:
            groups.append({"name": "semcov_nodecay", "params": no_decay,
                           "lr": lr_semcov, "weight_decay": 0.0})

        enc_groups = (self.encoder.param_groups()
                      if hasattr(self.encoder, "param_groups") else {})
        if enc_groups.get("lora"):
            groups.append({"name": "lora", "params": enc_groups["lora"],
                           "lr": lr_lora if lr_lora is not None else lr_semcov,
                           "weight_decay": 0.0})
        if enc_groups.get("dino"):
            groups.append({"name": "dino", "params": enc_groups["dino"],
                           "lr": (lr_dino if lr_dino is not None
                                  else 0.05 * lr_semcov),
                           "weight_decay": weight_decay})
        for g in groups:
            print(f"[optim] group '{g['name']}': "
                  f"{sum(p.numel() for p in g['params']) / 1e6:.2f}M params, "
                  f"lr={g['lr']:.2e}, wd={g['weight_decay']}")
        return [g for g in groups if len(g["params"]) > 0]

    # ── reporting / checkpointing ───────────────────────
    def parameter_report(self) -> Dict[str, object]:
        return parameter_report(self)

    def print_parameter_report(self) -> Dict[str, object]:
        return print_parameter_report(self)

    def verify_gradient_flow(self, loss: torch.Tensor,
                             expect_mode: Optional[str] = None,
                             expect_blocks: Optional[Sequence[int]] = None
                             ) -> Dict[str, object]:
        """Run this once before launching a full experiment."""
        return gradient_flow_check(self, loss, expect_mode, expect_blocks)

    @torch.no_grad()
    def ccss_state(self) -> Dict[str, object]:
        """
        The analysis artefacts wants in every checkpoint: prototype
        bank, variance bank, effective support, alpha and the coverage matrix.
        """
        if self.bank is None:
            return {}
        state = self.bank.export()
        state["coverage_bands"] = self.coverage_bands.detach().cpu().clone()
        state["cfg"] = asdict(self.cfg)
        return state

    def extra_repr(self) -> str:
        return (f"T={self.T}, K={self.K}, d_sem={self.cfg.d_sem}, "
                f"alpha_mode={self.cfg.alpha_mode}, "
                f"ccss_loss={self.cfg.ccss_loss}")


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Builder — one entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_name: str, encoder: BaseVisualEncoder, n_classes: int,
                n_concepts: int = 0, **cfg_overrides):
    """
    build_model('semcovnet', encoder, n_classes=200, n_concepts=312,
                d_sem=256, lambda_concept=0.5, lambda_ccss=0.25)
    """
    name = ALL_MODEL_ALIASES.get(model_name, model_name)

    if name in MODEL_CONFIGS and name not in SEMCOVNET_CONFIGS:
        from .semcov_model import build_model as build_phase7
        return build_phase7(name, encoder, n_classes, n_concepts,
                            **{k: v for k, v in cfg_overrides.items()
                               if hasattr(SemCovConfig, k)
                               or k in SemCovConfig.__dataclass_fields__})

    if name not in SEMCOVNET_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from "
                         f"{list(ALL_MODEL_CONFIGS)} "
                         f"(aliases {list(ALL_MODEL_ALIASES)})")

    cfg = SemCovNetConfig(**SEMCOVNET_CONFIGS[name])
    for k, v in cfg_overrides.items():
        if v is None:
            continue
        if k not in SemCovNetConfig.__dataclass_fields__:
            raise TypeError(f"Unknown SemCovNetConfig field: {k}")
        setattr(cfg, k, v)

    # a configuration without CCSS must not carry a CCSS loss weight
    if not cfg.use_ccss:
        cfg.lambda_ccss = 0.0
        cfg.ccss_loss = "none"

    print(f"[model] building '{name}' (alpha_mode={cfg.alpha_mode}, "
          f"shared={cfg.use_shared}, ccss={cfg.use_ccss})")
    return SemCovNet(encoder, n_classes, n_concepts, cfg)


def is_semcovnet(model_name: str) -> bool:
    return ALL_MODEL_ALIASES.get(model_name, model_name) in SEMCOVNET_CONFIGS
