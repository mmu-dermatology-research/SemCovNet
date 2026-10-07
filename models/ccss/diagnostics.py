"""
models/ccss/diagnostics.py
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

_BLOCK_RE = re.compile(r"(?:layer|layers|blocks?)\.(\d+)\.")


def _is_encoder_param(name: str) -> bool:
    return name.startswith("encoder.")


def _block_id(name: str) -> Optional[int]:
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else None


def _is_lora(name: str) -> bool:
    return ("lora_A" in name) or ("lora_B" in name)


# ─────────────────────────────────────────────────────────────────────────────
# §15.13 trainable-parameter accounting
# ─────────────────────────────────────────────────────────────────────────────
def parameter_report(model: nn.Module) -> Dict[str, object]:
    total = trainable = enc_total = enc_trainable = lora_trainable = 0
    semcov_trainable = 0
    blocks: Dict[int, int] = {}
    names: List[str] = []

    for n, p in model.named_parameters():
        num = p.numel()
        total += num
        enc = _is_encoder_param(n)
        enc_total += num if enc else 0
        if not p.requires_grad:
            continue
        trainable += num
        names.append(n)
        if enc:
            enc_trainable += num
            if _is_lora(n):
                lora_trainable += num
            b = _block_id(n)
            if b is not None:
                blocks[b] = blocks.get(b, 0) + num
        else:
            semcov_trainable += num

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": (trainable / total) if total else 0.0,
        "encoder_params": enc_total,
        "encoder_trainable_params": enc_trainable,
        "lora_trainable_params": lora_trainable,
        "semcov_trainable_params": semcov_trainable,
        "trainable_encoder_blocks": sorted(blocks),
        "trainable_param_names": names,
    }


def print_parameter_report(model: nn.Module, max_names: int = 12) -> Dict[str, object]:
    r = parameter_report(model)
    print("\n===== TRAINABLE PARAMETER ACCOUNTING =====")
    print(f"  total                 : {r['total_params'] / 1e6:9.2f} M")
    print(f"  trainable             : {r['trainable_params'] / 1e6:9.2f} M "
          f"({r['trainable_fraction']:.2%})")
    print(f"  trainable DINOv3      : {r['encoder_trainable_params'] / 1e6:9.2f} M "
          f"(of {r['encoder_params'] / 1e6:.2f} M)")
    print(f"    of which LoRA       : {r['lora_trainable_params'] / 1e6:9.2f} M")
    print(f"  trainable SemCovNet   : {r['semcov_trainable_params'] / 1e6:9.2f} M")
    print(f"  adapted encoder blocks: {r['trainable_encoder_blocks'] or '-'}")
    names = r["trainable_param_names"]
    enc = [n for n in names if _is_encoder_param(n)]
    if enc:
        print(f"  encoder tensors ({len(enc)}), first {min(max_names, len(enc))}:")
        for n in enc[:max_names]:
            print(f"      {n}")
    return r


# ─────────────────────────────────────────────────────────────────────────────
# §15.12 gradient-flow verification
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def gradient_flow_check(model: nn.Module, loss: torch.Tensor,
                        expect_mode: Optional[str] = None,
                        expect_blocks: Optional[Sequence[int]] = None,
                        verbose: bool = True) -> Dict[str, object]:
    """
    Backward `loss` (retaining the graph is the caller's business) and report
    which parameter families actually received gradients.

    Parameters
    ----------
    loss         : a scalar built from the model's outputs — pass L_CCSS alone
                   to verify the §15.1 claim specifically.
    expect_mode  : 'frozen' | 'lora' | 'lora_last3' | 'partial' | 'full_last3'
                   | 'last4' | 'full'. When given, the report contains a
                   boolean `ok` and a list of violations.
    expect_blocks: encoder block ids allowed to have gradients.
    """
    _zero_grads(model)
    loss.backward(retain_graph=True)

    got: Dict[str, float] = {}
    enc_blocks, lora_hit, enc_nonlora_hit, semcov_hit = set(), 0, 0, 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().abs().sum())
        if g == 0.0:
            continue
        got[n] = g
        if _is_encoder_param(n):
            b = _block_id(n)
            if b is not None:
                enc_blocks.add(b)
            if _is_lora(n):
                lora_hit += 1
            else:
                enc_nonlora_hit += 1
        else:
            semcov_hit += 1

    mode = (expect_mode or "").lower()
    violations: List[str] = []
    if mode:
        if semcov_hit == 0:
            violations.append("no SemCovNet parameter received a gradient")
        if mode in ("frozen", "freeze", "none"):
            if lora_hit or enc_nonlora_hit:
                violations.append(
                    f"frozen encoder received gradients "
                    f"(lora={lora_hit}, other={enc_nonlora_hit})")
        elif mode.startswith("lora"):
            if lora_hit == 0:
                violations.append("no LoRA parameter received a gradient")
            if enc_nonlora_hit:
                violations.append(
                    f"{enc_nonlora_hit} non-LoRA encoder tensors received "
                    f"gradients in a LoRA run")
        elif mode in ("partial", "last3", "full_last3", "last4", "full_last4"):
            if enc_nonlora_hit == 0:
                violations.append("no encoder block parameter received a gradient")
        elif mode in ("full", "full_ft", "finetune"):
            if enc_nonlora_hit == 0:
                violations.append("full fine-tuning but no encoder gradient")
        if expect_blocks is not None and enc_blocks:
            extra = sorted(set(enc_blocks) - set(expect_blocks))
            if extra:
                violations.append(f"gradients in unexpected blocks {extra}")

    report = {
        "n_params_with_grad": len(got),
        "encoder_blocks_with_grad": sorted(enc_blocks),
        "lora_tensors_with_grad": lora_hit,
        "encoder_nonlora_tensors_with_grad": enc_nonlora_hit,
        "semcov_tensors_with_grad": semcov_hit,
        "violations": violations,
        "ok": not violations,
    }
    if verbose:
        print("\n===== GRADIENT FLOW CHECK =====")
        print(f"  expected mode         : {expect_mode or '(unchecked)'}")
        print(f"  SemCovNet tensors     : {semcov_hit}")
        print(f"  LoRA tensors          : {lora_hit}")
        print(f"  other encoder tensors : {enc_nonlora_hit}")
        print(f"  encoder blocks        : {report['encoder_blocks_with_grad'] or '-'}")
        print(f"  result                : {'OK' if report['ok'] else 'FAILED'}")
        for v in violations:
            print(f"      ! {v}")
    _zero_grads(model)
    return report


def shared_structure_report(model: nn.Module) -> Dict[str, float]:
    """collapse diagnostic, tolerant of models without CCSS."""
    shared = getattr(model, "shared", None)
    if shared is None or not hasattr(shared, "collapse_report"):
        return {}
    return shared.collapse_report()
