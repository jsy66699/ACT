"""
Batched (per-row-independent) Auto-PGD attack family for ACTFuzzer's BI
two-stage mutation (see actfuzzer.py's `_mutate_hpgd_then_pgd`).

These are direct ports of the single-sample (batch=1) functions in
pattern_search_pgd.py, vectorized over a batch dimension B so every sample
gets its own independent step size / best-loss / target class instead of
sharing one scalar. `_attack_pgd`/`_violation_score` from pattern_search_pgd.py
are reused UNCHANGED -- they were already written batch-generically (pure
gather/scatter/max ops, one shared step-size float), unlike the DLR/CW loss
functions and `_auto_pgd_core`, which assumed B=1 (`outputs[0]`, python
scalar bookkeeping) and needed real rewrites here.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from act.front_end.specs import OutKind, OutputSpec


def _dlr_loss_untargeted_batched(outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Untargeted DLR margin (Croce & Hein, 2020), per row. Returns Tensor[B]."""
    y = y.clamp(min=0).view(-1, 1)
    zy = outputs.gather(1, y).squeeze(1)
    other = outputs.scatter(1, y, float("-inf"))
    z_best_other = other.max(dim=1).values
    z_sorted, _ = torch.sort(outputs, dim=1, descending=True)
    C = outputs.shape[1]
    z1 = z_sorted[:, 0]
    z3 = z_sorted[:, min(2, C - 1)]
    denom = (z1 - z3).clamp(min=1e-12)
    return (z_best_other - zy) / denom


def _cw_loss_batched(outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Carlini-Wagner untargeted margin (max_{i!=y} z_i - z_y), per row. Returns Tensor[B]."""
    y = y.clamp(min=0).view(-1, 1)
    zy = outputs.gather(1, y).squeeze(1)
    other = outputs.scatter(1, y, float("-inf"))
    return other.max(dim=1).values - zy


def _dlr_loss_targeted_batched(outputs: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Targeted DLR margin, per row -- `t` is a per-row target class Tensor[B]
    (each row may target a DIFFERENT class), unlike the single-sample version's
    shared python int. Returns Tensor[B]."""
    y = y.clamp(min=0).view(-1, 1)
    t = t.view(-1, 1)
    zy = outputs.gather(1, y).squeeze(1)
    zt = outputs.gather(1, t).squeeze(1)
    z_sorted, _ = torch.sort(outputs, dim=1, descending=True)
    C = outputs.shape[1]
    z1 = z_sorted[:, 0]
    z3 = z_sorted[:, min(2, C - 1)]
    z4 = z_sorted[:, min(3, C - 1)]
    denom = (z1 - (z3 + z4) / 2).clamp(min=1e-12)
    return (zt - zy) / denom


def _auto_pgd_core_batched(
    x0: torch.Tensor, model: nn.Module, lb: torch.Tensor, ub: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor], steps: int,
) -> torch.Tensor:
    """Per-row-independent Auto-PGD (Croce & Hein, 2020, Algorithm 1): momentum
    + adaptive step-size halving, vectorized over the batch dimension. Every
    step runs forward+backward on the FULL unmasked batch (`loss.sum()` ->
    one `torch.autograd.grad` call), which gives correct independent per-row
    gradients since standard eval-mode layers (Linear/Conv2d/ReLU, no
    train-mode BatchNorm) never mix rows -- the same assumption
    PGDMutation/HPGDMutation/_attack_pgd already rely on. Checkpoint-triggered
    halving is applied only to DETACHED per-row bookkeeping tensors via
    `torch.where`, never inside the backward step itself (same idiom already
    used by HPGDPullbackMutation.mutate).

    Returns the best point (by `loss_fn`) visited per row, projected into
    [lb, ub]."""
    from act.pipeline.fuzzing.pattern_search_pgd import _forward, _project_box

    if steps <= 0:
        return x0.detach()

    B = x0.shape[0]
    device = x0.device
    bshape = (B,) + (1,) * (x0.dim() - 1)

    alpha = (ub - lb).clamp(min=1e-12).clone()
    x_prev = x0.detach().clone()
    x_cur = x0.detach().clone()
    with torch.no_grad():
        best_loss = loss_fn(_forward(model, x_cur)).detach().clone()
    x_best = x_cur.clone()

    checkpoints = sorted({max(1, int(round(f * steps))) for f in
                           (0.22, 0.41, 0.55, 0.66, 0.75, 0.83, 0.90, 0.95, 1.0)})
    last_ckpt = 0
    success_count = torch.zeros(B, device=device)
    loss_at_last_ckpt = best_loss.clone()
    alpha_changed_last = torch.zeros(B, dtype=torch.bool, device=device)

    for k in range(steps):
        x_req = x_cur.detach().clone().requires_grad_(True)
        loss = loss_fn(_forward(model, x_req))
        grad = torch.autograd.grad(loss.sum(), x_req)[0]

        z = _project_box(x_cur.detach() + alpha * torch.sign(grad), lb, ub)
        if k == 0:
            x_next = z
        else:
            x_next = _project_box(
                x_cur.detach() + 0.75 * (z - x_cur.detach()) + 0.25 * (x_cur.detach() - x_prev), lb, ub,
            )

        with torch.no_grad():
            loss_next = loss_fn(_forward(model, x_next)).detach()

        improved = loss_next > best_loss
        best_loss = torch.where(improved, loss_next, best_loss)
        x_best = torch.where(improved.view(*bshape), x_next, x_best)
        success_count = success_count + (loss_next > loss.detach()).float()

        x_prev = x_cur.detach().clone()
        x_cur = x_next.detach()

        if (k + 1) in checkpoints:
            interval = (k + 1) - last_ckpt
            enough_progress = success_count >= 0.75 * interval
            ckpt_improved = best_loss > loss_at_last_ckpt + 1e-9
            halve = (~enough_progress) | (~ckpt_improved & ~alpha_changed_last)
            hb = halve.view(*bshape)
            alpha = torch.where(hb, (alpha / 2).clamp(min=1e-12), alpha)
            x_cur = torch.where(hb, x_best, x_cur)
            x_prev = torch.where(hb, x_best, x_prev)
            alpha_changed_last = halve
            success_count = torch.zeros(B, device=device)
            last_ckpt = k + 1
            loss_at_last_ckpt = best_loss.clone()

    return x_best


def _batched_apgd_t(
    x0: torch.Tensor, model: nn.Module, lb: torch.Tensor, ub: torch.Tensor,
    output_spec: Optional[OutputSpec], label: torch.Tensor, steps: int, n_target_classes: int,
) -> torch.Tensor:
    """Targeted Auto-PGD, batched: for `n_target_classes` rounds, each row
    targets its OWN r-th-ranked other class (by that row's natural,
    pre-attack logits), running one full `_auto_pgd_core_batched` call per
    round. Keeps each row's best-by-`_violation_score` result across rounds
    (loses the single-sample script's per-row early-exit on first success --
    the caller's own property check afterward still catches whichever round's
    result actually violates, so this only costs extra compute, not
    correctness)."""
    from act.pipeline.fuzzing.pattern_search_pgd import _forward, _violation_score

    B = x0.shape[0]
    bshape = (B,) + (1,) * (x0.dim() - 1)
    with torch.no_grad():
        logits0 = _forward(model, x0).clone()
    logits0.scatter_(1, label.clamp(min=0).view(-1, 1), float("-inf"))
    k = min(max(n_target_classes, 1), logits0.shape[1] - 1)
    target_classes = torch.topk(logits0, k=k, dim=1).indices  # [B, k]

    best_x = x0.detach().clone()
    best_score = torch.full((B,), float("-inf"), device=x0.device)
    for r in range(k):
        t = target_classes[:, r]
        x_cand = _auto_pgd_core_batched(
            x0, model, lb, ub, lambda out, t=t: _dlr_loss_targeted_batched(out, label, t), steps,
        )
        with torch.no_grad():
            score = _violation_score(output_spec, _forward(model, x_cand), label)
        improve = score > best_score
        best_score = torch.where(improve, score, best_score)
        best_x = torch.where(improve.view(*bshape), x_cand, best_x)
    return best_x


def _batched_pgd_family_attack(
    x0: torch.Tensor, model: nn.Module, lb: torch.Tensor, ub: torch.Tensor,
    output_spec: Optional[OutputSpec], label: torch.Tensor, steps: int,
    attack_strategy: str, apgd_t_target_classes: int = 5,
) -> torch.Tensor:
    """Dispatch to one of PatternSearchPGD/RandomStartPGD's five
    --attack-strategy variants, batched. "pgd" reuses _attack_pgd from
    pattern_search_pgd.py unchanged (already batch-generic).

    apgd_ce/apgd_dlr/apgd_cw/apgd_t all ascend classification-margin losses
    (softmax CE, or a normalized/raw gap between the label logit and the
    best "other class" logit) -- meaningless for LINEAR_LE/UNSAFE_LINEAR/
    RANGE specs, which have no notion of a "true class" at all (label is
    frequently -1 for these, silently clamped to class 0 by every one of
    those loss functions). Against those specs the requested apgd_* variant
    was ascending an arbitrary, disconnected quantity every step instead of
    the actual c^T y <= d / polytope-membership condition PropertyChecker
    checks afterward, so the search never converged toward a real violation
    -- this is why BI/GCE (default bi_attack_strategy: "apgd_cw") could
    never find counterexamples on non-classification ("safeLinear"-style)
    instances. Fall back to Auto-PGD's own momentum/adaptive-step machinery
    ascending `_violation_score` instead, which mirrors PropertyChecker's
    sign conventions for every OutKind (same loss "pgd" uses via
    _attack_pgd, just with Auto-PGD's step-size schedule instead of a fixed
    one)."""
    from act.pipeline.fuzzing.pattern_search_pgd import _attack_pgd, _violation_score

    is_classification_spec = output_spec is None or output_spec.kind in (
        OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST,
    )

    if attack_strategy == "pgd":
        return _attack_pgd(x0, model, lb, ub, output_spec, label, steps, step_size=None)
    if not is_classification_spec:
        return _auto_pgd_core_batched(
            x0, model, lb, ub, lambda out: _violation_score(output_spec, out, label), steps,
        )
    if attack_strategy == "apgd_ce":
        return _auto_pgd_core_batched(
            x0, model, lb, ub,
            lambda out: F.cross_entropy(out, label.clamp(min=0), reduction="none"), steps,
        )
    if attack_strategy == "apgd_dlr":
        return _auto_pgd_core_batched(x0, model, lb, ub, lambda out: _dlr_loss_untargeted_batched(out, label), steps)
    if attack_strategy == "apgd_cw":
        return _auto_pgd_core_batched(x0, model, lb, ub, lambda out: _cw_loss_batched(out, label), steps)
    if attack_strategy == "apgd_t":
        return _batched_apgd_t(x0, model, lb, ub, output_spec, label, steps, apgd_t_target_classes)
    raise ValueError(
        f"Unknown bi_attack_strategy: {attack_strategy!r}. "
        f"Valid options: 'pgd', 'apgd_ce', 'apgd_dlr', 'apgd_cw', 'apgd_t'."
    )
