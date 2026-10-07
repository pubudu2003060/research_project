"""Mixed precision and checked optimizer steps shared by preflight and training."""
import json
import sys

import torch

from training.profiling import stage


GRADIENT_PATHS = ("encoder1", "encoder2", "scorer", "bridge.kg_to_lm", "bridge.lm_to_kg")


def precision_settings(cfg, device):
    device = torch.device(device)
    enabled = cfg.use_amp and device.type == "cuda"
    dtype = getattr(torch, cfg.amp_dtype)
    if enabled and dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support BF16; use amp_dtype: float16")
    return enabled, dtype


def make_scaler(cfg, device):
    enabled, dtype = precision_settings(cfg, device)
    return torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)


def gradient_report(model, prefixes=GRADIENT_PATHS):
    """Count missing, all-zero, finite nonzero, NaN and Inf gradient tensors.

    NaN/Inf counts may overlap. Summaries inspect unscaled gradients, never a
    potentially overflowing sum of absolute values in reduced precision.
    """
    report = {}
    for prefix in prefixes:
        row = dict(parameters=0, none=0, zero=0, finite_nonzero=0, nan=0, inf=0,
                   max_finite_abs=0.0)
        for name, param in model.named_parameters():
            if not param.requires_grad or not name.startswith(prefix + "."):
                continue
            row["parameters"] += 1
            grad = param.grad
            if grad is None:
                row["none"] += 1
                continue
            grad = grad.detach()
            nan, inf = bool(torch.isnan(grad).any()), bool(torch.isinf(grad).any())
            row["nan"] += int(nan)
            row["inf"] += int(inf)
            if not nan and not inf:
                nonzero = bool(torch.count_nonzero(grad))
                row["finite_nonzero" if nonzero else "zero"] += 1
            finite_abs = torch.where(torch.isfinite(grad), grad.abs(), 0).max().item()
            row["max_finite_abs"] = max(row["max_finite_abs"], finite_abs)
        report[prefix] = row
    return report


def assert_gradient_paths(model, report):
    bad = {name: row for name, row in report.items()
           if row["finite_nonzero"] == 0 or row["nan"] or row["inf"]}
    if bad:
        raise RuntimeError("Invalid gradient paths (unscaled): " + json.dumps(bad))
    if model.bridge is not None and any(p.requires_grad or p.grad is not None
                                       for p in model.bridge.lm.parameters()):
        raise RuntimeError("Language model must remain frozen with no parameter gradients")


def checked_step(model, optimizer, scaler, forward_loss, *, grad_clip_norm,
                 max_retries=16, check_paths=False, profiler=None):
    """Retry ONLY non-finite scaled gradients; never accept a disconnected path.

    GradScaler's public step/update handles overflow (skipped step + backoff).
    Retry the same samples and dropout RNG so calibration does not silently
    discard training examples. Finite-loss/finite-gradient errors are distinct.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    device = parameters[0].device
    cpu_rng = torch.get_rng_state() if scaler.is_enabled() else None
    cuda_rng = torch.cuda.get_rng_state(device) if scaler.is_enabled() and device.type == "cuda" else None
    attempts = []
    for attempt in range(max_retries + 1):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite forward loss; loss-scaler backoff cannot repair forward overflow")
        with stage(profiler, "backward"):
            scaler.scale(loss).backward()
        with stage(profiler, "gradient_check_and_clip"):
            scaler.unscale_(optimizer)
            grads = [p.grad for p in parameters if p.grad is not None]
            finite = bool(grads) and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
            report = gradient_report(model) if check_paths or not finite else None
        if not finite:
            entry = {"attempt": attempt + 1, "scale_before": scaler.get_scale(), "gradients": report}
            if not grads or not scaler.is_enabled():
                raise RuntimeError("Non-finite or missing gradients without FP16 scaling: " + json.dumps(entry))
            # Do NOT clip invalid gradients; unscale_ has already marked the
            # optimizer for skipping, so step cannot corrupt its state.
            with stage(profiler, "optimizer_step"):
                scaler.step(optimizer)
                scaler.update()
            entry["scale_after"] = scaler.get_scale()
            attempts.append(entry)
            print(json.dumps({"event": "amp_overflow_retry", **entry}), file=sys.stderr, flush=True)
            if attempt == max_retries:
                raise RuntimeError("FP16 gradients remain non-finite after bounded scale backoff: "
                                   + json.dumps(entry))
            continue
        if check_paths:
            assert_gradient_paths(model, report)
        with stage(profiler, "gradient_check_and_clip"):
            norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm, error_if_nonfinite=True)
        with stage(profiler, "optimizer_step"):
            scaler.step(optimizer)
            scaler.update()
        return {"loss": float(loss.detach()), "gradient_norm": float(norm),
                "overflow_retries": len(attempts), "overflow_diagnostics": attempts,
                "scale": scaler.get_scale(), "gradients": report}
    raise AssertionError("unreachable")
