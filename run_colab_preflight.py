"""One real semantic optimizer step, with explicit AMP diagnostics and stage timings."""
import argparse
import json
from pathlib import Path
import random
import time

import torch

from models.reverse_jaket import ReverseJAKET
from preprocessing.experiment_data import load_experiment_dataset, prepare_text_tokens, validate_dataset
from preprocessing.sampling import build_incoming_adjacency, split_epoch_train
from training.batches import prepare_batch, batch_loss
from training.config import load_config
from training.precision import checked_step, make_scaler, precision_settings, GRADIENT_PATHS
from training.profiling import StageProfiler, memory_stats


def run_preflight(cfg):
    if cfg.variant != "proposed":
        raise ValueError("Use a proposed-model configuration to check the complete semantic path")
    torch.manual_seed(cfg.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    dataset = load_experiment_dataset(cfg)
    validate_dataset(dataset)
    tokens = prepare_text_tokens(cfg, dataset)
    if not tokens.has_text.any():
        raise ValueError("No usable entity descriptions")
    model = ReverseJAKET(dataset.num_entities, dataset.num_relations, cfg).to(device).train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    enabled, dtype = precision_settings(cfg, device)
    scaler = make_scaler(cfg, device)
    mp, supervised = split_epoch_train(dataset.train, cfg.mp_ratio, cfg.seed + 1)
    incoming = build_incoming_adjacency(mp, dataset.num_entities, dataset.num_relations)
    model.profiler = StageProfiler(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    batch = prepare_batch(supervised[:cfg.train_batch_size], dataset, set(dataset.train), incoming,
                          cfg, random.Random(cfg.seed), random.Random(cfg.seed), device, model.profiler)
    result = checked_step(model, optimizer, scaler,
        lambda: batch_loss(model, batch, tokens, cfg, use_text=True),
        grad_clip_norm=cfg.grad_clip_norm, max_retries=cfg.amp_max_retries,
        check_paths=True, profiler=model.profiler)
    timings = model.profiler.summary()
    return {
        "status": "passed", "purpose": "one training step; not benchmark results",
        "device": str(device), "kg_dim": cfg.kg_dim,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": str(dtype) if enabled else "float32", "loss_scaling": scaler.is_enabled(),
        "entities_with_text": int(tokens.has_text.sum()),
        "relations_with_text": int(tokens.relations.has_text.sum()),
        "sampled_nodes": len(batch.nodes), "sampled_edges": len(batch.types),
        **model.last_semantic_work, **result,
        "gradient_paths": list(GRADIENT_PATHS),
        "step_seconds": time.perf_counter() - started, "stages": timings,
        "lm_forward_calls_including_recomputation": model.bridge.lm.forward_calls,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        **memory_stats(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="experiments/configs/proposed_complex.yaml")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--amp-max-retries", type=int)
    parser.add_argument("--legacy", action="store_true", help="Encode all relations and transfer tokens per micro-batch")
    parser.add_argument("--output", help="New diagnostics JSON path (will not overwrite)")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    cfg = load_config(args.config)
    if args.amp_dtype:
        cfg.use_amp = args.amp_dtype != "float32"
        if cfg.use_amp:
            cfg.amp_dtype = args.amp_dtype
    if args.amp_max_retries is not None:
        cfg.amp_max_retries = args.amp_max_retries
    if args.legacy:
        cfg.optimize_semantic_batches = False
        cfg.relation_lm_micro_batch = None
    cfg.validate()
    try:
        result = run_preflight(cfg)
    except (RuntimeError, ValueError) as error:
        result = {"status": "failed", "error": str(error), "config": cfg.to_dict()}
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
