"""Matched semantic-step/epoch profiles, including checkpointing on/off.

Outputs are diagnostics, never final_results.json or research scores. Real GPU
profiles require running this in Colab; --synthetic is an offline harness check.
"""
import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from evaluation.metrics import build_filter_index, evaluate_filtered
from models.reverse_jaket import ReverseJAKET
from preprocessing.dataset import load_synthetic_toy_graph
from preprocessing.experiment_data import (EntityTokens, load_experiment_dataset, prepare_text_tokens,
                                            validate_dataset, dataset_manifest)
from preprocessing.graph_builder import build_train_graph
from preprocessing.sampling import build_incoming_adjacency, split_epoch_train
from training.batches import prepare_batch, batch_loss
from training.config import ExperimentConfig, load_config
from training.precision import checked_step, make_scaler, precision_settings
from training.profiling import StageProfiler, memory_stats
from training.train_pipeline import _lm_signature


def synthetic_resources(cfg):
    from transformers import RobertaConfig, RobertaModel
    dataset = load_synthetic_toy_graph(num_entities=128, num_relations=237, num_train=512,
                                      num_valid=32, num_test=32, seed=cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    def tokens(count):
        ids = torch.randint(3, 100, (count, cfg.max_text_length), generator=generator)
        ids[:, 0], ids[:, -1] = 0, 2
        return EntityTokens(ids, torch.ones_like(ids), torch.ones(count, dtype=torch.bool))
    text = tokens(dataset.num_entities)
    text.relations = tokens(dataset.num_relations)
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        lm = RobertaModel(RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=4, intermediate_size=64,
                         max_position_embeddings=cfg.max_text_length + 4), add_pooling_layer=False)
    return dataset, text, lm


def profile_case(cfg, dataset, tokens, model, initial_state, *, batches, epoch, full_epoch):
    model.cfg = cfg
    model.bridge.lm.activation_checkpointing = cfg.lm_activation_checkpointing
    model._token_cache.clear()
    model.load_trainable_state_dict(initial_state)
    model.train()
    device = next(model.parameters()).device
    model.profiler = StageProfiler(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(cfg, device)
    enabled, dtype = precision_settings(cfg, device)
    torch.manual_seed(cfg.seed + 1000000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed + 1000000)
    started = time.perf_counter()
    mp, supervised = split_epoch_train(dataset.train, cfg.mp_ratio, cfg.seed + epoch)
    incoming = build_incoming_adjacency(mp, dataset.num_entities, dataset.num_relations)
    setup_seconds = time.perf_counter() - started
    known_train = set(dataset.train)
    negative_rng, graph_rng = random.Random(cfg.seed + epoch * 2), random.Random(cfg.seed + epoch * 2 + 1)
    total_batches = math.ceil(len(supervised) / cfg.train_batch_size)
    measured_batches = total_batches if full_epoch else min(batches, total_batches)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    times, losses, work, retries = [], [], [], 0
    first_gradients = None
    calls_before = model.bridge.lm.forward_calls
    for index, start in enumerate(range(0, len(supervised), cfg.train_batch_size), 1):
        if index > measured_batches:
            break
        step_started = time.perf_counter()
        batch = prepare_batch(supervised[start:start + cfg.train_batch_size], dataset, known_train,
                              incoming, cfg, negative_rng, graph_rng, device, model.profiler)
        result = checked_step(model, optimizer, scaler,
            lambda: batch_loss(model, batch, tokens, cfg, True), grad_clip_norm=cfg.grad_clip_norm,
            max_retries=cfg.amp_max_retries, check_paths=index == 1, profiler=model.profiler)
        model.profiler.finish_batch()
        times.append(time.perf_counter() - step_started)
        losses.append(result["loss"])
        retries += result["overflow_retries"]
        if index == 1:
            first_gradients = result["gradients"]
        work.append({**model.last_semantic_work, "sampled_nodes": len(batch.nodes),
                     "sampled_edges": len(batch.types)})
        if index == 1 or index % cfg.log_every_batches == 0 or index == measured_batches:
            print(json.dumps({"event": "profile_progress", "batch": index, "batches": measured_batches,
                "step_seconds": times[-1], "loss": losses[-1], **work[-1], **memory_stats(device)}), flush=True)
    training_seconds = time.perf_counter() - started
    stages = model.profiler.summary()
    model.profiler = None
    # Record training peak separately before optional full-graph evaluation.
    training_memory = memory_stats(device)
    lm_calls = model.bridge.lm.forward_calls - calls_before
    validation_seconds = None
    if full_epoch:
        print(json.dumps({"event": "profile_validation_start"}), flush=True)
        started = time.perf_counter()
        model.eval()
        graph = build_train_graph(dataset.train, dataset.num_entities, dataset.num_relations)
        edges = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t().contiguous()
        types = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
        with torch.no_grad():
            representations = model.encode_full(edges, types, tokens, True)
            evaluate_filtered(model, representations, dataset.valid,
                *build_filter_index(dataset.train + dataset.valid + dataset.test),
                batch_size=cfg.eval_triple_batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - started
    steady = statistics.mean(times[1:]) if len(times) > 1 else None
    return {"status": "passed", "config": cfg.to_dict(), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": str(dtype) if enabled else "float32", "epoch": epoch,
        "full_epoch_measured": full_epoch, "epoch_setup_seconds": setup_seconds,
        "measured_batches": measured_batches, "total_epoch_batches": total_batches,
        "batch_seconds": times, "cold_batch_seconds": times[0], "steady_batch_mean_seconds": steady,
        "training_seconds": training_seconds, "validation_seconds": validation_seconds,
        "measured_epoch_seconds": setup_seconds + training_seconds + validation_seconds if full_epoch else None,
        "estimated_training_epoch_seconds": steady * total_batches if steady is not None else None,
        "estimate_scope": "same dataset/config only; excludes setup/validation/checkpoint I/O; tiny LM is not RoBERTa-base",
        "stages": stages, "losses": losses, "work": work, "first_batch_gradients": first_gradients,
        "amp_overflow_retries": retries, "final_scale": scaler.get_scale(),
        "lm_forward_calls_including_recomputation": lm_calls,
        "training_memory": training_memory, "including_validation_memory": memory_stats(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="experiments/configs/proposed_complex.yaml")
    parser.add_argument("--output", required=True, help="New JSON file; written after every case")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--full-epoch", action="store_true", help="Train every batch AND time full validation per case")
    parser.add_argument("--epoch", type=int, help="Epoch split to profile, default first semantic epoch")
    parser.add_argument("--checkpoint", help="Optional matching warm-up/current checkpoint for realistic prompts")
    parser.add_argument("--cases", nargs="+", choices=["legacy", "optimized", "no_checkpoint", "larger_relation_batch"],
                        default=["legacy", "optimized", "no_checkpoint"])
    parser.add_argument("--relation-micro-batch", type=int, default=16)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--synthetic", action="store_true", help="CPU harness check with tiny RANDOM LM; NOT research timing")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.batches < 1 or args.relation_micro_batch < 1:
        parser.error("Batch counts must be positive")
    cfg = load_config(args.config)
    if cfg.variant != "proposed":
        parser.error("Profile the proposed semantic model")
    if args.amp_dtype:
        cfg.use_amp = args.amp_dtype != "float32"
        if cfg.use_amp:
            cfg.amp_dtype = args.amp_dtype
    epoch = args.epoch if args.epoch is not None else cfg.warmup_epochs + 1
    if epoch < 1:
        parser.error("epoch must be positive")
    torch.manual_seed(cfg.seed)
    if args.synthetic:
        cfg = replace(cfg, dataset="synthetic", device="cpu", use_amp=False, lm_name="offline-random-roberta-profile")
        torch.set_num_threads(1)
        dataset, tokens, lm = synthetic_resources(cfg)
    else:
        dataset, lm = load_experiment_dataset(cfg), None
        tokens = prepare_text_tokens(cfg, dataset)
    validate_dataset(dataset)
    tokens.validate(dataset.num_entities, dataset.num_relations)
    model = ReverseJAKET(dataset.num_entities, dataset.num_relations, cfg, lm=lm)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 2 or saved["manifest"] != dataset_manifest(cfg, dataset, tokens):
            raise ValueError("Checkpoint format/data/text mismatch")
        if saved.get("lm_signature") != _lm_signature(model):
            raise ValueError("Checkpoint frozen LM mismatch")
        original_cfg = ExperimentConfig.from_dict(saved["config"])
        for key in ("variant", "kg_dim", "rgat_heads", "layers_per_phase", "num_blocks", "block_target_width"):
            if getattr(original_cfg, key) != getattr(cfg, key):
                raise ValueError(f"Checkpoint architecture mismatch: {key}")
        model.load_trainable_state_dict(saved["model_state"])
    initial = {k: v.detach().cpu().clone() for k, v in model.trainable_state_dict().items()}
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    model.to(device)
    report = {"purpose": "performance diagnostics, not research results", "synthetic": args.synthetic,
              "torch": str(torch.__version__), "checkpoint": args.checkpoint, "cases": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "legacy": dict(optimize_semantic_batches=False, relation_lm_micro_batch=None, lm_activation_checkpointing=True),
        "optimized": dict(optimize_semantic_batches=True, lm_activation_checkpointing=True),
        "no_checkpoint": dict(optimize_semantic_batches=True, lm_activation_checkpointing=False),
        "larger_relation_batch": dict(optimize_semantic_batches=True, lm_activation_checkpointing=True,
                                      relation_lm_micro_batch=args.relation_micro_batch),
    }
    for name in dict.fromkeys(args.cases):
        print(json.dumps({"event": "profile_case_start", "case": name}), flush=True)
        case_cfg = replace(cfg, **settings[name]).validate()
        try:
            report["cases"][name] = profile_case(case_cfg, dataset, tokens, model, initial,
                batches=args.batches, epoch=epoch, full_epoch=args.full_epoch)
        except (RuntimeError, ValueError) as error:
            report["cases"][name] = {"status": "failed", "error": str(error), "config": case_cfg.to_dict()}
            print(json.dumps({"event": "profile_case_failed", "case": name, "error": str(error)}), flush=True)
        finally:
            model.profiler = None
            model.zero_grad(set_to_none=True)
            model._token_cache.clear()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output)
    print(f"Profile saved to {output}", flush=True)
    if any(case["status"] != "passed" for case in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
