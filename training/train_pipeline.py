"""Sampled RGAT training, resumable artifacts and held-out evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import torch

from evaluation.metrics import build_filter_index, evaluate_filtered
from models.reverse_jaket import ReverseJAKET
from preprocessing.experiment_data import (
    dataset_manifest, load_experiment_dataset, prepare_text_tokens, validate_dataset,
)
from preprocessing.graph_builder import build_train_graph
from preprocessing.sampling import build_incoming_adjacency, sample_local_graph, split_epoch_train
from training.config import ExperimentConfig, semantic_enabled
from training.losses import bce_loss
from training.negative_sampling import sample_negatives


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.trainable_state_dict().items()}


def _write_history(path, history):
    keys = list(dict.fromkeys(key for row in history for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _lm_signature(model):
    if model.bridge is None:
        return None
    lm = model.bridge.lm.lm
    # Config/commit identity plus frozen-weight digest protect resume even if a
    # model alias moves, or a local pretrained checkpoint changes in place.
    digest = hashlib.sha256()
    for name, value in lm.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return {"name": model.cfg.lm_name, "commit": getattr(lm.config, "_commit_hash", None),
            "weights_sha256": digest.hexdigest()}


def _check_resume(checkpoint, cfg, manifest, lm_signature):
    if checkpoint.get("format_version") != 2:
        raise ValueError("Old fusion checkpoints are incompatible; start a new run")
    old = checkpoint["config"]
    current = cfg.to_dict()
    allowed = {"max_epochs", "device", "output_dir", "download_if_missing", "raw_dir", "text_dir"}
    changes = [k for k in current if k not in allowed and current[k] != old.get(k)]
    if changes:
        raise ValueError(f"Resume changes experiment settings: {changes}; use a new run")
    if checkpoint["manifest"] != manifest:
        raise ValueError("Dataset IDs, splits or tokenized text changed since checkpoint")
    if checkpoint.get("lm_signature") != lm_signature:
        raise ValueError("Frozen LM weights changed since checkpoint")
    if cfg.max_epochs < checkpoint["epoch"]:
        raise ValueError("max_epochs precedes the checkpoint epoch")


def _warm_start(model, path, manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Warm-start requires a checkpoint from the current architecture")
    old_manifest = checkpoint["manifest"]
    for key in ("scorer", "kg_dim", "rgat_heads", "layers_per_phase", "num_blocks", "block_target_width"):
        if checkpoint["config"][key] != model.cfg.to_dict()[key]:
            raise ValueError(f"Warm-start architecture mismatch: {key}")
    for key in ("dataset", "split_sha256", "entity_ids_sha256", "relation_ids_sha256"):
        if old_manifest[key] != manifest[key]:
            raise ValueError(f"Warm-start dataset mismatch: {key}")
    structural = {k: v for k, v in checkpoint["model_state"].items() if not k.startswith("bridge.")}
    current = model.state_dict()
    expected = {k for k in current if not k.startswith("bridge.")}
    if set(structural) != expected or any(structural[k].shape != current[k].shape for k in expected):
        raise ValueError("Warm-start requires matching RGAT dimensions, heads, blocks and scorer")
    model.load_state_dict(structural, strict=False)


def run_experiment(cfg: ExperimentConfig, resume=None, warm_start=None, *, dataset=None,
                   tokens=None, lm=None):
    """Injected dataset/LM are for offline tests; production loads real resources."""
    cfg.validate()
    if resume and warm_start:
        raise ValueError("Choose resume or warm-start, not both")
    if cfg.variant != "baseline" and cfg.max_epochs <= cfg.warmup_epochs:
        raise ValueError("Semantic experiments must include at least one epoch after warm-up")
    out = Path(cfg.output_dir)
    if not resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory is nonempty: {out}. Use --resume or a new output_dir")
    if resume and out.exists() and any(out.iterdir()) and Path(resume).resolve().parent != out.resolve():
        raise FileExistsError("Resume output must be the checkpoint's directory or a new empty directory")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    dataset = dataset if dataset is not None else load_experiment_dataset(cfg)
    validate_dataset(dataset)
    if cfg.variant != "baseline":
        tokens = tokens if tokens is not None else prepare_text_tokens(cfg, dataset)
        tokens.validate(dataset.num_entities, dataset.num_relations)
        if not tokens.relations.has_text.all():
            raise ValueError("Every relation requires a description")
        if not tokens.has_text.any():
            raise ValueError("Semantic experiment has no usable entity descriptions")
    else:
        tokens = None
    manifest = dataset_manifest(cfg, dataset, tokens)
    model = ReverseJAKET(dataset.num_entities, dataset.num_relations, cfg, lm=lm)
    lm_signature = _lm_signature(model)
    model = model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    graph = build_train_graph(dataset.train, dataset.num_entities, dataset.num_relations)
    full_edges = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t().contiguous()
    full_types = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    filters = build_filter_index(dataset.train + dataset.valid + dataset.test)
    known_train = set(dataset.train)
    history, best_state = [], None
    best_mrr, best_epoch, best_use_text, bad_evals, start_epoch = -1.0, 0, False, 0, 1
    previous_seconds = 0.0
    warm_start_source = str(Path(warm_start).resolve()) if warm_start else None
    if warm_start:
        _warm_start(model, warm_start, manifest)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("evaluation_only"):
            raise ValueError("This checkpoint is evaluation-only; resume from last_checkpoint.pt")
        _check_resume(saved, cfg, manifest, lm_signature)
        model.load_trainable_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history, best_state = saved["history"], saved["best_state"]
        best_mrr, best_epoch, best_use_text = saved["best_val_mrr"], saved["best_epoch"], saved["best_use_text"]
        bad_evals, start_epoch = saved["bad_evals"], saved["epoch"] + 1
        previous_seconds = saved.get("elapsed_seconds", 0.0)
        warm_start_source = saved.get("warm_start_source")
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    _write_json(out / "run_status.json", {"status": "running"})
    _write_json(out / "config.json", cfg.to_dict())
    _write_json(out / "data_manifest.json", manifest)
    import torch_geometric
    import transformers
    _write_json(out / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torch_geometric": torch_geometric.__version__, "transformers": transformers.__version__,
        "device": str(device), "amp_enabled": amp, "lm": lm_signature,
        "resolved_num_blocks": model.encoder1.num_blocks,
        "warm_start": warm_start_source,
        "resume": str(resume) if resume else None,
    })
    # Bridge initialization consumes random numbers. Reset the training RNG so
    # matched baseline/proposed warm-up runs receive identical dropout draws.
    if not resume:
        torch.manual_seed(cfg.seed + 1000000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed + 1000000)
    started = time.perf_counter()
    print(f"{cfg.variant}/{cfg.scorer}: {cfg.dataset}, dim={cfg.kg_dim}, device={device}", flush=True)
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        epoch_started = time.perf_counter()
        use_text = semantic_enabled(epoch, cfg)
        mp, supervised = split_epoch_train(dataset.train, cfg.mp_ratio, cfg.seed + epoch)
        incoming = build_incoming_adjacency(mp, dataset.num_entities, dataset.num_relations)
        negative_rng = random.Random(cfg.seed + epoch * 2)
        graph_rng = random.Random(cfg.seed + epoch * 2 + 1)
        model.train()
        loss_sum, examples = 0.0, 0
        for start in range(0, len(supervised), cfg.train_batch_size):
            pos = supervised[start:start + cfg.train_batch_size]
            neg = sample_negatives(pos, dataset.num_entities, known_train, cfg.num_negatives,
                                   negative_rng, balanced_head_tail=cfg.balanced_head_tail_negatives)
            seeds = torch.cat([pos[:, [0, 2]].flatten(), neg[..., [0, 2]].flatten()]).unique().tolist()
            local = sample_local_graph(incoming, seeds, 2 * cfg.layers_per_phase,
                                       cfg.neighbors_per_hop, graph_rng)
            local_pos = local.remap(pos).to(device)
            local_neg = local.remap(neg).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                entities = model.encode_local(local.global_nodes.to(device), local.edge_index.to(device),
                                               local.edge_type.to(device), tokens=tokens, use_text=use_text)
                positive_scores = model.score_triples(entities, local_pos)
                negative_scores = model.score_triples(entities, local_neg.reshape(-1, 3)).reshape(len(pos), -1)
                loss = bce_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm,
                                           error_if_nonfinite=not amp)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(pos)
            examples += len(pos)

        row = {"epoch": epoch, "loss": loss_sum / examples, "semantic_enabled": use_text,
               "message_triples": len(mp), "supervision_triples": len(supervised)}
        improved = False
        # Select semantic checkpoints only after the LM path is enabled.
        # Otherwise the proposed-model row could silently be a warm-up baseline.
        eligible = cfg.variant == "baseline" or use_text
        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            model.eval()
            representations = model.encode_full(full_edges, full_types, tokens=tokens, use_text=use_text)
            metrics = evaluate_filtered(model, representations, dataset.valid, *filters,
                                         batch_size=cfg.eval_triple_batch_size)
            del representations
            row.update({f"val_{key}": value for key, value in metrics.items()})
            if eligible:
                if metrics["MRR"] > best_mrr:
                    best_mrr, best_epoch, best_use_text = metrics["MRR"], epoch, use_text
                    best_state, bad_evals, improved = _cpu_state(model), 0, True
                else:
                    bad_evals += 1
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        history.append(row)
        _write_history(out / "history.csv", history)
        print(json.dumps(row), flush=True)
        stopped = bool(eligible and cfg.early_stopping_patience and bad_evals >= cfg.early_stopping_patience)
        if epoch % cfg.checkpoint_every == 0 or improved or stopped or epoch == cfg.max_epochs or epoch == cfg.warmup_epochs:
            state = {
                "format_version": 2, "epoch": epoch, "model_state": _cpu_state(model),
                "warm_start_source": warm_start_source,
                "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                "config": cfg.to_dict(), "manifest": manifest, "lm_signature": lm_signature,
                "history": history, "best_state": best_state, "best_val_mrr": best_mrr,
                "best_epoch": best_epoch, "best_use_text": best_use_text, "bad_evals": bad_evals,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "elapsed_seconds": previous_seconds + time.perf_counter() - started,
            }
            _save_checkpoint(out / "last_checkpoint.pt", state)
            if improved:
                _save_checkpoint(out / "best_checkpoint.pt", state)
            if epoch == cfg.warmup_epochs:
                _save_checkpoint(out / "warmup_checkpoint.pt", state)
        if stopped:
            break

    if best_state is None:
        raise RuntimeError("No eligible validation checkpoint; increase max_epochs")
    model.load_trainable_state_dict(best_state)
    model.eval()
    representations = model.encode_full(full_edges, full_types, tokens=tokens, use_text=best_use_text)
    validation = evaluate_filtered(model, representations, dataset.valid, *filters,
                                    batch_size=cfg.eval_triple_batch_size)
    test = evaluate_filtered(model, representations, dataset.test, *filters,
                              batch_size=cfg.eval_triple_batch_size)
    results = {
        "format_version": 2,
        "dataset": cfg.dataset, "synthetic": cfg.dataset == "synthetic", "variant": cfg.variant,
        "scorer": cfg.scorer, "kg_dim": cfg.kg_dim, "seed": cfg.seed,
        "warmup_epochs": cfg.warmup_epochs, "best_epoch": best_epoch,
        "best_semantic_enabled": best_use_text, "epochs_completed": history[-1]["epoch"],
        "validation": validation, "test": test,
        "elapsed_seconds": previous_seconds + time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint": str(out / "best_checkpoint.pt"),
    }
    # Always align the final best artifact with the metrics, including when
    # resuming an older checkpoint in a directory containing later artifacts.
    # last_checkpoint remains the complete resumable optimizer/RNG state.
    _save_checkpoint(out / "best_checkpoint.pt", {
        "format_version": 2, "model_state": best_state, "config": cfg.to_dict(),
        "warm_start_source": warm_start_source,
        "manifest": manifest, "lm_signature": lm_signature, "epoch": best_epoch,
        "best_use_text": best_use_text, "evaluation_only": True,
    })
    _write_history(out / "history.csv", history)
    _write_json(out / "final_results.json", results)
    _write_json(out / "run_status.json", {"status": "complete", "epoch": history[-1]["epoch"]})
    return {"results": results, "history": history, "model": model, "dataset": dataset}
