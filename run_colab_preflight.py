"""One real-data semantic training step to check downloads, gradients and memory."""
import argparse
import json
import random
import time

import torch

from models.reverse_jaket import ReverseJAKET
from preprocessing.experiment_data import load_experiment_dataset, prepare_text_tokens, validate_dataset
from preprocessing.sampling import build_incoming_adjacency, sample_local_graph, split_epoch_train
from training.config import load_config
from training.losses import bce_loss
from training.negative_sampling import sample_negatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="experiments/configs/proposed_complex.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
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
    amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    mp, supervised = split_epoch_train(dataset.train, cfg.mp_ratio, cfg.seed + 1)
    pos = supervised[:cfg.train_batch_size]
    neg = sample_negatives(pos, dataset.num_entities, set(dataset.train), cfg.num_negatives,
                           random.Random(cfg.seed), balanced_head_tail=cfg.balanced_head_tail_negatives)
    incoming = build_incoming_adjacency(mp, dataset.num_entities, dataset.num_relations)
    seeds = torch.cat([pos[:, [0, 2]].flatten(), neg[..., [0, 2]].flatten()]).unique().tolist()
    graph = sample_local_graph(incoming, seeds, 2 * cfg.layers_per_phase,
                               cfg.neighbors_per_hop, random.Random(cfg.seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        representations = model.encode_local(graph.global_nodes.to(device), graph.edge_index.to(device),
                                              graph.edge_type.to(device), tokens, use_text=True)
        positive = model.score_triples(representations, graph.remap(pos).to(device))
        negative = model.score_triples(representations, graph.remap(neg).reshape(-1, 3).to(device))
        loss = bce_loss(positive, negative)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss in preflight")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    checked = []
    for prefix in ("encoder1", "encoder2", "scorer", "bridge.kg_to_lm", "bridge.lm_to_kg"):
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
            raise RuntimeError(f"No finite nonzero gradient through {prefix}")
        checked.append(prefix)
    if any(p.requires_grad or p.grad is not None for p in model.bridge.lm.parameters()):
        raise RuntimeError("Language model must remain frozen")
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(json.dumps({
        "status": "passed", "purpose": "one training step; not benchmark results",
        "device": str(device), "kg_dim": cfg.kg_dim,
        "entities_with_text": int(tokens.has_text.sum()),
        "relations_with_text": int(tokens.relations.has_text.sum()),
        "sampled_nodes": len(graph.global_nodes), "sampled_edges": len(graph.edge_type),
        "loss": float(loss.detach()), "gradient_paths": checked,
        "step_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }, indent=2))


if __name__ == "__main__":
    main()
