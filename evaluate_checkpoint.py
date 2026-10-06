"""Recompute held-out metrics from a saved best checkpoint without training."""
import argparse
import json
from pathlib import Path

import torch

from evaluation.metrics import build_filter_index, evaluate_filtered
from models.reverse_jaket import ReverseJAKET
from preprocessing.experiment_data import (dataset_manifest, load_experiment_dataset,
                                            prepare_text_tokens, validate_dataset)
from preprocessing.graph_builder import build_train_graph
from training.config import ExperimentConfig
from training.train_pipeline import _lm_signature


def evaluate_checkpoint(path, device="auto", *, dataset=None, tokens=None, lm=None,
                        raw_dir=None, text_dir=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 2:
        raise ValueError("Old fusion checkpoints are incompatible; start a new run")
    cfg = ExperimentConfig.from_dict(saved["config"])
    if raw_dir is not None:
        cfg.raw_dir = raw_dir
    if text_dir is not None:
        cfg.text_dir = text_dir
    target = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    dataset = dataset if dataset is not None else load_experiment_dataset(cfg)
    validate_dataset(dataset)
    if cfg.variant != "baseline":
        tokens = tokens if tokens is not None else prepare_text_tokens(cfg, dataset)
        tokens.validate(dataset.num_entities, dataset.num_relations)
    else:
        tokens = None
    if saved["manifest"] != dataset_manifest(cfg, dataset, tokens):
        raise ValueError("Checkpoint dataset/ID/text fingerprint mismatch")
    model = ReverseJAKET(dataset.num_entities, dataset.num_relations, cfg, lm=lm)
    if saved.get("lm_signature") != _lm_signature(model):
        raise ValueError("Checkpoint frozen LM identity mismatch")
    state = saved.get("best_state") or saved["model_state"]
    model.load_trainable_state_dict(state)
    model = model.to(target).eval()
    graph = build_train_graph(dataset.train, dataset.num_entities, dataset.num_relations)
    edges = torch.tensor(graph.edge_index, dtype=torch.long, device=target).t().contiguous()
    types = torch.tensor(graph.edge_type, dtype=torch.long, device=target)
    use_text = saved["best_use_text"]
    representations = model.encode_full(edges, types, tokens=tokens, use_text=use_text)
    filters = build_filter_index(dataset.train + dataset.valid + dataset.test)
    return {
        "dataset": cfg.dataset, "synthetic": cfg.dataset == "synthetic",
        "best_epoch": saved.get("best_epoch", saved["epoch"]), "semantic_enabled": use_text,
        "validation": evaluate_filtered(model, representations, dataset.valid, *filters,
                                          batch_size=cfg.eval_triple_batch_size),
        "test": evaluate_filtered(model, representations, dataset.test, *filters,
                                    batch_size=cfg.eval_triple_batch_size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-dir")
    parser.add_argument("--text-dir")
    parser.add_argument("--output", help="Optional new JSON file; existing files are not overwritten")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.device, raw_dir=args.raw_dir, text_dir=args.text_dir)
    formatted = json.dumps(result, indent=2, allow_nan=False)
    print(formatted)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as file:
            file.write(formatted)


if __name__ == "__main__":
    main()
