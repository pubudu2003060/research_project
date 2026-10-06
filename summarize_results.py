"""Export measured comparison tables and plots; never synthesize missing results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def collect_results(root, include_synthetic=False):
    rows = []
    for path in sorted(Path(root).rglob("final_results.json")):
        status_path = path.parent / "run_status.json"
        if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8")).get("status") != "complete":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        # Old fusion runs are a different architecture and must not be pooled.
        if result.get("format_version") != 2:
            continue
        if result["synthetic"] and not include_synthetic:
            continue
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "data_manifest.json").read_text(encoding="utf-8"))
        # Prevent pooling different splits/ID maps under the same dataset label.
        identity = {key: manifest[key] for key in ("split_sha256", "entity_ids_sha256", "relation_ids_sha256")}
        protocol = {key: config[key] for key in (
            "lm_name", "max_text_length", "rgat_heads", "layers_per_phase", "num_blocks",
            "block_target_width", "attention_dropout", "feature_dropout",
            "mp_ratio", "neighbors_per_hop", "train_batch_size",
            "num_negatives", "balanced_head_tail_negatives",
            "max_epochs", "eval_every", "early_stopping_patience", "lr", "weight_decay",
            "grad_clip_norm", "use_amp", "lm_micro_batch", "lm_activation_checkpointing",
        )}
        environment = json.loads((path.parent / "environment.json").read_text(encoding="utf-8"))
        # Do not pool different text resources, frozen weights or warm-start
        # sources into the same repeated-seed group.
        protocol["tokens_sha256"] = manifest.get("tokens_sha256")
        protocol["relation_tokens_sha256"] = manifest.get("relation_tokens_sha256")
        protocol["architecture"] = "direct-soft-prompt-entity-relation-v2"
        protocol["lm_signature"] = environment.get("lm")
        protocol["warm_start"] = environment.get("warm_start")
        row = {key: result[key] for key in ("dataset", "synthetic", "variant", "scorer", "kg_dim", "seed",
                "warmup_epochs", "best_epoch", "epochs_completed", "elapsed_seconds", "peak_cuda_memory_bytes")}
        row["split_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        row["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        row["run"] = str(path.parent)
        row.update({f"test_{k}": v for k, v in result["test"].items()})
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, output_dir, include_synthetic=False, plots=True):
    rows = collect_results(root, include_synthetic)
    if not rows:
        raise ValueError("No completed eligible runs. Run experiments first; synthetic tests are excluded by default.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "baseline_vs_reverse_jaket.csv", rows)
    groups = {}
    keys = ("dataset", "synthetic", "split_id", "protocol_id", "variant", "scorer", "kg_dim", "warmup_epochs")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, runs in groups.items():
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seeds in comparison group {key}; avoid counting reruns twice")
        summary = dict(zip(keys, key))
        summary.update(n=len(runs), seeds=",".join(map(str, sorted(seeds))))
        for metric in ("test_MRR", "test_Hits@1", "test_Hits@3", "test_Hits@10"):
            values = [r[metric] for r in runs]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)
    write_csv(output / "summary.csv", summaries)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(9, len(summaries)), 5))
        labels = [f"{r['dataset']} {r['variant']}\n{r['scorer']} d={r['kg_dim']}\nwarmup={r['warmup_epochs']} n={r['n']}\n{r['protocol_id'][:6]}" for r in summaries]
        ax.bar(range(len(summaries)), [r["test_MRR_mean"] for r in summaries],
               yerr=[r["test_MRR_std"] or 0 for r in summaries], capsize=3)
        ax.set_xticks(range(len(summaries)), labels, rotation=45, ha="right")
        ax.set_ylabel("Filtered test MRR (mean; sample SD where n > 1)")
        ax.set_title("Synthetic correctness runs" if all(r["synthetic"] for r in rows) else "Measured experiment results")
        fig.tight_layout()
        fig.savefig(output / "test_mrr.png", dpi=180)
        plt.close(fig)
        for row in rows:
            with (Path(row["run"]) / "history.csv").open(encoding="utf-8") as file:
                history = list(csv.DictReader(file))
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot([int(r["epoch"]) for r in history], [float(r["loss"]) for r in history])
            axes[0].set(xlabel="Epoch", ylabel="Training loss")
            validation = [r for r in history if r.get("val_MRR")]
            axes[1].plot([int(r["epoch"]) for r in validation], [float(r["val_MRR"]) for r in validation])
            axes[1].set(xlabel="Epoch", ylabel="Filtered validation MRR")
            fig.suptitle(Path(row["run"]).name)
            fig.tight_layout()
            plot_id = hashlib.sha256(row["run"].encode()).hexdigest()[:8]
            fig.savefig(output / f"{Path(row['run']).name}_{plot_id}_training.png", dpi=180)
            plt.close(fig)
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="experiments/comparisons")
    parser.add_argument("--output-dir", default="experiments/reports")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rows, summary = summarize(args.root, args.output_dir, args.include_synthetic, not args.no_plots)
    print(f"Exported {len(rows)} measured runs and {len(summary)} comparison groups to {args.output_dir}")


if __name__ == "__main__":
    main()
