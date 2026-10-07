"""Generate matched ComplEx baseline, warm-up and dimension experiments."""
from dataclasses import replace
import argparse
import json
from pathlib import Path

import yaml

from training.config import load_config
from training.train_pipeline import run_experiment


def comparison_plan(base, root, seeds, dimensions):
    variants = [
        ("baseline_complex", "baseline", "complex", base.kg_dim, base.warmup_epochs),
        ("text_enhanced_complex", "text_enhanced", "complex", base.kg_dim, base.warmup_epochs),
        ("proposed_complex", "proposed", "complex", base.kg_dim, base.warmup_epochs),
        ("proposed_no_warmup", "proposed", "complex", base.kg_dim, 0),
    ]
    for dim in dict.fromkeys(dimensions):
        if dim != base.kg_dim:
            # Pair each dimension with its own structural control.
            variants.extend([(f"baseline_dim{dim}", "baseline", "complex", dim, base.warmup_epochs),
                             (f"proposed_dim{dim}", "proposed", "complex", dim, base.warmup_epochs)])
    return [(f"{name}_seed{seed}", replace(base, variant=variant, scorer=scorer, kg_dim=dim,
                    num_blocks=None, warmup_epochs=warmup, seed=seed,
                    output_dir=str(Path(root) / f"{name}_seed{seed}")).validate())
            for name, variant, scorer, dim, warmup in variants for seed in dict.fromkeys(seeds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/proposed_complex.yaml")
    parser.add_argument("--output-root", default="experiments/comparisons")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--dimensions", nargs="*", type=int, default=[128, 200, 256])
    parser.add_argument("--execute", action="store_true", help="Train the generated plan (default: write configs only)")
    args = parser.parse_args()
    plan = comparison_plan(load_config(args.config), args.output_root, args.seeds, args.dimensions)
    config_dir = Path(args.output_root) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in plan:
        (config_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        print(f"{name}: {cfg.variant}, {cfg.scorer}, dim={cfg.kg_dim}, seed={cfg.seed}", flush=True)
        if args.execute:
            out = Path(cfg.output_dir)
            completed = out / "final_results.json"
            saved_config = out / "config.json"
            status_path = out / "run_status.json"
            complete = status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete"
            if completed.exists() and saved_config.exists() and complete:
                if load_config(saved_config).to_dict() == cfg.to_dict():
                    print("  already completed; keeping existing results", flush=True)
                    continue
            last = out / "last_checkpoint.pt"
            run_experiment(cfg, resume=last if last.exists() else None)
    print(f"{'Executed' if args.execute else 'Prepared'} {len(plan)} experiments. Configurations: {config_dir}")


if __name__ == "__main__":
    main()
