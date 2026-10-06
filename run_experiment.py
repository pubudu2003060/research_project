"""Run one configured experiment; see README for Colab and comparison commands."""
import argparse

from training.config import load_config
from training.train_pipeline import run_experiment


def main(default_config="experiments/configs/proposed_complex.yaml"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=default_config)
    parser.add_argument("--resume", help="Resume last_checkpoint.pt including optimizer/RNG")
    parser.add_argument("--warm-start", help="Initialize structural weights from a compatible checkpoint")
    parser.add_argument("--output-dir", help="Override run directory (e.g. persistent Colab Drive)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_experiment(cfg, resume=args.resume, warm_start=args.warm_start)


if __name__ == "__main__":
    main()
