"""Settings for the Milestone 3 methodology with FB15k-237 and ComplEx."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    dataset: str = "fb15k237"
    raw_dir: str = "data/raw/fb15k237"
    text_dir: str = "data/text/fb15k237"
    download_if_missing: bool = True
    output_dir: str = "experiments/runs/proposed_complex"
    variant: str = "proposed"
    scorer: str = "complex"
    kg_dim: int = 32
    lm_name: str = "roberta-base"
    max_text_length: int = 64
    lm_micro_batch: int = 4
    relation_lm_micro_batch: int | None = None
    lm_activation_checkpointing: bool = True
    optimize_semantic_batches: bool = True
    rgat_heads: int = 2
    layers_per_phase: int = 1
    num_blocks: int | None = None
    block_target_width: int = 4
    attention_dropout: float = 0.10
    feature_dropout: float = 0.10
    warmup_epochs: int = 10
    mp_ratio: float = 0.90
    neighbors_per_hop: int = 2
    train_batch_size: int = 8
    num_negatives: int = 4
    balanced_head_tail_negatives: bool = True
    max_epochs: int = 300
    eval_every: int = 5
    early_stopping_patience: int = 12
    lr: float = 0.001
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "float16"
    amp_max_retries: int = 16
    log_every_batches: int = 50
    profile_training: bool = False
    eval_node_batch_size: int = 128
    eval_triple_batch_size: int = 64
    checkpoint_every: int = 1
    seed: int = 42
    device: str = "auto"

    def validate(self):
        if self.dataset not in {"fb15k237", "synthetic"}:
            raise ValueError("Supported datasets: fb15k237, synthetic (tests only)")
        if self.variant not in {"baseline", "text_enhanced", "proposed"}:
            raise ValueError("variant must be baseline, text_enhanced or proposed")
        if self.scorer != "complex":
            raise ValueError("Only the ComplEx scorer is supported (scorer: complex)")
        for name in ("kg_dim", "lm_micro_batch", "rgat_heads", "layers_per_phase",
                     "block_target_width", "neighbors_per_hop", "train_batch_size",
                     "num_negatives", "max_epochs", "eval_every", "checkpoint_every",
                     "eval_node_batch_size", "eval_triple_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.kg_dim % 2:
            raise ValueError("ComplEx needs an even kg_dim (packed real/imaginary halves)")
        if self.relation_lm_micro_batch is not None and self.relation_lm_micro_batch < 1:
            raise ValueError("relation_lm_micro_batch must be positive or null")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16")
        if self.amp_max_retries < 0 or self.log_every_batches < 1:
            raise ValueError("amp_max_retries must be nonnegative; log_every_batches must be positive")
        if self.max_text_length < 2 or self.max_text_length > 511:
            raise ValueError("max_text_length must be between 2 and 511 (one extra soft-prompt token)")
        if self.num_blocks is not None and (self.num_blocks < 1 or self.kg_dim % self.num_blocks):
            raise ValueError("num_blocks must be a positive divisor of kg_dim")
        if not 0 < self.mp_ratio < 1:
            raise ValueError("mp_ratio must be strictly between zero and one")
        for name in ("attention_dropout", "feature_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("warmup_epochs", "early_stopping_patience", "weight_decay"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.lr <= 0 or self.grad_clip_norm <= 0:
            raise ValueError("lr and grad_clip_norm must be positive")
        if self.balanced_head_tail_negatives and self.num_negatives % 2:
            raise ValueError("Balanced corruption requires an even num_negatives")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values).validate()


def load_config(path):
    return ExperimentConfig.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def semantic_enabled(epoch: int, cfg: ExperimentConfig) -> bool:
    """Hard switch after optional KG-only warm-up; no fusion or ramp."""
    return cfg.variant != "baseline" and epoch > cfg.warmup_epochs
