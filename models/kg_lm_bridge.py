"""KG -> frozen LM -> KG projections, with no semantic fusion module."""
from __future__ import annotations

import torch
from torch import nn

from models.frozen_lm import FrozenLM


class KGLMBridge(nn.Module):
    def __init__(self, kg_dim=32, model_name="roberta-base", lm=None,
                 activation_checkpointing=True):
        super().__init__()
        self.kg_dim = kg_dim
        self.lm = FrozenLM(model_name, lm=lm, activation_checkpointing=activation_checkpointing)
        self.lm_dim = self.lm.hidden_size
        self.kg_to_lm = nn.Linear(kg_dim, self.lm_dim)
        self.lm_to_kg = nn.Linear(self.lm_dim, kg_dim)

    def forward(self, kg_vectors, input_ids, attention_mask, micro_batch=4,
                condition_on_kg=True):
        if kg_vectors.ndim != 2 or kg_vectors.shape[-1] != self.kg_dim:
            raise ValueError("KG vector shape does not match kg_dim")
        if micro_batch < 1:
            raise ValueError("micro_batch must be positive")
        if len(input_ids) != len(kg_vectors) or input_ids.shape != attention_mask.shape:
            raise ValueError("Text and KG batch sizes/masks must match")
        outputs = []
        for start in range(0, len(kg_vectors), micro_batch):
            h = kg_vectors[start:start + micro_batch]
            prompt = self.kg_to_lm(h) if condition_on_kg else h.new_zeros((len(h), self.lm_dim))
            semantic = self.lm(prompt, input_ids[start:start + micro_batch].to(h.device),
                               attention_mask[start:start + micro_batch].to(h.device))
            outputs.append(self.lm_to_kg(semantic))
        return torch.cat(outputs) if outputs else kg_vectors
