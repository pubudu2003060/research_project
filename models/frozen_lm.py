"""Frozen RoBERTa with one prepended continuous KG soft-prompt token."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel


class FrozenLM(nn.Module):
    def __init__(self, model_name="roberta-base", lm=None, activation_checkpointing=True):
        super().__init__()
        self.model_name = model_name
        self.lm = lm if lm is not None else AutoModel.from_pretrained(
            model_name, add_pooling_layer=False)
        self.lm.requires_grad_(False)
        self.lm.eval()
        self.lm.config.use_cache = False
        self.hidden_size = int(self.lm.config.hidden_size)
        self.activation_checkpointing = activation_checkpointing

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()
        return self

    def _forward(self, soft_prompt, input_ids, attention_mask):
        # Keep autograd enabled: frozen parameters do not mean frozen inputs.
        embeddings = self.lm.get_input_embeddings()(input_ids)
        embeddings = torch.cat([soft_prompt.unsqueeze(1), embeddings], dim=1)
        prompt_mask = attention_mask.new_ones((len(attention_mask), 1))
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        out = self.lm(inputs_embeds=embeddings, attention_mask=extended_mask,
                      return_dict=True)
        return out.last_hidden_state[:, 0, :]

    def forward(self, soft_prompt, input_ids, attention_mask):
        if soft_prompt.ndim != 2 or soft_prompt.shape[-1] != self.hidden_size:
            raise ValueError("soft_prompt must have shape [batch, LM hidden size]")
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, length] shapes")
        if len(soft_prompt) != len(input_ids) or input_ids.shape[1] == 0:
            raise ValueError("Batch sizes must match and text sequences must not be empty")
        if self.activation_checkpointing and self.training and soft_prompt.requires_grad:
            # Recompute transformer activations on backward; micro-batching alone
            # would retain every micro-batch's activations until loss.backward().
            return checkpoint(self._forward, soft_prompt, input_ids, attention_mask,
                              use_reentrant=False)
        return self._forward(soft_prompt, input_ids, attention_mask)
