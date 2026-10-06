"""Two RGAT stages with a direct, entity-and-relation semantic bridge."""
from __future__ import annotations

from typing import NamedTuple
import torch
from torch import nn

from models.kg_encoder_rgat import RGATEncoder
from models.kg_lm_bridge import KGLMBridge
from models.scorer import ComplExScorer


class KGRepresentations(NamedTuple):
    entities: torch.Tensor
    relations: torch.Tensor


class ReverseJAKET(nn.Module):
    def __init__(self, num_entities, num_relations, cfg, lm=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.entity_emb = nn.Embedding(num_entities, cfg.kg_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        encoder_args = dict(dim=cfg.kg_dim, num_message_relations=2 * num_relations,
                            num_layers=cfg.layers_per_phase, dropout=cfg.feature_dropout,
                            heads=cfg.rgat_heads, num_blocks=cfg.num_blocks,
                            block_target_width=cfg.block_target_width,
                            attention_dropout=cfg.attention_dropout)
        self.encoder1 = RGATEncoder(**encoder_args)
        self.encoder2 = RGATEncoder(**encoder_args)
        self.scorer = ComplExScorer(num_relations, cfg.kg_dim)
        self.bridge = None if cfg.variant == "baseline" else KGLMBridge(
            kg_dim=cfg.kg_dim, model_name=cfg.lm_name, lm=lm,
            activation_checkpointing=cfg.lm_activation_checkpointing)

    def _project_text(self, vectors, global_ids, tokens, condition_on_kg):
        if tokens is None:
            raise ValueError("Semantic encoding requires aligned entity AND relation tokens")
        ids = global_ids.detach().cpu()
        # Missing descriptions are empty tokenized strings, not a bypass or
        # residual blend: every representation still follows the same LM path.
        return self.bridge(vectors, tokens.input_ids[ids], tokens.attention_mask[ids],
                           micro_batch=self.cfg.lm_micro_batch, condition_on_kg=condition_on_kg)

    def _encode(self, nodes, edges, types, tokens, use_text, full):
        entities = self.entity_emb(nodes)
        relations = self.scorer.relation_emb.weight
        def graph_stage(encoder, h, r):
            if full:
                h = encoder.encode_full(h, edges, types, self.cfg.eval_node_batch_size, relations=r)
            else:
                h = encoder(h, edges, types, relations=r)
            return h, encoder.encode_relations(r, h, edges, types)

        active = use_text and self.bridge is not None
        if active and (tokens is None or tokens.relations is None):
            raise ValueError("Semantic encoding requires aligned entity AND relation tokens")
        relation_ids = torch.arange(len(relations), device=nodes.device)
        # Simple text-enhanced control: unconditioned description embeddings
        # replace the initial features, then pass through the same graph stack.
        if active and self.cfg.variant == "text_enhanced":
            entities = self._project_text(entities, nodes, tokens, False)
            relations = self._project_text(relations, relation_ids, tokens.relations, False)
        h1, r1 = graph_stage(self.encoder1, entities, relations)
        if active and self.cfg.variant == "proposed":
            h1 = self._project_text(h1, nodes, tokens, True)
            r1 = self._project_text(r1, relation_ids, tokens.relations, True)
        # No activation, residual semantic blend, gate or normalization between
        # LM-to-KG projection and encoder 2. Graph layers have their own residuals.
        h2, r2 = graph_stage(self.encoder2, h1, r1)
        return KGRepresentations(h2, r2)

    def encode_local(self, global_nodes, edge_index, edge_type, tokens=None, use_text=False):
        return self._encode(global_nodes, edge_index, edge_type, tokens, use_text, False)

    @torch.no_grad()
    def encode_full(self, edge_index, edge_type, tokens=None, use_text=False):
        if self.training:
            raise RuntimeError("Full-graph encoding is evaluation-only")
        nodes = torch.arange(self.entity_emb.num_embeddings, device=self.entity_emb.weight.device)
        return self._encode(nodes, edge_index, edge_type, tokens, use_text, True)

    def score_triples(self, representations, triples):
        h = representations.entities[triples[:, 0]]
        t = representations.entities[triples[:, 2]]
        return self.scorer.score(h, triples[:, 1], t, relations=representations.relations)

    def trainable_state_dict(self):
        """Omit frozen pretrained weights; reconstruct them from recorded lm_name."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("bridge.lm.")}

    def load_trainable_state_dict(self, state):
        result = self.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("bridge.lm.")]
        if missing or result.unexpected_keys:
            raise ValueError(f"Checkpoint architecture mismatch: {missing}, {result.unexpected_keys}")
