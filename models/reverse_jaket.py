"""Two RGAT stages with a direct, entity-and-relation semantic bridge."""
from __future__ import annotations

from typing import NamedTuple
import torch
from torch import nn

from models.kg_encoder_rgat import RGATEncoder
from models.kg_lm_bridge import KGLMBridge
from models.scorer import ComplExScorer
from training.profiling import stage


class KGRepresentations(NamedTuple):
    entities: torch.Tensor
    relations: torch.Tensor
    valid_relations: torch.Tensor | None = None


class ReverseJAKET(nn.Module):
    def __init__(self, num_entities, num_relations, cfg, lm=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.profiler = None
        self._token_cache = {}
        self.last_semantic_work = {}
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

    def _project_text(self, vectors, global_ids, tokens, condition_on_kg, relation=False):
        if tokens is None:
            raise ValueError("Semantic encoding requires aligned entity AND relation tokens")
        if self.cfg.optimize_semantic_batches:
            # Cache integer IDs/masks only (~15 MB for all FB15k-237 text).
            # Never cache prompt-conditioned transformer outputs. Tensor version
            # checks invalidate this cache if callers edit text in place.
            key = "relation" if relation else "entity"
            signature = (id(tokens), tokens.input_ids.data_ptr(), tokens.input_ids._version,
                         tokens.attention_mask.data_ptr(), tokens.attention_mask._version,
                         vectors.device)
            cached = self._token_cache.get(key)
            if cached is None or cached[0] != signature:
                cached = (signature, tokens, tokens.input_ids.to(vectors.device),
                          tokens.attention_mask.to(vectors.device))
                self._token_cache[key] = cached
            ids = global_ids.to(vectors.device)
            input_ids, mask = cached[2][ids], cached[3][ids]
        else:
            ids = global_ids.detach().cpu()
            input_ids, mask = tokens.input_ids[ids], tokens.attention_mask[ids]
        # Missing descriptions are empty tokenized strings, not a bypass or
        # residual blend: every representation still follows the same LM path.
        micro_batch = (self.cfg.relation_lm_micro_batch or self.cfg.lm_micro_batch) if relation else self.cfg.lm_micro_batch
        return self.bridge(vectors, input_ids, mask, micro_batch=micro_batch,
                           condition_on_kg=condition_on_kg,
                           bulk_transfer=self.cfg.optimize_semantic_batches)

    def _encode(self, nodes, edges, types, tokens, use_text, full, scored_relation_ids=None):
        entities = self.entity_emb(nodes)
        relations = self.scorer.relation_emb.weight
        def entity_stage(encoder, h, r):
            if full:
                return encoder.encode_full(h, edges, types, self.cfg.eval_node_batch_size, relations=r)
            return encoder(h, edges, types, relations=r)

        active = use_text and self.bridge is not None
        if active and (tokens is None or tokens.relations is None):
            raise ValueError("Semantic encoding requires aligned entity AND relation tokens")
        relation_ids = torch.arange(len(relations), device=nodes.device)
        valid_relations = None
        if active and not full and self.cfg.optimize_semantic_batches and scored_relation_ids is not None:
            # Only these relation rows reach phase-2 messages or scored triples.
            # Relation readout/LayerNorm operate independently per row; RoBERTa
            # examples do not attend to other examples. Prune unused LM work,
            # retaining GLOBAL IDs (including inverse-edge modulo mapping).
            relation_ids = torch.cat((types % len(relations), scored_relation_ids.to(nodes.device).flatten())).unique()
            valid_relations = torch.zeros(len(relations), dtype=torch.bool, device=nodes.device)
            valid_relations[relation_ids] = True
        self.last_semantic_work = {"entity_prompts": len(nodes) if active else 0,
                                   "relation_prompts": len(relation_ids) if active else 0}

        def semantic_entities(h, conditioned):
            with stage(self.profiler, "entity_semantic_bridge"):
                return self._project_text(h, nodes, tokens, conditioned)

        def semantic_relations(r, conditioned):
            with stage(self.profiler, "relation_semantic_bridge"):
                projected = self._project_text(r[relation_ids], relation_ids, tokens.relations, conditioned, relation=True)
                if valid_relations is None:
                    return projected
                # Uncomputed rows are not public representations; score_triples
                # rejects their use. Full evaluation always computes every row.
                return projected.new_zeros(r.shape).index_copy(0, relation_ids, projected)
        # Simple text-enhanced control: unconditioned description embeddings
        # replace the initial features, then pass through the same graph stack.
        if active and self.cfg.variant == "text_enhanced":
            entities = semantic_entities(entities, False)
            relations = semantic_relations(relations, False)
        with stage(self.profiler, "encoder1_entities"):
            h1 = entity_stage(self.encoder1, entities, relations)
        with stage(self.profiler, "encoder1_relations"):
            r1 = self.encoder1.encode_relations(relations, h1, edges, types)
        if active and self.cfg.variant == "proposed":
            h1 = semantic_entities(h1, True)
            r1 = semantic_relations(r1, True)
        # No activation, residual semantic blend, gate or normalization between
        # LM-to-KG projection and encoder 2. Graph layers have their own residuals.
        with stage(self.profiler, "encoder2"):
            h2 = entity_stage(self.encoder2, h1, r1)
            r2 = self.encoder2.encode_relations(r1, h2, edges, types)
        return KGRepresentations(h2, r2, valid_relations)

    def encode_local(self, global_nodes, edge_index, edge_type, tokens=None, use_text=False,
                     scored_relation_ids=None):
        return self._encode(global_nodes, edge_index, edge_type, tokens, use_text, False, scored_relation_ids)

    @torch.no_grad()
    def encode_full(self, edge_index, edge_type, tokens=None, use_text=False):
        if self.training:
            raise RuntimeError("Full-graph encoding is evaluation-only")
        nodes = torch.arange(self.entity_emb.num_embeddings, device=self.entity_emb.weight.device)
        return self._encode(nodes, edge_index, edge_type, tokens, use_text, True)

    def score_triples(self, representations, triples):
        if representations.valid_relations is not None and not bool(representations.valid_relations[triples[:, 1]].all()):
            raise ValueError("Scored relation was not requested during local semantic encoding")
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
