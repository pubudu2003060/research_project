"""Identical sampled batches and loss for training, preflight and profiling."""
from dataclasses import dataclass

import torch

from preprocessing.sampling import sample_local_graph
from training.losses import bce_loss
from training.negative_sampling import sample_negatives
from training.precision import precision_settings
from training.profiling import stage


@dataclass
class TrainingBatch:
    nodes: torch.Tensor
    edges: torch.Tensor
    types: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor
    scored_relations: torch.Tensor


def prepare_batch(pos, dataset, known_train, incoming, cfg, negative_rng, graph_rng, device, profiler=None):
    with stage(profiler, "negative_sampling", cpu=True):
        neg = sample_negatives(pos, dataset.num_entities, known_train, cfg.num_negatives,
                               negative_rng, balanced_head_tail=cfg.balanced_head_tail_negatives)
    with stage(profiler, "local_graph_sampling", cpu=True):
        seeds = torch.cat([pos[:, [0, 2]].flatten(), neg[..., [0, 2]].flatten()]).unique().tolist()
        local = sample_local_graph(incoming, seeds, 2 * cfg.layers_per_phase,
                                   cfg.neighbors_per_hop, graph_rng)
        local_pos, local_neg = local.remap(pos), local.remap(neg)
        scored = torch.cat((pos[:, 1], neg[..., 1].flatten())).unique()
    with stage(profiler, "batch_transfer"):
        return TrainingBatch(local.global_nodes.to(device), local.edge_index.to(device),
                             local.edge_type.to(device), local_pos.to(device), local_neg.to(device),
                             scored.to(device))


def batch_loss(model, batch, tokens, cfg, use_text):
    enabled, dtype = precision_settings(cfg, batch.nodes.device)
    with torch.autocast(device_type=batch.nodes.device.type, dtype=dtype, enabled=enabled):
        representations = model.encode_local(batch.nodes, batch.edges, batch.types, tokens, use_text,
                                               scored_relation_ids=batch.scored_relations)
        with stage(model.profiler, "complex_scoring_and_loss"):
            positive = model.score_triples(representations, batch.positives)
            negative = model.score_triples(representations, batch.negatives.reshape(-1, 3))
            return bce_loss(positive, negative.reshape(len(batch.positives), -1))
