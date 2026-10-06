"""Training-only epoch split and reproducible incoming-neighbor sampling."""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


def split_epoch_train(train, mp_ratio, seed):
    train = torch.as_tensor(train, dtype=torch.long).reshape(-1, 3).cpu()
    if len(train) < 2 or not 0 < mp_ratio < 1:
        raise ValueError("Need at least two training triples and 0 < mp_ratio < 1")
    if len(set(map(tuple, train.tolist()))) != len(train):
        raise ValueError("Duplicate training triples would break disjoint epoch splitting")
    order = torch.from_numpy(np.random.default_rng(seed).permutation(len(train)))
    n_mp = min(max(1, int(len(train) * mp_ratio)), len(train) - 1)
    return train[order[:n_mp]], train[order[n_mp:]]


def build_incoming_adjacency(mp_triples, num_entities, num_relations):
    incoming = [[] for _ in range(num_entities)]
    for h, r, t in torch.as_tensor(mp_triples).tolist():
        incoming[t].append((h, r))
        incoming[h].append((t, r + num_relations))
    return incoming


@dataclass
class LocalGraph:
    global_nodes: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor

    def remap(self, triples):
        result = triples.clone()
        # global_nodes is sorted; all scoring endpoints are sampling seeds.
        result[..., 0] = torch.searchsorted(self.global_nodes, triples[..., 0].contiguous())
        result[..., 2] = torch.searchsorted(self.global_nodes, triples[..., 2].contiguous())
        return result


def sample_local_graph(incoming, seeds, num_hops, neighbors_per_hop, rng: random.Random):
    """Sample one bounded incoming neighborhood per visited destination.

    Include positive AND negative endpoints. Two RGAT phases require
    2 * layers_per_phase hops. All sampled nodes with text are eligible for
    semantic enrichment; boundary nodes provide structural context.
    """
    nodes = set(int(x) for x in seeds)
    frontier = sorted(nodes)
    expanded = set()
    edges = set()
    for _ in range(num_hops):
        next_frontier = set()
        for dst in frontier:
            if dst in expanded:
                continue
            expanded.add(dst)
            choices = incoming[dst]
            if len(choices) > neighbors_per_hop:
                choices = rng.sample(choices, neighbors_per_hop)
            for src, rel in choices:
                edges.add((src, dst, rel))
                nodes.add(src)
                if src not in expanded:
                    next_frontier.add(src)
        frontier = sorted(next_frontier)
    ordered_nodes = sorted(nodes)
    mapping = {node: i for i, node in enumerate(ordered_nodes)}
    ordered_edges = sorted(edges)
    edge_index = torch.tensor([(mapping[s], mapping[t]) for s, t, _ in ordered_edges],
                              dtype=torch.long).reshape(-1, 2).t().contiguous()
    edge_type = torch.tensor([r for _, _, r in ordered_edges], dtype=torch.long)
    return LocalGraph(torch.tensor(ordered_nodes, dtype=torch.long), edge_index, edge_type)
