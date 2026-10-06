"""Relation-aware graph attention with entity and relation feature updates."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import RGATConv


def resolve_num_blocks(dim: int, target_width: int = 4) -> int:
    """Choose a divisor giving blocks no wider than target_width."""
    for width in range(min(target_width, dim), 0, -1):
        if dim % width == 0:
            return dim // width
    raise ValueError("dim and target_width must be positive")


class RGATEncoder(nn.Module):
    def __init__(self, dim, num_message_relations, num_layers=1, dropout=0.1,
                 heads=2, num_bases=None, num_blocks=None,
                 block_target_width=4, attention_dropout=None):
        super().__init__()
        if dim < 1 or num_layers < 1 or heads < 1 or block_target_width < 1:
            raise ValueError("Dimensions, layers, heads and block width must be positive")
        if num_bases is not None:
            raise ValueError("This architecture uses block decomposition, not num_bases")
        self.num_blocks = num_blocks or resolve_num_blocks(dim, block_target_width)
        if self.num_blocks < 1 or dim % self.num_blocks:
            raise ValueError("num_blocks must divide the KG feature width")
        self.layers = nn.ModuleList([
            RGATConv(dim, dim, num_relations=num_message_relations,
                     heads=heads, concat=False, num_blocks=self.num_blocks, edge_dim=dim,
                     dropout=dropout if attention_dropout is None else attention_dropout,
                     attention_mechanism="across-relation",
                     attention_mode="additive-self-attention")
            for _ in range(num_layers)
        ])
        # PyG allocates optional mod parameters even with mod=None. They are
        # unused, and some versions leave l2 uninitialized/non-finite.
        for layer in self.layers:
            for name in ("w", "l1", "b1", "l2", "b2"):
                param = getattr(layer, name, None)
                if isinstance(param, nn.Parameter):
                    nn.init.zeros_(param)
                    param.requires_grad_(False)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.layers])
        self.feature_dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        self.relation_update = nn.Linear(dim, dim)
        self.relation_norm = nn.LayerNorm(dim)

    @staticmethod
    def _edge_features(relations, edge_type):
        if relations is None:
            return None
        values = relations[edge_type % len(relations)]
        return values * torch.where(edge_type < len(relations), 1, -1).unsqueeze(-1)

    def encode_relations(self, relations, entities, edge_index, edge_type):
        """Graph readout for relations, an implementation detail left open by the PDF.

        Only message-passing edges are used. Inverse edges reverse the endpoint
        difference so both orientations describe the same canonical relation.
        """
        src, dst = edge_index
        rel_ids = edge_type % len(relations)
        signs = torch.where(edge_type < len(relations), 1, -1).unsqueeze(-1)
        messages = (entities[dst] - entities[src]) * signs
        context = torch.zeros_like(relations).index_add(0, rel_ids, messages.to(relations.dtype))
        counts = torch.bincount(rel_ids, minlength=len(relations)).clamp_min(1).unsqueeze(-1)
        return self.relation_norm(relations + self.relation_update(context / counts))

    def forward(self, x, edge_index, edge_type, relations=None):
        h = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            msg = conv(self.feature_dropout(h), edge_index, edge_type,
                       edge_attr=self._edge_features(relations, edge_type))
            h = norm(h + self.feature_dropout(msg))
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h

    @torch.no_grad()
    def encode_full(self, x, edge_index, edge_type, node_batch_size=128, relations=None):
        """Exact evaluation by destination chunks, retaining ALL neighbors.

        Each destination's entire neighborhood stays in one attention call,
        preserving across-relation softmax while bounding per-edge allocations.
        """
        if self.training:
            raise RuntimeError("encode_full requires eval mode")
        if node_batch_size < 1:
            raise ValueError("node_batch_size must be positive")
        h = x
        order = edge_index[1].argsort()
        edges, types = edge_index[:, order], edge_type[order]
        counts = torch.bincount(edges[1], minlength=len(h)).cpu()
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            out = torch.empty_like(h)
            for start in range(0, len(h), node_batch_size):
                end = min(start + node_batch_size, len(h))
                lo, hi = int(offsets[start]), int(offsets[end])
                msg = conv(h, edges[:, lo:hi], types[lo:hi],
                           edge_attr=self._edge_features(relations, types[lo:hi]))[start:end]
                value = norm(h[start:end] + msg)
                out[start:end] = self.act(value) if i < len(self.layers) - 1 else value
            h = out
        return h
