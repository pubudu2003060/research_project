"""ComplEx scoring for every model variant.

dim is total real-valued width. ComplEx splits it into equal real/imaginary
halves: dim=32 means 16 complex coordinates.
"""

from __future__ import annotations

import torch.nn as nn


class ComplExScorer(nn.Module):
    """Re(sum(h * r * conjugate(t))), including asymmetric relations."""

    def __init__(self, num_relations: int, dim: int):
        if dim <= 0 or dim % 2:
            raise ValueError("ComplEx requires a positive, even total feature width")
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h, relation_ids, t, relations=None):
        hr, hi = h.chunk(2, dim=-1)
        rr, ri = self._relations(relation_ids, relations).chunk(2, dim=-1)
        tr, ti = t.chunk(2, dim=-1)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    def _relations(self, relation_ids, relations):
        return self.relation_emb(relation_ids) if relations is None else relations[relation_ids]

    def score_all_tails(self, entity_emb, h, relation_ids, relations=None):
        hr, hi = h.chunk(2, dim=-1)
        rr, ri = self._relations(relation_ids, relations).chunk(2, dim=-1)
        er, ei = entity_emb.chunk(2, dim=-1)
        return (hr * rr - hi * ri) @ er.t() + (hi * rr + hr * ri) @ ei.t()

    def score_all_heads(self, entity_emb, relation_ids, t, relations=None):
        tr, ti = t.chunk(2, dim=-1)
        rr, ri = self._relations(relation_ids, relations).chunk(2, dim=-1)
        er, ei = entity_emb.chunk(2, dim=-1)
        return (rr * tr + ri * ti) @ er.t() + (rr * ti - ri * tr) @ ei.t()
