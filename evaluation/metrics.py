"""Filtered link-prediction evaluation: MRR, Hits@1, Hits@3, Hits@10.

Filtering removes OTHER known-true candidates from the ranking (for both
head and tail prediction) before computing the rank of the correct answer —
the standard KGC evaluation protocol. `build_filter_index` should be built
from every split combined (train+valid+test) — this is a different,
broader "known" set than `training.negative_sampling`'s train-only one, and
serves a different purpose: it never touches training, only ranking.
"""

from __future__ import annotations

from collections import defaultdict

import torch

IdTriple = tuple[int, int, int]


def build_filter_index(
    all_triples: list[IdTriple],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    """Returns (hr_to_tails, rt_to_heads) built from every split combined —
    used only to filter candidates out of a ranking, never as training
    signal.
    """
    hr_to_tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    rt_to_heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in all_triples:
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)
    return hr_to_tails, rt_to_heads


@torch.no_grad()
def evaluate_filtered(
    model,
    entity_repr: torch.Tensor,
    eval_triples: list[IdTriple],
    hr_to_tails: dict[tuple[int, int], set[int]],
    rt_to_heads: dict[tuple[int, int], set[int]],
    batch_size: int = 64,
) -> dict[str, float]:
    """Computes filtered MRR/Hits@K for tail and head prediction, averaged
    together into one set of numbers."""
    if not eval_triples:
        raise ValueError("Cannot evaluate an empty split")
    relations = None
    if hasattr(entity_repr, "entities"):
        relations = entity_repr.relations
        entity_repr = entity_repr.entities
    device = entity_repr.device
    triples_t = torch.tensor(eval_triples, dtype=torch.long, device=device)

    head_ranks, tail_ranks = [], []
    for start in range(0, len(triples_t), batch_size):
        batch = triples_t[start : start + batch_size]
        h_ids, r_ids, t_ids = batch[:, 0], batch[:, 1], batch[:, 2]
        h = entity_repr[h_ids]
        t = entity_repr[t_ids]
        row_ids = torch.arange(len(batch), device=device)

        # --- Tail prediction: (h, r, ?) ---
        tail_scores = model.scorer.score_all_tails(entity_repr, h, r_ids, relations=relations)
        if not torch.isfinite(tail_scores).all():
            raise RuntimeError("Non-finite tail scores during evaluation")
        true_tail_scores = tail_scores[row_ids, t_ids].clone()
        for i, (hh, rr, tt) in enumerate(batch.tolist()):
            known = hr_to_tails.get((hh, rr), set())
            if known:
                idx = torch.tensor(list(known), dtype=torch.long, device=device)
                tail_scores[i, idx] = float("-inf")
            tail_scores[i, tt] = true_tail_scores[i]
        # Average rank for ties; an all-equal scorer must not get perfect MRR.
        tail_rank = 1 + (tail_scores > true_tail_scores[:, None]).sum(dim=1)
        tail_rank = tail_rank + 0.5 * ((tail_scores == true_tail_scores[:, None]).sum(dim=1) - 1)
        tail_ranks.append(tail_rank.float())

        # --- Head prediction: (?, r, t) ---
        head_scores = model.scorer.score_all_heads(entity_repr, r_ids, t, relations=relations)
        if not torch.isfinite(head_scores).all():
            raise RuntimeError("Non-finite head scores during evaluation")
        true_head_scores = head_scores[row_ids, h_ids].clone()
        for i, (hh, rr, tt) in enumerate(batch.tolist()):
            known = rt_to_heads.get((rr, tt), set())
            if known:
                idx = torch.tensor(list(known), dtype=torch.long, device=device)
                head_scores[i, idx] = float("-inf")
            head_scores[i, hh] = true_head_scores[i]
        head_rank = 1 + (head_scores > true_head_scores[:, None]).sum(dim=1)
        head_rank = head_rank + 0.5 * ((head_scores == true_head_scores[:, None]).sum(dim=1) - 1)
        head_ranks.append(head_rank.float())

    heads, tails = torch.cat(head_ranks), torch.cat(tail_ranks)
    all_ranks = torch.cat([heads, tails])
    return {
        "MRR": float((1.0 / all_ranks).mean()),
        "Hits@1": float((all_ranks <= 1).float().mean()),
        "Hits@3": float((all_ranks <= 3).float().mean()),
        "Hits@10": float((all_ranks <= 10).float().mean()),
        "Head_MRR": float((1.0 / heads).mean()),
        "Tail_MRR": float((1.0 / tails).mean()),
    }
