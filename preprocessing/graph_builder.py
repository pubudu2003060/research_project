"""Build a training-only message-passing graph from KG triples.

Only `train` triples are ever used to construct edges here — validation and
test triples must never be visible to the graph encoder. `assert_no_leakage`
is a cheap sanity check meant to
catch mistakes (e.g. accidentally concatenating splits) before any model
code is written or trained.
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.dataset import IdTriple


@dataclass
class TrainGraph:
    """Edge list for message passing, built from training triples only.

    `edge_index` is a list of (src, dst) pairs and `edge_type` the relation
    id for each edge, kept as plain Python lists here (framework-agnostic) —
    convert to torch tensors / a PyG `Data` object inside the modeling code
    (Phase 1), not here. Inverse edges (t -> h with relation id
    `r + num_relations`) are added by default because relation-aware
    encoders need messages to flow in both directions along a
    relation.
    """

    edge_index: list[tuple[int, int]]
    edge_type: list[int]
    num_entities: int
    num_relations: int  # original relation count, BEFORE inverse doubling
    has_inverse_edges: bool


def build_train_graph(
    train_triples: list[IdTriple],
    num_entities: int,
    num_relations: int,
    add_inverse_edges: bool = True,
) -> TrainGraph:
    edge_index: list[tuple[int, int]] = []
    edge_type: list[int] = []

    for h, r, t in train_triples:
        edge_index.append((h, t))
        edge_type.append(r)
        if add_inverse_edges:
            edge_index.append((t, h))
            edge_type.append(r + num_relations)

    return TrainGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        num_entities=num_entities,
        num_relations=num_relations,
        has_inverse_edges=add_inverse_edges,
    )


def assert_no_leakage(
    graph: TrainGraph,
    train_triples: list[IdTriple],
    valid_triples: list[IdTriple],
    test_triples: list[IdTriple],
) -> None:
    """Fail loudly on the most likely graph-construction mistakes.

    This can't prove by inspecting `graph` alone that it excludes val/test
    structure (a (h, t) pair might legitimately recur across splits with a
    different relation), so instead it re-derives the expected edge count
    from `train_triples` and checks it matches — catching bugs like
    accidentally concatenating splits before building the graph, or
    double-counting edges. It also flags any *exact* (h, r, t) triple
    overlap between splits, which is a dataset-integrity issue rather than
    a graph-building bug, but worth surfacing either way.
    """
    expected_edges = len(train_triples) * (2 if graph.has_inverse_edges else 1)
    if len(graph.edge_index) != expected_edges:
        raise AssertionError(
            f"Train graph has {len(graph.edge_index)} edges, expected "
            f"{expected_edges} from {len(train_triples)} training triples. "
            "Check that build_train_graph() was called with train triples only."
        )

    train_set = set(train_triples)
    overlap_valid = train_set.intersection(valid_triples)
    overlap_test = train_set.intersection(test_triples)
    if overlap_valid:
        raise AssertionError(
            f"{len(overlap_valid)} triples appear in both train and valid — "
            "this is a dataset/split issue, check how the splits were built."
        )
    if overlap_test:
        raise AssertionError(
            f"{len(overlap_test)} triples appear in both train and test — "
            "this is a dataset/split issue, check how the splits were built."
        )
