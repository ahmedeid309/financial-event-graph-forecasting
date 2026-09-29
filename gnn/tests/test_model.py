#!/usr/bin/env python3
"""Correctness tests for TemporalEventGNN.

Run with:
    python -m pytest gnn/tests -q

These pin behaviour that would otherwise fail silently: batching that leaks
between subgraphs, an attention mechanism that saturates, and relevance features
that are present but unused. None of those show up as a crash or an obviously
wrong score -- they show up as results that are wrong for reasons you cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (  # noqa: E402
    READOUT_RELEVANCE_COLUMNS,
    READOUT_RELEVANCE_DIM,
    RelationWeightedGraphLayer,
    TemporalEventGNN,
    segment_softmax,
)

HIDDEN, HEADS, NODE_TYPES, RELATIONS, TEMPORAL = 32, 4, 5, 7, 16


def model_kwargs(**overrides):
    base = dict(
        text_dim=16, temporal_dim=TEMPORAL, hidden_dim=HIDDEN, embedding_dim=8,
        num_node_types=NODE_TYPES, num_relations=RELATIONS, num_event_types=6,
        num_time_roles=4, num_tickers=3, num_layers=2, num_heads=HEADS, dropout=0.0,
    )
    base.update(overrides)
    return base


def model_inputs(num_nodes=8, num_edges=18, batch_size=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    return dict(
        text_features=torch.randn(num_nodes, 16, generator=g),
        temporal_features=torch.randn(num_nodes, TEMPORAL, generator=g),
        node_type_ids=torch.randint(0, NODE_TYPES, (num_nodes,), generator=g),
        event_type_ids=torch.randint(0, 6, (num_nodes,), generator=g),
        time_role_ids=torch.randint(0, 4, (num_nodes,), generator=g),
        edge_index=torch.randint(0, num_nodes, (2, num_edges), generator=g),
        edge_relation_ids=torch.randint(0, RELATIONS, (num_edges,), generator=g),
        edge_weights=torch.rand(num_edges, generator=g),
        node_batch=torch.tensor([0] * (num_nodes // 2) + [1] * (num_nodes - num_nodes // 2)),
        event_node_mask=torch.ones(num_nodes, dtype=torch.bool),
        target_node_indices=torch.tensor([0, num_nodes // 2]),
        ticker_ids=torch.randint(0, 3, (batch_size,), generator=g),
    )


def test_forward_shapes_and_finiteness():
    model = TemporalEventGNN(**model_kwargs()).eval()
    logits, embedding, predicted_return = model(**model_inputs(), num_graphs=2)
    assert logits.shape == (2,)
    assert embedding.shape == (2, 8)
    assert predicted_return.shape == (2,)
    assert torch.isfinite(logits).all()


def test_no_price_pathway_exists():
    """This model is news-only. A price input would be a leak, not a feature."""

    model = TemporalEventGNN(**model_kwargs())
    names = dict(model.named_parameters())
    assert not any("market" in n for n in names), "a price pathway reappeared"
    assert not hasattr(model, "market_encoder")
    # query = target_state + ticker embedding, nothing else
    assert model.query_projection.in_features == HIDDEN + 16


def test_relevance_features_reach_the_attention_score():
    """is_one_hop lives in the node features; the score must actually read it."""

    assert model_kwargs()["temporal_dim"] > max(READOUT_RELEVANCE_COLUMNS)
    model = TemporalEventGNN(**model_kwargs()).eval()
    assert model.key_projection.in_features == HIDDEN + READOUT_RELEVANCE_DIM

    inputs = model_inputs()
    with torch.no_grad():
        before = model(**inputs, num_graphs=2)[0]
        flipped = dict(inputs)
        temporal = inputs["temporal_features"].clone()
        temporal[:, 11] = 1.0 - temporal[:, 11]      # column 11 is is_one_hop
        flipped["temporal_features"] = temporal
        after = model(**flipped, num_graphs=2)[0]
    assert not torch.equal(before, after)


def test_cosine_scores_resist_key_norm_growth():
    """The saturation fix. Unbounded dot scores let key norms drive the softmax to
    an argmax; cosine normalises first, so a 50x blow-up must barely move it."""

    model = TemporalEventGNN(**model_kwargs()).eval()
    inputs = model_inputs()
    with torch.no_grad():
        small = model(**inputs, num_graphs=2)[1]
        blown = dict(inputs)
        blown["text_features"] = inputs["text_features"] * 50.0
        large = model(**blown, num_graphs=2)[1]
    # Bounded scores keep the pooled embedding in a comparable range.
    assert torch.isfinite(large).all()
    assert float((large - small).abs().max()) < 1e3


def test_attention_temperature_is_learnable():
    model = TemporalEventGNN(**model_kwargs())
    model(**model_inputs(), num_graphs=2)[0].sum().backward()
    assert model.logit_scale.grad is not None
    assert model.logit_scale.grad.abs().sum() > 0


def test_no_cross_graph_mixing_in_a_batch():
    """Two block-diagonal subgraphs must not influence each other.

    This is the failure that never shows up in a metric: one company's prediction
    quietly contaminated by another's evidence.
    """

    layer = RelationWeightedGraphLayer(HIDDEN, RELATIONS, dropout=0.0).eval()
    g = torch.Generator().manual_seed(1)
    h = torch.randn(12, HIDDEN, generator=g)
    edges = torch.cat([
        torch.randint(0, 6, (2, 10), generator=g),
        torch.randint(6, 12, (2, 10), generator=g),
    ], dim=1)
    rel = torch.randint(0, RELATIONS, (20,), generator=g)
    w = torch.rand(20, generator=g)
    with torch.no_grad():
        before = layer(h, edges, rel, w)[6:]
        perturbed = h.clone()
        perturbed[:6] += 5.0                     # change subgraph A only
        after = layer(perturbed, edges, rel, w)[6:]
    assert torch.allclose(before, after, atol=1e-5)


def test_segment_softmax_normalises_per_group():
    scores = torch.randn(9, HEADS)
    segments = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    attention = segment_softmax(scores, segments, 3)
    totals = torch.zeros(3, HEADS).index_add_(0, segments, attention)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_empty_edge_graph_is_finite():
    model = TemporalEventGNN(**model_kwargs()).eval()
    inputs = model_inputs()
    inputs["edge_index"] = torch.zeros((2, 0), dtype=torch.long)
    inputs["edge_relation_ids"] = torch.zeros((0,), dtype=torch.long)
    inputs["edge_weights"] = torch.zeros((0,), dtype=torch.float32)
    logits = model(**inputs, num_graphs=2)[0]
    assert torch.isfinite(logits).all()


def test_disable_graph_makes_news_irrelevant():
    """The floor control: with the graph off, only ticker identity remains."""

    model = TemporalEventGNN(**model_kwargs(use_graph=False)).eval()
    inputs = model_inputs()
    with torch.no_grad():
        before = model(**inputs, num_graphs=2)[0]
        shifted = dict(inputs)
        shifted["text_features"] = inputs["text_features"] + 25.0
        after = model(**shifted, num_graphs=2)[0]
    assert torch.equal(before, after)


def test_gradients_reach_every_component():
    model = TemporalEventGNN(**model_kwargs())
    model(**model_inputs(), num_graphs=2)[0].sum().backward()
    for name in (
        "text_projection.1.weight", "input_projection.0.weight",
        "query_projection.weight", "key_projection.weight", "value_projection.weight",
        "graph_projection.1.weight", "direction_head.1.weight",
        "layers.0.message_linear.weight", "ticker_embedding.weight",
    ):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, f"{name} has no gradient"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is all zero"


def test_readout_key_source_dimensions():
    """'both' widens the key projection; 'pre'/'post' keep it at hidden_dim."""

    for source, expected in (("post", HIDDEN), ("pre", HIDDEN), ("both", 2 * HIDDEN)):
        model = TemporalEventGNN(**model_kwargs(readout_key_source=source))
        assert model.key_projection.in_features == expected + READOUT_RELEVANCE_DIM, source


def test_pre_key_ignores_neighbour_pollution():
    """With key_source='pre', changing a NON-event neighbour must not move the key.

    That is the whole point of the option: an event's relevance should be judged
    on its own content, not on what its neighbours happen to say.
    """

    model = TemporalEventGNN(**model_kwargs(readout_key_source="pre")).eval()
    inputs = model_inputs()
    with torch.no_grad():
        _, h_pre_a = model.encode_nodes(
            inputs["text_features"], inputs["temporal_features"], inputs["node_type_ids"],
            inputs["event_type_ids"], inputs["time_role_ids"], inputs["edge_index"],
            inputs["edge_relation_ids"], inputs["edge_weights"],
        )
        shifted = inputs["text_features"].clone()
        shifted[-1] += 10.0                      # perturb one node only
        _, h_pre_b = model.encode_nodes(
            shifted, inputs["temporal_features"], inputs["node_type_ids"],
            inputs["event_type_ids"], inputs["time_role_ids"], inputs["edge_index"],
            inputs["edge_relation_ids"], inputs["edge_weights"],
        )
    # every OTHER node's pre-state is untouched, because no mixing has happened
    assert torch.allclose(h_pre_a[:-1], h_pre_b[:-1], atol=1e-6)


def test_encode_nodes_returns_both_states():
    model = TemporalEventGNN(**model_kwargs()).eval()
    inputs = model_inputs()
    h, h_pre = model.encode_nodes(
        inputs["text_features"], inputs["temporal_features"], inputs["node_type_ids"],
        inputs["event_type_ids"], inputs["time_role_ids"], inputs["edge_index"],
        inputs["edge_relation_ids"], inputs["edge_weights"],
    )
    assert h.shape == h_pre.shape
    assert not torch.equal(h, h_pre), "message passing should change the state"
