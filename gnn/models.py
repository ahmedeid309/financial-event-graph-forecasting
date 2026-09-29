#!/usr/bin/env python3
"""News-only temporal event encoder.

One model: ``TemporalEventGNN``. It reads a day's event subgraph around a target
ticker and predicts the direction of that ticker's move. There is no price
pathway at all -- prices belong to the downstream forecaster, not here.

Three design decisions are baked in rather than configurable, because each was
measured and the alternative was worse:

* **The readout query is built from the target's own updated node state.**
  Without it the query is a ticker lookup, identical on every date, so the
  attention criterion cannot adapt to what happened that day.
* **Attention scores are cosine, not dot product.** Unbounded dot scores over
  sqrt(head_dim) let key norms grow until the softmax saturates; measured, every
  head collapsed onto a single event at exactly 100.000%.
* **Structural relevance is concatenated onto the readout key.** These features
  exist on the nodes, but by readout time they have been through the input
  projection and two rounds of message passing. Attention needs them unmixed.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Run this with the project Python environment/venv.") from exc


# Temporal-feature columns appended to the readout key. These answer "is this
# event about the forecast target, and how strongly" -- the question the
# attention was measured failing to ask.
READOUT_RELEVANCE_COLUMNS = [7, 10, 11, 12, 14, 15]
#                            ^importance ^polarity ^is_one_hop ^hop_dist
#                                              ^anchor_impact ^semantic_relevance
READOUT_RELEVANCE_DIM = len(READOUT_RELEVANCE_COLUMNS)


def segment_softmax(scores: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax over variable-length groups packed into one flat tensor."""

    if scores.numel() == 0:
        return scores
    maxima = torch.full(
        (num_segments,) + scores.shape[1:], float("-inf"), dtype=scores.dtype, device=scores.device
    )
    maxima = maxima.index_reduce(0, segment_ids, scores, "amax", include_self=True)
    maxima = torch.nan_to_num(maxima, neginf=0.0)
    exponent = torch.exp(scores - maxima[segment_ids])
    denominator = torch.zeros_like(maxima).index_add_(0, segment_ids, exponent)
    return exponent / denominator[segment_ids].clamp_min(1e-12)


class RelationWeightedGraphLayer(nn.Module):
    """Relation-aware weighted message passing without one Python loop per relation.

    A shared message projection is modulated by learned relation gates and
    biases, so each edge stays relation-specific while the work stays linear in
    active edges rather than in the number of relation types.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        dropout: float = 0.1,
        normalization: str = "sqrt",
    ) -> None:
        super().__init__()
        if normalization not in {"mean", "sqrt", "sum"}:
            raise ValueError(f"Unsupported normalization: {normalization}")
        self.normalization = normalization
        self.message_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.relation_gate = nn.Embedding(num_relations, hidden_dim)
        self.relation_bias = nn.Embedding(num_relations, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.relation_gate.weight)
        nn.init.zeros_(self.relation_bias.weight)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relation_ids: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(F.gelu(self.self_linear(h)))

        src, dst = edge_index[0], edge_index[1]
        relation_gate = 2.0 * torch.sigmoid(self.relation_gate(edge_relation_ids))
        relation_bias = self.relation_bias(edge_relation_ids)
        messages = (
            self.message_linear(h[src]) * relation_gate + relation_bias
        ) * edge_weights.to(h.dtype).unsqueeze(-1)

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        if self.normalization != "sum":
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            degree.index_add_(0, dst, edge_weights.abs().to(h.dtype).unsqueeze(-1))
            degree = degree.clamp_min(1.0)
            # Mean aggregation over hundreds of neighbours erases day-to-day
            # variation. Square root keeps evidence volume in the signal while
            # still bounding the message scale.
            aggregated = aggregated / (degree if self.normalization == "mean" else degree.sqrt())

        return self.norm(F.gelu(self.self_linear(h) + self.dropout(aggregated)))


class TemporalEventGNN(nn.Module):
    """Encode a day's event subgraph and predict the target ticker's direction."""

    def __init__(
        self,
        text_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_node_types: int,
        num_relations: int,
        num_event_types: int,
        num_time_roles: int,
        num_tickers: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        text_projection_dim: int = 96,
        aggregation: str = "sqrt",
        graph_gate_init: float = 2.0,
        use_graph: bool = True,
        readout_key_source: str = "post",
    ) -> None:
        super().__init__()
        # use_graph=False removes the graph pathway and leaves only ticker
        # identity. It is the floor this model has to beat: a news-only model
        # still gets a ticker embedding, so it can score above chance purely by
        # memorising which companies move more often.
        self.use_graph = bool(use_graph)
        self.text_dim = int(text_dim)
        self.temporal_dim = int(temporal_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        if hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = hidden_dim // self.num_heads

        # The 1024-d text block dwarfs every other signal and cannot be supported
        # by a few thousand labels, so it is compressed first.
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_type_embedding = nn.Embedding(num_node_types, 16)
        self.event_type_embedding = nn.Embedding(num_event_types, 24)
        self.time_role_embedding = nn.Embedding(num_time_roles, 8)
        self.ticker_embedding = nn.Embedding(num_tickers, 16)

        node_input_dim = text_projection_dim + temporal_dim + 16 + 24 + 8
        self.input_projection = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                RelationWeightedGraphLayer(
                    hidden_dim, num_relations, dropout=dropout, normalization=aggregation
                )
                for _ in range(num_layers)
            ]
        )

        # Readout. The query is target state + ticker identity; the key carries
        # the event state plus its structural relevance.
        #
        # readout_key_source decides WHICH event state the key sees:
        #   "post" -- after message passing. Contextualised, but message passing
        #             also makes events more alike, which blunts discrimination.
        #   "pre"  -- the event's own content only, before any neighbour mixing.
        #   "both" -- concatenate them and let the model weigh the two.
        # The query is always post-message-passing: the Ticker node's own text is
        # a constant string, so a pre-message-passing query would be identical on
        # every date and could not adapt to the day.
        if readout_key_source not in {"post", "pre", "both"}:
            raise ValueError(f"Unsupported readout_key_source: {readout_key_source}")
        self.readout_key_source = readout_key_source
        key_state_dim = hidden_dim * (2 if readout_key_source == "both" else 1)
        self.query_projection = nn.Linear(hidden_dim + 16, hidden_dim)
        self.key_projection = nn.Linear(key_state_dim + READOUT_RELEVANCE_DIM, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        # Cosine scores live in [-1, 1]; this learned temperature sets how sharp
        # the distribution may become, so sharpness is learned rather than
        # arrived at by key norms drifting upward during training.
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

        # graph_vec, target-node state, evidence-volume pair
        graph_summary_dim = hidden_dim * 2 + 2
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(graph_summary_dim),
            nn.Linear(graph_summary_dim, embedding_dim),
        )
        self.graph_gate = nn.Parameter(torch.tensor(float(graph_gate_init)))

        head_input_dim = embedding_dim + 16
        self.direction_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.return_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_nodes(
        self,
        text_features: torch.Tensor,
        temporal_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
        time_role_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relation_ids: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (final state, pre-message-passing state).

        The pre-state is kept so the readout can optionally score events on their
        own content, before neighbour mixing makes them more alike.
        """

        h_pre = self.input_projection(
            torch.cat(
                [
                    self.text_projection(text_features),
                    temporal_features,
                    self.node_type_embedding(node_type_ids),
                    self.event_type_embedding(event_type_ids),
                    self.time_role_embedding(time_role_ids),
                ],
                dim=-1,
            )
        )
        h = h_pre
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_relation_ids, edge_weights)
        return h, h_pre

    def readout(
        self,
        h: torch.Tensor,
        node_batch: torch.Tensor,
        event_node_mask: torch.Tensor,
        target_node_indices: torch.Tensor,
        ticker_state: torch.Tensor,
        batch_size: int,
        event_relevance: torch.Tensor,
        h_pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool node states into one embedding per subgraph.

        ``h_pre`` is the node state *before* message passing, needed only when
        readout_key_source is "pre" or "both".
        """

        target_state = h[target_node_indices]
        query = self.query_projection(torch.cat([target_state, ticker_state], dim=-1))

        event_positions = torch.nonzero(event_node_mask, as_tuple=False).squeeze(-1)
        graph_vector = torch.zeros((batch_size, self.hidden_dim), dtype=h.dtype, device=h.device)
        if event_positions.numel() > 0:
            event_states = h[event_positions]
            event_batch = node_batch[event_positions]
            if self.readout_key_source == "post" or h_pre is None:
                key_state = event_states
            elif self.readout_key_source == "pre":
                key_state = h_pre[event_positions]
            else:
                key_state = torch.cat([h_pre[event_positions], event_states], dim=-1)
            keys = self.key_projection(
                torch.cat([key_state, event_relevance[event_positions]], dim=-1)
            ).view(-1, self.num_heads, self.head_dim)
            values = self.value_projection(event_states).view(-1, self.num_heads, self.head_dim)
            queries = query.view(batch_size, self.num_heads, self.head_dim)[event_batch]
            scores = (
                F.normalize(keys, dim=-1) * F.normalize(queries, dim=-1)
            ).sum(-1) * self.logit_scale.exp().clamp(max=100.0)
            attention = segment_softmax(scores, event_batch, batch_size)
            pooled = torch.zeros(
                (batch_size, self.num_heads, self.head_dim), dtype=h.dtype, device=h.device
            ).index_add_(0, event_batch, values * attention.unsqueeze(-1))
            graph_vector = self.attention_norm(pooled.reshape(batch_size, self.hidden_dim))

        event_counts = torch.zeros((batch_size,), dtype=h.dtype, device=h.device).index_add_(
            0, node_batch, event_node_mask.to(h.dtype)
        )
        node_counts = torch.zeros((batch_size,), dtype=h.dtype, device=h.device).index_add_(
            0, node_batch, torch.ones_like(node_batch, dtype=h.dtype)
        )
        # How much evidence there is, independent of what it says.
        volume = torch.stack([torch.log1p(event_counts), torch.log1p(node_counts)], dim=-1) / 5.0
        return self.graph_projection(torch.cat([graph_vector, target_state, volume], dim=-1))

    def forward(
        self,
        text_features: torch.Tensor,
        temporal_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
        time_role_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relation_ids: torch.Tensor,
        edge_weights: torch.Tensor,
        node_batch: torch.Tensor,
        event_node_mask: torch.Tensor,
        target_node_indices: torch.Tensor,
        ticker_ids: torch.Tensor,
        num_graphs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(num_graphs if num_graphs is not None else ticker_ids.shape[0])
        h, h_pre = self.encode_nodes(
            text_features,
            temporal_features,
            node_type_ids,
            event_type_ids,
            time_role_ids,
            edge_index,
            edge_relation_ids,
            edge_weights,
        )
        ticker_state = self.ticker_embedding(ticker_ids)
        graph_embedding = self.readout(
            h,
            node_batch,
            event_node_mask,
            target_node_indices,
            ticker_state,
            batch_size,
            temporal_features[:, READOUT_RELEVANCE_COLUMNS],
            h_pre,
        )
        gate = torch.sigmoid(self.graph_gate) if self.use_graph else torch.zeros((), device=h.device)
        combined = torch.cat([gate * graph_embedding, ticker_state], dim=-1)
        return (
            self.direction_head(combined).squeeze(-1),
            graph_embedding,
            self.return_head(combined).squeeze(-1),
        )
