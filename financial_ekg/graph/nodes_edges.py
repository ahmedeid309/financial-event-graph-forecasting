from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from financial_ekg.graph.embeddings import fit_feature_dim, get_text_embeddings
from financial_ekg.models import GraphEdge, GraphNode
from financial_ekg.utils.serialization import json_dumps
from financial_ekg.utils.text import clean_text

def add_node(nodes: Dict[str, GraphNode], node_id: str, node_type: str, name: str, ticker: str = "", date: str = "", attrs: Optional[Dict[str, Any]] = None) -> None:
    """Add a node to the node accumulator if it does not exist.

    Args:
        nodes: Mutable node dictionary.
        node_id: Stable node identifier.
        node_type: Semantic node type.
        name: Human-readable node label.
        ticker: Optional ticker.
        date: Optional date.
        attrs: Optional JSON attributes.

    Returns:
        None.
    """
    if node_id not in nodes:
        nodes[node_id] = GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            ticker=ticker,
            date=date,
            attributes_json=json_dumps(attrs or {}),
        )


def add_edge(edges: Dict[Tuple[str, str, str, str, str], GraphEdge], source: str, target: str, edge_type: str, weight: float = 1.0, article_id: str = "", event_id: str = "", attrs: Optional[Dict[str, Any]] = None) -> None:
    """Add an edge to the edge accumulator if it does not exist.

    Args:
        edges: Mutable edge dictionary.
        source: Source node ID.
        target: Target node ID.
        edge_type: Semantic relation type.
        weight: Edge weight.
        article_id: Article provenance ID.
        event_id: Event provenance ID.
        attrs: Optional JSON attributes.

    Returns:
        None.
    """
    key = (source, target, edge_type, article_id, event_id)
    if key not in edges:
        edges[key] = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=float(weight),
            article_id=article_id,
            event_id=event_id,
            attributes_json=json_dumps(attrs or {}),
        )


def add_semantic_entity_company_resolution(
    unresolved_entities: Sequence[Dict[str, str]],
    company_records: Dict[str, Dict[str, str]],
    nodes: Dict[str, GraphNode],
    edges: Dict[Tuple[str, str, str, str, str], GraphEdge],
    args: argparse.Namespace,
    ensure_company_node_fn: Any,
) -> None:
    """Resolve unmatched entity nodes to company nodes using text similarity.

    Args:
        unresolved_entities: Entity mentions that were not resolved by ticker
            or conservative alias matching.
        company_records: Registered company records keyed by company node ID.
        nodes: Mutable graph-node accumulator.
        edges: Mutable graph-edge accumulator.
        args: Parsed CLI arguments controlling embedding model and threshold.
        ensure_company_node_fn: Callback that materializes a company node by
            company node ID.

    Returns:
        None.
    """
    if not unresolved_entities or not company_records:
        return
    threshold = float(getattr(args, "entity_resolution_threshold", 0.96) or 0.96)
    if threshold <= 0:
        return

    entity_items = [item for item in unresolved_entities if clean_text(item.get("entity_name", ""))]
    company_items = [
        {"company_id": comp_id, **record}
        for comp_id, record in company_records.items()
        if clean_text(record.get("name", ""))
    ]
    if not entity_items or not company_items:
        return

    entity_texts = [clean_text(item["entity_name"]) for item in entity_items]
    company_texts = [clean_text(item["name"]) for item in company_items]
    dim = int(getattr(args, "feature_embedding_dim", 64) or 64)
    entity_emb = get_text_embeddings(entity_texts, getattr(args, "embedding_model", ""), getattr(args, "device", "cpu"), int(getattr(args, "embedding_batch_size", 64) or 64), fallback_dim=dim)
    company_emb = get_text_embeddings(company_texts, getattr(args, "embedding_model", ""), getattr(args, "device", "cpu"), int(getattr(args, "embedding_batch_size", 64) or 64), fallback_dim=dim)
    common_dim = max(entity_emb.shape[1], company_emb.shape[1])
    entity_emb = fit_feature_dim(entity_emb, common_dim)
    company_emb = fit_feature_dim(company_emb, common_dim)

    def normalize_rows(arr: np.ndarray) -> np.ndarray:
        """L2-normalize each row in an embedding matrix.

        Args:
            arr: Embedding matrix to normalize.

        Returns:
            Row-normalized matrix with zero rows left unchanged.
        """
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    entity_emb = normalize_rows(entity_emb)
    company_emb = normalize_rows(company_emb)
    sims = entity_emb @ company_emb.T
    for ent_idx, item in enumerate(entity_items):
        best_idx = int(np.argmax(sims[ent_idx]))
        best_sim = float(sims[ent_idx, best_idx])
        if best_sim < threshold:
            continue
        company_id = clean_text(company_items[best_idx]["company_id"])
        ensure_company_node_fn(company_id)
        add_edge(
            edges,
            clean_text(item["entity_id"]),
            company_id,
            "SEMANTICALLY_RESOLVES_TO_COMPANY",
            weight=best_sim,
            article_id=clean_text(item.get("article_id", "")),
            event_id=clean_text(item.get("event_id", "")),
            attrs={"similarity": round(best_sim, 4), "entity_name": item.get("entity_name", "")},
        )
        add_edge(
            edges,
            clean_text(item["event_node"]),
            company_id,
            "SEMANTIC_MENTIONS_COMPANY",
            weight=best_sim,
            article_id=clean_text(item.get("article_id", "")),
            event_id=clean_text(item.get("event_id", "")),
            attrs={"similarity": round(best_sim, 4), "entity_name": item.get("entity_name", "")},
        )
