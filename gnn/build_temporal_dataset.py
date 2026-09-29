#!/usr/bin/env python3
"""Build leakage-free temporal subgraphs from the Financial EKG.

Every forecast task receives the inbound neighbourhood around its ticker.
Evidence is selected by *availability date* -- when an article was published --
never by a semantic occurrence date. Semantic start/end dates survive as signed
relative-time features, so a forward-looking event stays usable without making
future reports visible.

No price data enters here. The model this feeds is news-only; prices belong to
the downstream forecaster.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Run this with the project Python environment/venv.") from exc


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REQUIRED_GRAPH_FILES = [
    "articles.csv", "events.csv", "nodes.csv", "edges.csv", "node_features.npy",
    "event_cluster_members.csv", "entity_cluster_members.csv",
]
TEMPORAL_FEATURE_DIM = 16
UNKNOWN_ORDINAL = -2_147_483_648

# Event types whose direction is unambiguous from the label alone, used as a weak
# prior so the encoder need not rediscover polarity from text.
POSITIVE_EVENT_TYPES = {
    "revenue_growth", "profit_growth", "analyst_upgrade", "product_launch",
    "partnership", "dividend_announcement", "share_buyback",
    "market_cap_milestone", "market_share",
}
NEGATIVE_EVENT_TYPES = {
    "revenue_decline", "profit_decline", "analyst_downgrade", "lawsuit",
    "regulatory_investigation", "layoffs", "debt_financing", "supply_chain_event",
}

# Relations carrying a real directional judgement about an anchor company. The
# bare MENTIONS_ANCHOR_COMPANY link is excluded: it records that an event names
# the anchor, not that it helps or hurts it, and its weight is always +1.0.
DIRECTIONAL_ANCHOR_RELATIONS = {
    "DIRECTLY_IMPACTS_ANCHOR", "COMPETES_WITH_ANCHOR", "SUPPLY_CUSTOMER_IMPACT_ON_ANCHOR",
    "PARTNERS_WITH_ANCHOR", "PEER_COMPARED_WITH_ANCHOR", "ECOSYSTEM_IMPACT_ON_ANCHOR",
    "PORTFOLIO_EXPOSURE_TO_ANCHOR", "MACRO_IMPACT_ON_ANCHOR", "REGULATORY_IMPACT_ON_ANCHOR",
    "MARKET_SENTIMENT_IMPACT_ON_ANCHOR", "RELATED_TO_ANCHOR",
}


@dataclass(frozen=True)
class ForecastTask:
    task_id: str
    ticker: str
    cutoff_date: date
    horizon_trading_days: int
    label_return: float
    label_direction: int
    split: str


@dataclass
class GraphArtifacts:
    graph_dir: Path
    node_ids: List[str]
    node_features: np.ndarray
    node_type_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    node_type_ids: np.ndarray
    node_available_ord: np.ndarray
    node_semantic_start_ord: np.ndarray
    node_semantic_end_ord: np.ndarray
    node_time_confidence: np.ndarray
    node_event_confidence: np.ndarray
    node_importance: np.ndarray
    node_event_type_ids: np.ndarray
    node_time_role_ids: np.ndarray
    node_polarity: np.ndarray
    event_type_to_id: Dict[str, int]
    time_role_to_id: Dict[str, int]
    event_node_mask: np.ndarray
    article_node_mask: np.ndarray
    fusion_node_mask: np.ndarray
    anchor_impact_by_ticker: Dict[str, Dict[int, float]]
    events_by_ticker: Dict[str, set]
    fusion_targets_by_event: Dict[int, np.ndarray]
    fusion_members: Dict[int, np.ndarray]
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    edge_relation_ids: np.ndarray
    edge_base_weights: np.ndarray
    edge_available_ord: np.ndarray
    incoming_order: np.ndarray
    incoming_indptr: np.ndarray
    ticker_to_node_idx: Dict[str, int]


@dataclass
class TemporalExample:
    task_id: str
    ticker: str
    current_date: date
    split: str
    label_return: float
    label: int
    node_indices: np.ndarray
    node_type_ids: np.ndarray
    node_event_type_ids: np.ndarray
    node_time_role_ids: np.ndarray
    event_node_mask: np.ndarray
    temporal_features: np.ndarray
    fusion_node_mask: np.ndarray
    target_node_idx: int
    edge_index: np.ndarray
    edge_relation_ids: np.ndarray
    edge_weights: np.ndarray


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalize_ticker(value: object) -> str:
    return clean_text(value).upper()


def parse_date(value: object) -> Optional[date]:
    text = clean_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except Exception:
        return None


def date_ordinal(value: object) -> int:
    parsed = value if isinstance(value, date) else parse_date(value)
    return parsed.toordinal() if isinstance(parsed, date) else UNKNOWN_ORDINAL


def parse_tickers(value: str) -> List[str]:
    return list(dict.fromkeys(normalize_ticker(item) for item in value.split(",") if item.strip()))


def iter_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        yield from csv.DictReader(handle)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _reverse_relation_name(name: str) -> str:
    return name[4:] if name.startswith("REV_") else f"REV_{name}"


def _event_available_ord(event: Dict[str, str]) -> int:
    return date_ordinal(
        event.get("available_date", "") or event.get("article_date", "") or event.get("date", "")
    )


def load_graph_artifacts(graph_dir: str | Path, tickers: Optional[Sequence[str]] = None) -> GraphArtifacts:
    """Load the graph and build temporal/inbound indices.

    Reverse relations are synthesised only where the graph does not already carry
    the corresponding reverse type, which preserves directed semantics without
    duplicating the explicit Event<->Company reverse edges.
    """

    graph_path = Path(graph_dir)
    missing = [name for name in REQUIRED_GRAPH_FILES if not (graph_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{graph_path} is missing required graph files {missing}.")

    events = {}
    keep_event_fields = {
        "event_id", "article_id", "ticker", "date", "article_date", "available_date",
        "event_date_start", "event_date_end", "event_time_confidence", "event_type",
        "event_time_role", "importance_score", "confidence",
    }
    for row in iter_csv_rows(graph_path / "events.csv"):
        event_id = clean_text(row.get("event_id", ""))
        if event_id:
            events[event_id] = {k: clean_text(row.get(k, "")) for k in keep_event_fields}

    articles = {}
    for row in iter_csv_rows(graph_path / "articles.csv"):
        article_id = clean_text(row.get("article_id", ""))
        if article_id:
            articles[article_id] = clean_text(row.get("_date", "") or row.get("date", ""))

    node_features = np.load(graph_path / "node_features.npy", mmap_mode="r")
    if node_features.ndim != 2:
        raise ValueError("node_features.npy must be a 2D feature matrix")

    node_ids: List[str] = []
    node_id_to_idx: Dict[str, int] = {}
    node_type_to_id: Dict[str, int] = {}
    cols = {k: [] for k in (
        "type_ids", "available", "sem_start", "sem_end", "time_conf", "event_conf",
        "importance", "event_type", "time_role", "polarity",
    )}
    event_node_mask: List[bool] = []
    article_node_mask: List[bool] = []
    fusion_node_mask: List[bool] = []
    ticker_to_node_idx: Dict[str, int] = {}
    # Index 0 is reserved for "not an event" / unknown in both vocabularies.
    event_type_to_id: Dict[str, int] = {"": 0}
    time_role_to_id: Dict[str, int] = {"": 0}

    for idx, row in enumerate(iter_csv_rows(graph_path / "nodes.csv")):
        node_id = clean_text(row.get("node_id", ""))
        node_type = clean_text(row.get("node_type", ""))
        if not node_id or not node_type:
            raise ValueError("nodes.csv contains a row with empty node_id or node_type")
        if node_id in node_id_to_idx:
            raise ValueError(f"Duplicate node_id in nodes.csv: {node_id}")
        node_id_to_idx[node_id] = idx
        node_ids.append(node_id)
        node_type_to_id.setdefault(node_type, len(node_type_to_id))
        cols["type_ids"].append(node_type_to_id[node_type])

        available = UNKNOWN_ORDINAL
        sem_start = date_ordinal(row.get("date", ""))
        sem_end = sem_start
        time_conf = event_conf = importance = polarity = 0.0
        event_type_id = time_role_id = 0

        if node_type == "Event" and node_id.startswith("event:"):
            event = events.get(node_id.split(":", 1)[1])
            if event is None:
                raise ValueError(f"Event node {node_id} is missing from events.csv")
            available = _event_available_ord(event)
            sem_start = date_ordinal(event.get("event_date_start", "") or event.get("date", ""))
            sem_end = date_ordinal(
                event.get("event_date_end", "") or event.get("event_date_start", "")
                or event.get("date", "")
            )
            time_conf = _safe_float(event.get("event_time_confidence", ""))
            event_conf = _safe_float(event.get("confidence", ""))
            importance = _safe_float(event.get("importance_score", ""))
            event_type = clean_text(event.get("event_type", ""))
            time_role = clean_text(event.get("event_time_role", ""))
            event_type_id = event_type_to_id.setdefault(event_type, len(event_type_to_id))
            time_role_id = time_role_to_id.setdefault(time_role, len(time_role_to_id))
            polarity = 1.0 if event_type in POSITIVE_EVENT_TYPES else (
                -1.0 if event_type in NEGATIVE_EVENT_TYPES else 0.0
            )
        elif node_type == "Article" and node_id.startswith("article:"):
            available = date_ordinal(articles.get(node_id.split(":", 1)[1], "") or row.get("date", ""))
            sem_start = sem_end = available

        cols["available"].append(available)
        cols["sem_start"].append(sem_start)
        cols["sem_end"].append(sem_end)
        cols["time_conf"].append(time_conf)
        cols["event_conf"].append(event_conf)
        cols["importance"].append(importance)
        cols["event_type"].append(event_type_id)
        cols["time_role"].append(time_role_id)
        cols["polarity"].append(polarity)
        event_node_mask.append(node_type == "Event")
        article_node_mask.append(node_type == "Article")
        fusion_node_mask.append(node_type in {"EventCluster", "EntityCluster"})

        if node_type == "Ticker":
            ticker = normalize_ticker(
                row.get("ticker", "") or row.get("name", "") or node_id.split(":", 1)[-1]
            )
            if ticker:
                ticker_to_node_idx[ticker] = idx

    if len(node_ids) != int(node_features.shape[0]):
        raise ValueError(
            f"nodes.csv has {len(node_ids)} rows, node_features.npy has {node_features.shape[0]}."
        )

    relation_to_id: Dict[str, int] = {}
    base_sources: List[int] = []
    base_targets: List[int] = []
    base_relation_ids: List[int] = []
    base_weights: List[float] = []
    base_available_ord: List[int] = []
    anchor_impact_raw: Dict[str, Dict[int, float]] = {}
    events_by_ticker_raw: Dict[str, set] = {}
    fusion_targets_raw: Dict[int, List[int]] = {}
    fusion_members_raw: Dict[int, List[int]] = {}
    event_mask_arr = event_node_mask
    fusion_mask_arr = fusion_node_mask

    for edge in iter_csv_rows(graph_path / "edges.csv"):
        source_id = clean_text(edge.get("source", ""))
        target_id = clean_text(edge.get("target", ""))
        source = node_id_to_idx.get(source_id)
        target = node_id_to_idx.get(target_id)
        if source is None or target is None:
            raise ValueError(f"edges.csv references a missing node: {source_id!r} -> {target_id!r}")

        relation = clean_text(edge.get("edge_type", "")) or "RELATED_TO"
        relation_to_id.setdefault(relation, len(relation_to_id))

        # An edge cannot be known before its latest component, so take the max.
        candidates: List[int] = []
        event_id = clean_text(edge.get("event_id", ""))
        if event_id:
            event = events.get(event_id)
            if event is None:
                raise ValueError(f"edges.csv references missing event_id {event_id}")
            candidates.append(_event_available_ord(event))
        article_id = clean_text(edge.get("article_id", ""))
        if article_id and article_id in articles:
            candidates.append(date_ordinal(articles[article_id]))
        for node_idx in (source, target):
            if cols["available"][node_idx] != UNKNOWN_ORDINAL:
                candidates.append(cols["available"][node_idx])
        known = [v for v in candidates if v != UNKNOWN_ORDINAL]

        base_sources.append(source)
        base_targets.append(target)
        base_relation_ids.append(relation_to_id[relation])
        base_weights.append(_safe_float(edge.get("weight", 1.0), 1.0))
        base_available_ord.append(max(known) if known else UNKNOWN_ORDINAL)

        # Which company each event is actually about. Used to keep foreign
        # companies' events out of a target's subgraph; they otherwise arrive
        # through articles that merely list several tickers.
        if relation == "INVOLVES_TICKER" and target_id.startswith("ticker:"):
            events_by_ticker_raw.setdefault(
                normalize_ticker(target_id.split(":", 1)[1]), set()
            ).add(source)

        if relation in DIRECTIONAL_ANCHOR_RELATIONS and target_id.startswith("company_ticker:"):
            anchor = normalize_ticker(target_id.split(":", 1)[1])
            score = _safe_float(edge.get("weight", 0.0))
            bucket = anchor_impact_raw.setdefault(anchor, {})
            # Several events can hit the same anchor; keep the strongest signed
            # judgement rather than letting neutral links dilute it.
            if abs(score) > abs(bucket.get(source, 0.0)):
                bucket[source] = score

        if relation in {"FUSED_INTO_EVENT_CLUSTER", "MENTIONS_ENTITY_CLUSTER"} \
                and event_mask_arr[source] and fusion_mask_arr[target]:
            fusion_targets_raw.setdefault(source, []).append(target)
            fusion_members_raw.setdefault(target, []).append(source)

    base_relation_names = set(relation_to_id)
    reverse_id_by_id: Dict[int, int] = {}
    synthesize_by_id: Dict[int, bool] = {}
    for relation, relation_id in list(relation_to_id.items()):
        reverse_name = _reverse_relation_name(relation)
        synthesize_by_id[relation_id] = reverse_name not in base_relation_names
        relation_to_id.setdefault(reverse_name, len(relation_to_id))
        reverse_id_by_id[relation_id] = relation_to_id[reverse_name]

    src = np.asarray(base_sources, dtype=np.int64)
    dst = np.asarray(base_targets, dtype=np.int64)
    rel = np.asarray(base_relation_ids, dtype=np.int64)
    weight = np.asarray(base_weights, dtype=np.float32)
    available_ord = np.asarray(base_available_ord, dtype=np.int32)
    synth = np.asarray([synthesize_by_id[int(v)] for v in rel], dtype=bool)
    reverse_rel = np.asarray([reverse_id_by_id[int(v)] for v in rel[synth]], dtype=np.int64)

    edge_sources = np.concatenate([src, dst[synth]])
    edge_targets = np.concatenate([dst, src[synth]])
    edge_relation_ids = np.concatenate([rel, reverse_rel])
    edge_base_weights = np.concatenate([weight, weight[synth]])
    edge_available_ord = np.concatenate([available_ord, available_ord[synth]])

    incoming_order = np.argsort(edge_targets, kind="stable")
    incoming_counts = np.bincount(edge_targets, minlength=len(node_ids))
    incoming_indptr = np.empty(len(node_ids) + 1, dtype=np.int64)
    incoming_indptr[0] = 0
    np.cumsum(incoming_counts, out=incoming_indptr[1:])

    missing_tickers = [
        t for t in (normalize_ticker(x) for x in tickers or []) if t not in ticker_to_node_idx
    ]
    if missing_tickers:
        raise ValueError(f"Requested ticker nodes are missing from graph: {missing_tickers}")

    return GraphArtifacts(
        graph_dir=graph_path,
        node_ids=node_ids,
        node_features=node_features,
        node_type_to_id=node_type_to_id,
        relation_to_id=relation_to_id,
        node_type_ids=np.asarray(cols["type_ids"], dtype=np.int64),
        node_available_ord=np.asarray(cols["available"], dtype=np.int32),
        node_semantic_start_ord=np.asarray(cols["sem_start"], dtype=np.int32),
        node_semantic_end_ord=np.asarray(cols["sem_end"], dtype=np.int32),
        node_time_confidence=np.asarray(cols["time_conf"], dtype=np.float32),
        node_event_confidence=np.asarray(cols["event_conf"], dtype=np.float32),
        node_importance=np.asarray(cols["importance"], dtype=np.float32),
        node_event_type_ids=np.asarray(cols["event_type"], dtype=np.int64),
        node_time_role_ids=np.asarray(cols["time_role"], dtype=np.int64),
        node_polarity=np.asarray(cols["polarity"], dtype=np.float32),
        event_type_to_id=event_type_to_id,
        time_role_to_id=time_role_to_id,
        event_node_mask=np.asarray(event_node_mask, dtype=bool),
        article_node_mask=np.asarray(article_node_mask, dtype=bool),
        fusion_node_mask=np.asarray(fusion_node_mask, dtype=bool),
        anchor_impact_by_ticker=anchor_impact_raw,
        events_by_ticker=events_by_ticker_raw,
        fusion_targets_by_event={
            k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in fusion_targets_raw.items()
        },
        fusion_members={
            k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in fusion_members_raw.items()
        },
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_relation_ids=edge_relation_ids,
        edge_base_weights=edge_base_weights,
        edge_available_ord=edge_available_ord,
        incoming_order=incoming_order,
        incoming_indptr=incoming_indptr,
        ticker_to_node_idx=ticker_to_node_idx,
    )


def load_forecast_tasks(
    path: str | Path,
    tickers: Sequence[str],
    horizon_trading_days: int,
    min_abs_label_return: float,
) -> List[ForecastTask]:
    """Load chronological binary-direction supervision rows."""

    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Forecast task file not found: {task_path}")
    requested = {normalize_ticker(t) for t in tickers}
    threshold = max(0.0, float(min_abs_label_return))
    tasks: List[ForecastTask] = []
    for row in iter_csv_rows(task_path):
        ticker = normalize_ticker(row.get("ticker", ""))
        if requested and ticker not in requested:
            continue
        cutoff = parse_date(row.get("cutoff_date", ""))
        horizon = int(_safe_float(row.get("horizon_trading_days", 1), 1.0))
        split = clean_text(row.get("split", "")).lower()
        if cutoff is None or split not in {"train", "val", "test"} \
                or horizon != int(horizon_trading_days):
            continue
        label_return = _safe_float(row.get("label_return", ""))
        tasks.append(
            ForecastTask(
                task_id=clean_text(row.get("task_id", "")),
                ticker=ticker,
                cutoff_date=cutoff,
                horizon_trading_days=horizon,
                label_return=label_return,
                label_direction=1 if label_return > threshold else 0,
                split=split,
            )
        )
    tasks.sort(key=lambda task: (task.cutoff_date, task.ticker, task.task_id))
    if not tasks:
        raise ValueError(f"No usable forecast tasks found in {task_path}")
    return tasks


def _incoming_edges(graph: GraphArtifacts, nodes: np.ndarray) -> np.ndarray:
    chunks = [
        graph.incoming_order[graph.incoming_indptr[int(n)] : graph.incoming_indptr[int(n) + 1]]
        for n in nodes
        if graph.incoming_indptr[int(n) + 1] > graph.incoming_indptr[int(n)]
    ]
    return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.int64)


def _active_nodes(
    graph: GraphArtifacts,
    current_ord: int,
    start_ord: int,
    ticker: str,
    event_ticker_scope: str,
) -> np.ndarray:
    """Nodes visible at the cutoff.

    ``event_ticker_scope`` decides which events may enter at all:
      ``all``              every event reachable within num_hops, including other
                           companies' events that merely share an article
      ``target``           only events explicitly about the target
      ``target_or_anchor`` those, plus events the extractor judged as impacting
                           the target through a directional anchor relation
    """

    active = np.ones((len(graph.node_ids),), dtype=bool)
    dated = graph.event_node_mask | graph.article_node_mask
    active[dated] = (
        (graph.node_available_ord[dated] >= start_ord)
        & (graph.node_available_ord[dated] <= current_ord)
    )

    if event_ticker_scope != "all" and ticker:
        keep = np.zeros((len(graph.node_ids),), dtype=bool)
        for node in graph.events_by_ticker.get(normalize_ticker(ticker), ()):
            keep[node] = True
        if event_ticker_scope == "target_or_anchor":
            for node in graph.anchor_impact_by_ticker.get(normalize_ticker(ticker), {}):
                keep[node] = True
        active[graph.event_node_mask & ~keep] = False

    # A fusion node activates only when an available member points at it.
    active[graph.fusion_node_mask] = False
    for event_node in np.flatnonzero(graph.event_node_mask & active):
        targets = graph.fusion_targets_by_event.get(int(event_node))
        if targets is not None:
            active[targets] = True
    return active


def _active_edge_mask(
    graph: GraphArtifacts,
    edge_ids: np.ndarray,
    active_nodes: np.ndarray,
    start_ord: int,
    current_ord: int,
) -> np.ndarray:
    if edge_ids.size == 0:
        return np.zeros((0,), dtype=bool)
    knowledge = graph.edge_available_ord[edge_ids]
    date_ok = (knowledge == UNKNOWN_ORDINAL) | (
        (knowledge >= start_ord) & (knowledge <= current_ord)
    )
    return date_ok & active_nodes[graph.edge_sources[edge_ids]] & active_nodes[graph.edge_targets[edge_ids]]


def _relative_time_features(
    graph: GraphArtifacts,
    node_indices: np.ndarray,
    active_nodes: np.ndarray,
    current_ord: int,
    window_days: int,
    hop_distance: np.ndarray,
    in_degree: np.ndarray,
    num_hops: int,
    ticker: str,
    semantic_tau_days: float,
) -> np.ndarray:
    """The 16 temporal/structural features per node. Column order is load-bearing:
    models.READOUT_RELEVANCE_COLUMNS indexes into it."""

    available = graph.node_available_ord[node_indices].astype(np.int64)
    start = graph.node_semantic_start_ord[node_indices].astype(np.int64)
    end = graph.node_semantic_end_ord[node_indices].astype(np.int64)
    known_available = available != UNKNOWN_ORDINAL
    known_start = start != UNKNOWN_ORDINAL
    known_end = end != UNKNOWN_ORDINAL

    features = np.zeros((len(node_indices), TEMPORAL_FEATURE_DIM), dtype=np.float32)
    age = np.zeros((len(node_indices),), dtype=np.float32)
    age[known_available] = np.maximum(0, current_ord - available[known_available])
    features[:, 0] = np.exp(-age / max(1.0, float(window_days)))          # recency decay
    features[:, 1] = np.clip(age / max(1.0, float(window_days)), 0.0, 10.0)  # normalised age
    features[known_start, 2] = np.clip((start[known_start] - current_ord) / 365.0, -5.0, 5.0)
    features[known_end, 3] = np.clip((end[known_end] - current_ord) / 365.0, -5.0, 5.0)
    both_semantic = known_start & known_end
    features[both_semantic, 4] = np.clip(
        (end[both_semantic] - start[both_semantic]) / 365.0, 0.0, 5.0
    )
    features[:, 5] = np.clip(graph.node_time_confidence[node_indices], 0.0, 1.0)
    features[:, 6] = np.clip(graph.node_event_confidence[node_indices], 0.0, 1.0)
    features[:, 7] = np.clip(graph.node_importance[node_indices] / 3.0, 0.0, 1.0)
    features[:, 8] = (known_start & (start > current_ord)).astype(np.float32)   # forward-looking
    features[:, 9] = known_available.astype(np.float32)
    # Structural context. Without these the encoder cannot distinguish news
    # *about* the target from an incidental co-mention.
    features[:, 10] = graph.node_polarity[node_indices]
    features[:, 11] = (hop_distance == 1).astype(np.float32)
    features[:, 12] = np.clip(hop_distance / max(1.0, float(num_hops)), 0.0, 1.0)
    features[:, 13] = np.clip(np.log1p(in_degree) / 5.0, 0.0, 2.0)

    impacts = graph.anchor_impact_by_ticker.get(normalize_ticker(ticker))
    if impacts:
        features[:, 14] = np.asarray(
            [impacts.get(int(n), 0.0) for n in node_indices], dtype=np.float32
        )

    # Semantic relevance: distance from the cutoff to the described period, zero
    # while the cutoff sits inside it. A closed quarter decays away; a deadline
    # grows more relevant as it approaches.
    distance = np.zeros((len(node_indices),), dtype=np.float64)
    both = known_start & known_end
    before = both & (current_ord < start)
    after = both & (current_ord > end)
    distance[before] = start[before] - current_ord
    distance[after] = current_ord - end[after]
    features[both, 15] = np.exp(-distance[both] / max(1.0, float(semantic_tau_days)))
    features[~both, 15] = 1.0   # undated events stay neutral rather than zeroed

    global_to_local = np.full((len(graph.node_ids),), -1, dtype=np.int64)
    global_to_local[node_indices] = np.arange(len(node_indices), dtype=np.int64)
    for fusion_global in node_indices[graph.fusion_node_mask[node_indices]]:
        members = graph.fusion_members.get(int(fusion_global))
        if members is None:
            continue
        member_local = global_to_local[members[active_nodes[members]]]
        member_local = member_local[member_local >= 0]
        fusion_local = global_to_local[int(fusion_global)]
        if member_local.size and fusion_local >= 0:
            features[fusion_local] = features[member_local].mean(axis=0)
    return features


def build_window_subgraph(
    graph: GraphArtifacts,
    ticker: str,
    current_date: date,
    window_days: int,
    decay_lambda: float,
    num_hops: int,
    semantic_tau_days: float = 30.0,
    event_ticker_scope: str = "target_or_anchor",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """The exact active inbound neighbourhood for one ticker/cutoff.

    A "hop" limits which nodes are invited into the subgraph, not how far
    information travels once inside: message passing runs over the whole
    subgraph, so every node talks to every neighbour it has.
    """

    target_global = graph.ticker_to_node_idx[normalize_ticker(ticker)]
    current_ord = current_date.toordinal()
    start_ord = (current_date - timedelta(days=int(window_days))).toordinal()
    active_nodes = _active_nodes(graph, current_ord, start_ord, ticker, event_ticker_scope)

    selected = np.zeros((len(graph.node_ids),), dtype=bool)
    selected[target_global] = True
    hop_of = np.zeros((len(graph.node_ids),), dtype=np.int32)
    frontier = np.asarray([target_global], dtype=np.int64)
    for hop in range(1, max(1, int(num_hops)) + 1):
        candidates = _incoming_edges(graph, frontier)
        valid = _active_edge_mask(graph, candidates, active_nodes, start_ord, current_ord)
        sources = np.unique(graph.edge_sources[candidates[valid]])
        new_nodes = sources[~selected[sources]]
        if new_nodes.size == 0:
            break
        selected[new_nodes] = True
        hop_of[new_nodes] = hop
        frontier = new_nodes

    node_indices = np.flatnonzero(selected).astype(np.int64)
    candidate_edges = _incoming_edges(graph, node_indices)
    candidate_edges = candidate_edges[
        _active_edge_mask(graph, candidate_edges, active_nodes, start_ord, current_ord)
    ]
    candidate_edges = candidate_edges[selected[graph.edge_sources[candidate_edges]]]

    global_to_local = np.full((len(graph.node_ids),), -1, dtype=np.int64)
    global_to_local[node_indices] = np.arange(len(node_indices), dtype=np.int64)
    local_targets = global_to_local[graph.edge_targets[candidate_edges]]
    edge_index = np.vstack(
        [global_to_local[graph.edge_sources[candidate_edges]], local_targets]
    ).astype(np.int64)
    in_degree = np.bincount(local_targets, minlength=len(node_indices)).astype(np.float32)

    knowledge = graph.edge_available_ord[candidate_edges].astype(np.int64)
    age_days = np.zeros((len(candidate_edges),), dtype=np.float32)
    known = knowledge != UNKNOWN_ORDINAL
    age_days[known] = np.maximum(0, current_ord - knowledge[known])
    # Yesterday's news is worth exp(-0.10) of today's; ten-day-old news ~0.37.
    weights = graph.edge_base_weights[candidate_edges] * np.exp(
        -float(decay_lambda) * age_days
    ).astype(np.float32)

    temporal_features = _relative_time_features(
        graph, node_indices, active_nodes, current_ord, int(window_days),
        hop_of[node_indices].astype(np.float32), in_degree, int(num_hops),
        ticker, float(semantic_tau_days),
    )
    return (
        node_indices,
        graph.node_type_ids[node_indices],
        temporal_features,
        graph.fusion_node_mask[node_indices],
        edge_index,
        graph.edge_relation_ids[candidate_edges],
        weights,
        int(global_to_local[target_global]),
    )


def build_temporal_examples(
    graph: GraphArtifacts,
    tasks: Sequence[ForecastTask],
    window_days: int,
    decay_lambda: float,
    num_hops: int,
    semantic_tau_days: float = 30.0,
    event_ticker_scope: str = "target_or_anchor",
) -> List[TemporalExample]:
    examples: List[TemporalExample] = []
    total = len(tasks)
    for task_idx, task in enumerate(tasks, start=1):
        (
            node_indices, node_type_ids, temporal_features, fusion_mask,
            edge_index, relation_ids, edge_weights, target_local,
        ) = build_window_subgraph(
            graph, task.ticker, task.cutoff_date, window_days, decay_lambda,
            num_hops, semantic_tau_days, event_ticker_scope,
        )
        examples.append(
            TemporalExample(
                task_id=task.task_id,
                ticker=task.ticker,
                current_date=task.cutoff_date,
                split=task.split,
                label_return=task.label_return,
                label=task.label_direction,
                node_indices=node_indices,
                node_type_ids=node_type_ids,
                node_event_type_ids=graph.node_event_type_ids[node_indices],
                node_time_role_ids=graph.node_time_role_ids[node_indices],
                event_node_mask=graph.event_node_mask[node_indices],
                temporal_features=temporal_features,
                fusion_node_mask=fusion_mask,
                target_node_idx=target_local,
                edge_index=edge_index,
                edge_relation_ids=relation_ids.astype(np.int64),
                edge_weights=edge_weights.astype(np.float32),
            )
        )
        if task_idx % 100 == 0 or task_idx == total:
            print(f"built_temporal_examples={task_idx}/{total}", flush=True)
    return examples


def split_examples(examples: Sequence[TemporalExample], split: str) -> List[TemporalExample]:
    return [example for example in examples if example.split == split]


def summarize_examples(name: str, examples: Sequence[TemporalExample]) -> Dict[str, object]:
    dates = [e.current_date for e in examples]
    positives = sum(e.label == 1 for e in examples)
    node_counts = [len(e.node_indices) for e in examples]
    edge_counts = [int(e.edge_index.shape[1]) for e in examples]
    return {
        "name": name,
        "count": len(examples),
        "start_date": str(min(dates)) if dates else "",
        "end_date": str(max(dates)) if dates else "",
        "ticker_count": len({e.ticker for e in examples}),
        "positive": positives,
        "positive_rate": positives / len(examples) if examples else float("nan"),
        "mean_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
        "max_nodes": max(node_counts) if node_counts else 0,
        "mean_edges": float(np.mean(edge_counts)) if edge_counts else 0.0,
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph_dir", default="ekg_final")
    parser.add_argument("--forecast_tasks", default="gnn_data/forecast_task.csv")
    parser.add_argument("--output_dir", default="gnn_outputs")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--window_days", type=int, default=1)
    parser.add_argument("--horizon_trading_days", type=int, default=0)
    parser.add_argument("--decay_lambda", type=float, default=0.10)
    parser.add_argument("--num_hops", type=int, default=3)
    parser.add_argument(
        "--event_ticker_scope",
        choices=["all", "target", "target_or_anchor"],
        default="target_or_anchor",
        help=(
            "Which events may enter a subgraph. 'all' lets other companies' events "
            "in through shared articles, which measured worse than filtering."
        ),
    )
    parser.add_argument(
        "--semantic_tau_days",
        type=float,
        default=30.0,
        help="Decay constant for semantic relevance: exp(-distance_to_period / tau).",
    )
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--min_abs_label_return", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")


def main() -> None:
    """Validate that subgraphs build, without training anything."""

    parser = argparse.ArgumentParser(description="Validate temporal GNN examples.")
    add_common_args(parser)
    args = parser.parse_args()
    tasks = load_forecast_tasks(
        args.forecast_tasks, parse_tickers(args.tickers),
        args.horizon_trading_days, args.min_abs_label_return,
    )
    tickers = sorted({task.ticker for task in tasks})
    graph = load_graph_artifacts(args.graph_dir, tickers)
    examples = build_temporal_examples(
        graph, tasks, args.window_days, args.decay_lambda, args.num_hops,
        args.semantic_tau_days, args.event_ticker_scope,
    )
    summary = {
        "graph_dir": str(Path(args.graph_dir)),
        "forecast_tasks": str(Path(args.forecast_tasks)),
        "ticker_count": len(tickers),
        "event_ticker_scope": args.event_ticker_scope,
        "node_count": int(graph.node_features.shape[0]),
        "feature_dim": int(graph.node_features.shape[1]),
        "relation_count": len(graph.relation_to_id),
        "splits": [summarize_examples(s, split_examples(examples, s)) for s in ("train", "val", "test")],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
