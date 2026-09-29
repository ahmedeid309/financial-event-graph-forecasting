from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from financial_ekg.config import TEXT_AMBIGUOUS_ANCHOR_TICKERS
from financial_ekg.graph.embeddings import get_text_embeddings
from financial_ekg.utils.anchors import (
    build_anchor_lookup,
    load_anchor_company_context,
    resolve_anchor_ticker,
)
from financial_ekg.utils.dates import parse_date
from financial_ekg.utils.serialization import json_dumps, json_loads_maybe
from financial_ekg.utils.text import (
    clean_text,
    entity_resolution_keys,
    normalize_entity_name,
    normalize_event_type_value,
    normalize_ticker,
    sha1_short,
)


@dataclass(frozen=True)
class FusionConfig:
    """Python-side defaults for EventRAG-style fusion."""

    entity_fusion_threshold: float = 0.92
    event_fusion_threshold: float = 0.90
    fusion_candidate_neighbors: int = 20
    fusion_max_event_days: int = 7
    fusion_verify_mode: str = "borderline"
    fusion_llm_max_pairs: int = 20000


@dataclass
class FusionResult:
    """Fusion artifacts and lookup maps used by graph construction."""

    config: FusionConfig
    event_clusters: pd.DataFrame
    event_cluster_members: pd.DataFrame
    entity_clusters: pd.DataFrame
    entity_cluster_members: pd.DataFrame
    report: Dict[str, Any]
    event_cluster_by_event_id: Dict[str, str]
    event_cluster_by_id: Dict[str, Dict[str, Any]]
    entity_cluster_by_lookup_key: Dict[str, str]
    entity_cluster_by_id: Dict[str, Dict[str, Any]]
    primary_entity_cluster_by_event_id: Dict[str, str]

class UnionFind:
    def __init__(self, items: Iterable[Any]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: Any) -> Any:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Any, right: Any) -> bool:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot == rroot:
            return False
        self.parent[rroot] = lroot
        return True

def run_fusion(
    event_df: pd.DataFrame,
    event_embeddings: np.ndarray,
    args: Any,
    out_dir: Path,
) -> FusionResult:
    """Build entity and event fusion clusters and write fusion artifacts."""
    config = FusionConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_context = load_anchor_company_context(
        getattr(args, "anchor_tickers", ""),
        getattr(args, "anchor_aliases_json", ""),
    )
    anchor_lookup = build_anchor_lookup(anchor_context)

    entity_result = build_entity_clusters(event_df, args, config, anchor_context, anchor_lookup)
    event_result = build_event_clusters(
        event_df,
        event_embeddings,
        args,
        config,
        entity_result["primary_entity_cluster_by_event_id"],
    )

    event_clusters = event_result["event_clusters"]
    event_members = event_result["event_cluster_members"]
    entity_clusters = entity_result["entity_clusters"]
    entity_members = entity_result["entity_cluster_members"]

    event_clusters.to_csv(out_dir / "event_clusters.csv", index=False)
    event_members.to_csv(out_dir / "event_cluster_members.csv", index=False)
    entity_clusters.to_csv(out_dir / "entity_clusters.csv", index=False)
    entity_members.to_csv(out_dir / "entity_cluster_members.csv", index=False)

    report = {
        "config": asdict(config),
        "enabled": True,
        "event_count": int(len(event_df)),
        "event_cluster_count": int(len(event_clusters)),
        "event_multi_member_cluster_count": int((event_clusters.get("cluster_size", pd.Series(dtype=int)).astype(int) > 1).sum()),
        "event_fused_member_count": int((event_members.get("cluster_size", pd.Series(dtype=int)).astype(int) > 1).sum()),
        "entity_mention_count": int(len(entity_members)),
        "entity_cluster_count": int(len(entity_clusters)),
        "entity_multi_member_cluster_count": int((entity_clusters.get("member_mention_count", pd.Series(dtype=int)).astype(int) > 1).sum()),
        "event_pairs_considered": int(event_result["pairs_considered"]),
        "event_pairs_accepted": int(event_result["pairs_accepted"]),
        "entity_pairs_considered": int(entity_result["pairs_considered"]),
        "entity_pairs_accepted": int(entity_result["pairs_accepted"]),
        "llm_verifier": "not_configured_rule_fallback",
        "notes": [
            "Article/source title is intentionally excluded from event fusion candidate text.",
            "Original events and entities are preserved; clusters add canonical structure.",
        ],
    }
    with (out_dir / "fusion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return FusionResult(
        config=config,
        event_clusters=event_clusters,
        event_cluster_members=event_members,
        entity_clusters=entity_clusters,
        entity_cluster_members=entity_members,
        report=report,
        event_cluster_by_event_id=event_result["event_cluster_by_event_id"],
        event_cluster_by_id=event_clusters.set_index("event_cluster_id", drop=False).to_dict(orient="index") if not event_clusters.empty else {},
        entity_cluster_by_lookup_key=entity_result["entity_cluster_by_lookup_key"],
        entity_cluster_by_id=entity_clusters.set_index("entity_cluster_id", drop=False).to_dict(orient="index") if not entity_clusters.empty else {},
        primary_entity_cluster_by_event_id=entity_result["primary_entity_cluster_by_event_id"],
    )


def build_entity_clusters(
    event_df: pd.DataFrame,
    args: Any,
    config: FusionConfig,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
) -> Dict[str, Any]:
    mentions = collect_entity_mentions(event_df, anchor_context, anchor_lookup)
    uf = UnionFind(range(len(mentions)))
    key_to_indices: Dict[str, List[int]] = defaultdict(list)
    ticker_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, mention in enumerate(mentions):
        for key in mention["lookup_keys"]:
            key_to_indices[key].append(idx)
        if mention["resolved_ticker"]:
            ticker_to_indices[mention["resolved_ticker"]].append(idx)

    for indices in key_to_indices.values():
        _union_many(uf, indices)
    for indices in ticker_to_indices.values():
        _union_many(uf, indices)

    pairs_considered = 0
    pairs_accepted = 0
    eligible = [idx for idx, mention in enumerate(mentions) if entity_embedding_eligible(mention)]
    if len(eligible) > 1:
        texts = [mentions[idx]["embedding_text"] for idx in eligible]
        emb = get_text_embeddings(
            texts,
            model_name=getattr(args, "embedding_model", ""),
            device=getattr(args, "device", "cpu"),
            batch_size=int(getattr(args, "embedding_batch_size", 64) or 64),
            fallback_dim=max(int(getattr(args, "feature_embedding_dim", 64) or 64), 64),
        )
        emb = normalize_rows(emb)
        if not np.allclose(emb, 0.0):
            from sklearn.neighbors import NearestNeighbors

            n_neighbors = min(config.fusion_candidate_neighbors + 1, len(eligible))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
            nn.fit(emb)
            distances, neighbors = nn.kneighbors(emb)
            for local_i, neighs in enumerate(neighbors):
                i = eligible[local_i]
                for pos, local_j in enumerate(neighs):
                    j = eligible[int(local_j)]
                    if j <= i:
                        continue
                    sim = 1.0 - float(distances[local_i, pos])
                    if sim < config.entity_fusion_threshold:
                        continue
                    pairs_considered += 1
                    if not entity_mentions_compatible(mentions[i], mentions[j]):
                        continue
                    if uf.union(i, j):
                        pairs_accepted += 1

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(mentions)):
        components[int(uf.find(idx))].append(idx)

    cluster_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []
    lookup_to_cluster: Dict[str, str] = {}
    primary_by_event: Dict[str, str] = {}

    for indices in sorted(components.values(), key=lambda xs: min(xs)):
        member_mentions = [mentions[idx] for idx in indices]
        rep_name, rep_ticker = representative_entity(member_mentions, anchor_context)
        seed_values = sorted({m["resolved_ticker"] or m["normalized_text"].lower() or m["entity_text"].lower() for m in member_mentions if m["entity_text"]})
        seed = "|".join([rep_ticker, rep_name.lower(), *seed_values[:12]])
        cluster_id = "ENTC_" + sha1_short(seed, 18)
        source_events = sorted({m["event_id"] for m in member_mentions if m["event_id"]})
        texts = sorted({m["entity_text"] for m in member_mentions if m["entity_text"]})
        cluster_rows.append(
            {
                "entity_cluster_id": cluster_id,
                "representative_name": rep_name,
                "ticker": rep_ticker,
                "member_mention_count": len(member_mentions),
                "source_event_count": len(source_events),
                "member_texts_json": json_dumps(texts[:100]),
                "source_event_ids_json": json_dumps(source_events[:100]),
            }
        )
        representative_seen = False
        for mention in member_mentions:
            is_rep = not representative_seen and (
                (rep_ticker and mention["resolved_ticker"] == rep_ticker)
                or normalize_entity_name(mention["entity_text"]).lower() == normalize_entity_name(rep_name).lower()
            )
            representative_seen = representative_seen or is_rep
            member_rows.append(
                {
                    "entity_cluster_id": cluster_id,
                    "mention_id": mention["mention_id"],
                    "event_id": mention["event_id"],
                    "entity_text": mention["entity_text"],
                    "normalized_text": mention["normalized_text"],
                    "ticker": mention["resolved_ticker"],
                    "source": mention["source"],
                    "is_representative": bool(is_rep),
                    "cluster_size": len(member_mentions),
                }
            )
            for key in mention["lookup_keys"]:
                lookup_to_cluster.setdefault(key, cluster_id)
            if mention["source"] == "main_company" and mention["event_id"]:
                primary_by_event.setdefault(mention["event_id"], cluster_id)

    return {
        "entity_clusters": pd.DataFrame(cluster_rows),
        "entity_cluster_members": pd.DataFrame(member_rows),
        "entity_cluster_by_lookup_key": lookup_to_cluster,
        "primary_entity_cluster_by_event_id": primary_by_event,
        "pairs_considered": pairs_considered,
        "pairs_accepted": pairs_accepted,
    }


def collect_entity_mentions(
    event_df: pd.DataFrame,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    seen = set()

    def add(event_id: str, text: Any, ticker: Any, source: str) -> None:
        entity_text = clean_text(text)
        if not entity_text:
            return
        ticker_norm = normalize_ticker(ticker)
        normalized = normalize_entity_name(entity_text)
        if not normalized and not ticker_norm:
            return
        resolved_ticker = resolve_entity_ticker(entity_text, ticker_norm, anchor_context, anchor_lookup)
        lookup_keys = entity_lookup_keys(entity_text, resolved_ticker)
        if not lookup_keys:
            return
        mention_key = (event_id, entity_text.lower(), resolved_ticker, source)
        if mention_key in seen:
            return
        seen.add(mention_key)
        mentions.append(
            {
                "mention_id": "ENTM_" + sha1_short("|".join(mention_key), 18),
                "event_id": event_id,
                "entity_text": entity_text,
                "normalized_text": normalized,
                "resolved_ticker": resolved_ticker,
                "source": source,
                "lookup_keys": lookup_keys,
                "embedding_text": " | ".join(part for part in [resolved_ticker, normalized or entity_text] if part),
            }
        )

    for _, row in event_df.iterrows():
        event_id = clean_text(row.get("event_id", ""))
        add(event_id, row.get("main_company", ""), row.get("ticker", ""), "main_company")
        for item in json_loads_maybe(row.get("participants_json", ""), []):
            add(event_id, item, "", "participant")
        for item in json_loads_maybe(row.get("mentioned_entities_json", ""), []):
            add(event_id, item, "", "mentioned_entity")
        for item in json_loads_maybe(row.get("impacted_anchor_companies_json", ""), []):
            if isinstance(item, dict):
                add(
                    event_id,
                    item.get("company", "") or item.get("raw_mention", "") or item.get("ticker", ""),
                    item.get("ticker", ""),
                    "anchor_impact",
                )
    return mentions


def resolve_entity_ticker(
    entity_text: str,
    ticker: str,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
) -> str:
    anchor_by_name = resolve_anchor_ticker(entity_text, "", anchor_context, anchor_lookup)
    if anchor_by_name:
        return anchor_by_name
    ticker_norm = normalize_ticker(ticker)
    if not ticker_norm:
        return ""
    if ticker_norm in TEXT_AMBIGUOUS_ANCHOR_TICKERS:
        return ticker_norm if anchor_by_name == ticker_norm else ""
    return ticker_norm


def entity_lookup_keys(entity_text: str, ticker: str = "") -> List[str]:
    keys: List[str] = []
    ticker_norm = normalize_ticker(ticker)
    if ticker_norm:
        keys.append(f"ticker:{ticker_norm}")
    for key in entity_resolution_keys(entity_text, ticker_norm):
        if key and len(key) > 1:
            keys.append(f"name:{key}")
    return list(dict.fromkeys(keys))


def entity_embedding_eligible(mention: Dict[str, Any]) -> bool:
    if mention.get("resolved_ticker"):
        return False
    normalized = clean_text(mention.get("normalized_text", ""))
    if len(normalized) < 4:
        return False
    if re.fullmatch(r"[A-Z0-9.\-]{1,4}", clean_text(mention.get("entity_text", ""))):
        return False
    return True


def entity_mentions_compatible(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    lticker = clean_text(left.get("resolved_ticker", ""))
    rticker = clean_text(right.get("resolved_ticker", ""))
    if lticker and rticker and lticker != rticker:
        return False
    return True


def representative_entity(
    mentions: Sequence[Dict[str, Any]],
    anchor_context: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    tickers = [m["resolved_ticker"] for m in mentions if m.get("resolved_ticker")]
    if tickers:
        ticker = Counter(tickers).most_common(1)[0][0]
        if ticker in anchor_context:
            return clean_text(anchor_context[ticker].get("company", "")) or ticker, ticker
        names = [m["normalized_text"] or m["entity_text"] for m in mentions if m.get("resolved_ticker") == ticker]
        return representative_text(names) or ticker, ticker
    names = [m["normalized_text"] or m["entity_text"] for m in mentions]
    return representative_text(names), ""


def representative_text(values: Sequence[str]) -> str:
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    max_count = max(counts.values())
    candidates = [value for value, count in counts.items() if count == max_count]
    return sorted(candidates, key=lambda value: (-len(value), value.lower()))[0]


def build_event_clusters(
    event_df: pd.DataFrame,
    event_embeddings: np.ndarray,
    args: Any,
    config: FusionConfig,
    primary_entity_cluster_by_event_id: Dict[str, str],
) -> Dict[str, Any]:
    rows = [event_record(idx, row, primary_entity_cluster_by_event_id) for idx, row in event_df.reset_index(drop=True).iterrows()]
    uf = UnionFind(range(len(rows)))
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if not row["event_id"]:
            continue
        group_key = row["primary_entity_cluster_id"] or f"ticker:{row['ticker']}" or f"company:{row['main_company_key']}"
        if not group_key:
            continue
        groups[(group_key, row["event_type"])].append(idx)

    pairs_considered = 0
    pairs_accepted = 0
    candidate_texts = [event_fusion_text(row) for row in rows]
    if len(rows) > 1:
        emb = get_text_embeddings(
            candidate_texts,
            model_name=getattr(args, "embedding_model", ""),
            device=getattr(args, "device", "cpu"),
            batch_size=int(getattr(args, "embedding_batch_size", 64) or 64),
            fallback_dim=max(int(getattr(args, "feature_embedding_dim", 64) or 64), 64),
        )
        emb = normalize_rows(emb)
    else:
        emb = np.zeros((len(rows), max(int(getattr(args, "feature_embedding_dim", 64) or 64), 64)), dtype=np.float32)

    description_emb, description_has_text = embed_optional_event_view([row["description_text"] for row in rows], args)
    metric_emb, metric_has_text = embed_optional_event_view([row["metric_text"] for row in rows], args)
    entity_emb, entity_has_text = embed_optional_event_view([row["mentioned_entity_text"] for row in rows], args)

    if len(rows) > 1 and not np.allclose(emb, 0.0):
        from sklearn.neighbors import NearestNeighbors

        for indices in groups.values():
            if len(indices) < 2:
                continue
            group_emb = emb[indices]
            n_neighbors = min(config.fusion_candidate_neighbors + 1, len(indices))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
            nn.fit(group_emb)
            distances, neighbors = nn.kneighbors(group_emb)
            candidate_pairs: List[Tuple[float, int, int]] = []
            for local_i, neighs in enumerate(neighbors):
                i = indices[local_i]
                for pos, local_j in enumerate(neighs):
                    j = indices[int(local_j)]
                    if j <= i:
                        continue
                    sim = 1.0 - float(distances[local_i, pos])
                    if sim >= config.event_fusion_threshold:
                        candidate_pairs.append((sim, i, j))
            for sim, i, j in sorted(candidate_pairs, reverse=True):
                pairs_considered += 1
                if not events_compatible(rows[i], rows[j], config.fusion_max_event_days):
                    continue
                if uf.union(i, j):
                    pairs_accepted += 1

    # Exact duplicate fallback catches deterministic repeats even when embeddings
    # are disabled or unavailable.
    exact_groups: Dict[Tuple[str, str, str, str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        exact_groups[
            (
                row["primary_entity_cluster_id"] or row["ticker"] or row["main_company_key"],
                row["event_type"],
                normalize_event_text(row["event_description"]),
                "|".join(row["metric_values"]),
                row["event_date_start"],
            )
        ].append(idx)
    for indices in exact_groups.values():
        if len(indices) < 2:
            continue
        _union_many(uf, indices)

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(rows)):
        components[int(uf.find(idx))].append(idx)

    cluster_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []
    cluster_by_event_id: Dict[str, str] = {}
    for indices in sorted(components.values(), key=lambda xs: min(xs)):
        member_records = [rows[idx] for idx in indices]
        rep = representative_event(member_records)
        member_ids = sorted([item["event_id"] for item in member_records if item["event_id"]])
        cluster_id = "EVTC_" + sha1_short("|".join(member_ids), 18)
        cluster_size = len(member_records)
        cluster_rows.append(
            {
                "event_cluster_id": cluster_id,
                "representative_event_id": rep["event_id"],
                "cluster_size": cluster_size,
                "ticker": rep["ticker"],
                "event_type": rep["event_type"],
                "main_company": rep["main_company"],
                "event_date_start": rep["event_date_start"],
                "event_date_end": rep["event_date_end"],
                "canonical_description": rep["event_description"],
                "member_event_ids_json": json_dumps(member_ids),
            }
        )
        rep_idx = rep["row_idx"]
        for record in member_records:
            sim = embedding_similarity(emb, rep_idx, record["row_idx"], default=0.0)
            member_rows.append(
                {
                    "event_cluster_id": cluster_id,
                    "event_id": record["event_id"],
                    "is_representative": record["event_id"] == rep["event_id"],
                    "similarity_to_representative": round(sim, 6),
                    "whole_similarity_to_representative": round(sim, 6),
                    "description_similarity_to_representative": optional_embedding_similarity(
                        description_emb,
                        description_has_text,
                        rep_idx,
                        record["row_idx"],
                    ),
                    "metric_similarity_to_representative": optional_embedding_similarity(
                        metric_emb,
                        metric_has_text,
                        rep_idx,
                        record["row_idx"],
                    ),
                    "mentioned_entity_similarity_to_representative": optional_embedding_similarity(
                        entity_emb,
                        entity_has_text,
                        rep_idx,
                        record["row_idx"],
                    ),
                    "merge_reason": "same_cluster" if cluster_size > 1 else "singleton",
                    "cluster_size": cluster_size,
                }
            )
            if record["event_id"]:
                cluster_by_event_id[record["event_id"]] = cluster_id

    return {
        "event_clusters": pd.DataFrame(cluster_rows),
        "event_cluster_members": pd.DataFrame(member_rows),
        "event_cluster_by_event_id": cluster_by_event_id,
        "pairs_considered": pairs_considered,
        "pairs_accepted": pairs_accepted,
    }


def event_record(
    idx: int,
    row: pd.Series,
    primary_entity_cluster_by_event_id: Dict[str, str],
) -> Dict[str, Any]:
    event_id = clean_text(row.get("event_id", ""))
    metrics = [clean_text(item) for item in json_loads_maybe(row.get("financial_metric_mentions_json", ""), []) if clean_text(item)]
    mentioned_entities = event_mentioned_entity_terms(row)
    start = parse_date(row.get("event_date_start", "")) or parse_date(row.get("date", ""))
    end = parse_date(row.get("event_date_end", "")) or start
    text_for_direction = " ".join(
        [
            clean_text(row.get("event_trigger", "")),
            clean_text(row.get("event_description", "")),
            " ".join(metrics),
        ]
    )
    return {
        "row_idx": idx,
        "event_id": event_id,
        "article_id": clean_text(row.get("article_id", "")),
        "ticker": normalize_ticker(row.get("ticker", "")),
        "event_type": normalize_event_type_value(row.get("event_type", "")),
        "event_trigger": clean_text(row.get("event_trigger", "")),
        "event_description": clean_text(row.get("event_description", "")),
        "description_text": clean_text(row.get("event_description", "")),
        "main_company": clean_text(row.get("main_company", "")),
        "main_company_key": normalize_entity_name(row.get("main_company", "")).lower(),
        "metrics": metrics,
        "metric_text": " | ".join(metrics),
        "mentioned_entity_text": " | ".join(mentioned_entities),
        "metric_values": metric_values(metrics + [row.get("event_description", "")]),
        "temporal_expression": clean_text(row.get("temporal_expression", "")),
        "event_date_start": start,
        "event_date_end": end,
        "event_date_start_ord": date_ordinal(start),
        "event_date_end_ord": date_ordinal(end),
        "importance_score": safe_float(row.get("importance_score", 1), 1.0),
        "confidence": safe_float(row.get("confidence", 0.0), 0.0),
        "direction_signature": direction_signature(text_for_direction),
        "primary_entity_cluster_id": primary_entity_cluster_by_event_id.get(event_id, ""),
    }


def event_fusion_text(record: Dict[str, Any]) -> str:
    return clean_text(record.get("event_description", ""))


def event_mentioned_entity_terms(row: pd.Series) -> List[str]:
    terms: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("company", "entity", "name", "raw_mention", "ticker"):
                text = clean_text(value.get(key, ""))
                if text:
                    terms.append(text)
            return
        text = clean_text(value)
        if text:
            terms.append(text)

    for column in ("participants_json", "mentioned_entities_json", "impacted_anchor_companies_json"):
        for item in json_loads_maybe(row.get(column, ""), []):
            add(item)
    return list(dict.fromkeys(terms))


def embed_optional_event_view(texts: Sequence[str], args: Any) -> Tuple[np.ndarray, List[bool]]:
    cleaned = [clean_text(text) for text in texts]
    has_text = [bool(text) for text in cleaned]
    nonempty_indices = [idx for idx, text in enumerate(cleaned) if text]
    if not nonempty_indices:
        return np.zeros((len(cleaned), 0), dtype=np.float32), has_text

    view_emb = get_text_embeddings(
        [cleaned[idx] for idx in nonempty_indices],
        model_name=getattr(args, "embedding_model", ""),
        device=getattr(args, "device", "cpu"),
        batch_size=int(getattr(args, "embedding_batch_size", 64) or 64),
        fallback_dim=max(int(getattr(args, "feature_embedding_dim", 64) or 64), 64),
    )
    view_emb = normalize_rows(view_emb)
    out = np.zeros((len(cleaned), view_emb.shape[1]), dtype=np.float32)
    for local_idx, row_idx in enumerate(nonempty_indices):
        out[row_idx] = view_emb[local_idx]
    return out, has_text


def embedding_similarity(emb: np.ndarray, left_idx: int, right_idx: int, default: float = 0.0) -> float:
    if emb.ndim != 2 or emb.shape[1] == 0:
        return float(default)
    if left_idx >= emb.shape[0] or right_idx >= emb.shape[0]:
        return float(default)
    return float(np.dot(emb[left_idx], emb[right_idx]))


def optional_embedding_similarity(emb: np.ndarray, has_text: Sequence[bool], left_idx: int, right_idx: int) -> Any:
    if left_idx >= len(has_text) or right_idx >= len(has_text):
        return ""
    if not has_text[left_idx] or not has_text[right_idx]:
        return ""
    return round(embedding_similarity(emb, left_idx, right_idx, default=0.0), 6)


def events_compatible(left: Dict[str, Any], right: Dict[str, Any], max_days: int) -> bool:
    if left["event_type"] != right["event_type"]:
        return False
    if left["ticker"] and right["ticker"] and left["ticker"] != right["ticker"]:
        return False
    if left["primary_entity_cluster_id"] and right["primary_entity_cluster_id"] and left["primary_entity_cluster_id"] != right["primary_entity_cluster_id"]:
        return False
    if left["event_date_start_ord"] is not None and right["event_date_start_ord"] is not None:
        if abs(left["event_date_start_ord"] - right["event_date_start_ord"]) > max_days:
            return False
    if left["event_date_end_ord"] is not None and right["event_date_end_ord"] is not None:
        if abs(left["event_date_end_ord"] - right["event_date_end_ord"]) > max_days:
            return False
    if left["metric_values"] and right["metric_values"] and set(left["metric_values"]).isdisjoint(set(right["metric_values"])):
        return False
    if {left["direction_signature"], right["direction_signature"]} == {"positive", "negative"}:
        return False
    return True


def representative_event(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        records,
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            -float(item.get("importance_score", 1.0)),
            -len(clean_text(item.get("event_description", ""))),
            clean_text(item.get("event_id", "")),
        ),
    )[0]


def metric_values(values: Sequence[Any]) -> List[str]:
    text = " ".join(clean_text(value) for value in values)
    found = []
    for match in re.finditer(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|trillion|x|times|shares?)?", text, flags=re.I):
        value = re.sub(r"\s+", "", match.group(0).lower().replace(",", ""))
        value = value.replace("percent", "%")
        if value:
            found.append(value)
    return sorted(set(found))


def direction_signature(text: Any) -> str:
    lower = clean_text(text).lower()
    positive = bool(
        re.search(
            r"\b(up|rose|risen|rise|gain|gained|gains|growth|grew|soared|jumped|climbed|increased|higher|bullish|outperform)\b",
            lower,
        )
    )
    negative = bool(
        re.search(
            r"\b(down|fell|fall|fallen|decline|declined|loss|lost|off|dropped|plunged|decreased|lower|bearish|underperform)\b",
            lower,
        )
    )
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "neutral"


def normalize_event_text(value: Any) -> str:
    text = clean_text(value).lower().strip().rstrip(".")
    text = re.sub(r"[^a-z0-9%$., ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def date_ordinal(value: Any) -> Optional[int]:
    parsed = parse_date(value)
    if not parsed:
        return None
    try:
        return int(pd.to_datetime(parsed).date().toordinal())
    except Exception:
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape((arr.shape[0], -1))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def _union_many(uf: UnionFind, indices: Sequence[int]) -> None:
    if len(indices) < 2:
        return
    first = indices[0]
    for idx in indices[1:]:
        uf.union(first, idx)
