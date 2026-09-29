from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

from financial_ekg.graph.embeddings import add_event_embeddings, build_and_save_node_features
from financial_ekg.graph.fusion import FusionResult, run_fusion
from financial_ekg.graph.nodes_edges import (
    add_edge,
    add_node,
    add_semantic_entity_company_resolution,
)
from financial_ekg.graph.tensor_export import export_heterogeneous_graph_tensors
from financial_ekg.io.articles import save_articles_snapshot
from financial_ekg.io.run_config import save_config
from financial_ekg.models import GraphEdge, GraphNode
from financial_ekg.utils.anchors import (
    anchor_impact_direction_score,
    anchor_impact_edge_type,
    build_anchor_lookup,
    derive_anchor_impacts_for_event,
    event_company_edge_type,
    load_anchor_company_context,
    normalize_anchor_impact_direction,
    normalize_anchor_impact_type,
    reverse_edge_type,
    resolve_anchor_ticker,
)
from financial_ekg.utils.dates import parse_date
from financial_ekg.utils.serialization import json_loads_maybe
from financial_ekg.utils.text import clean_text, entity_resolution_keys, normalize_entity_name, normalize_ticker, sha1_short

def build_graph_tables(
    articles: pd.DataFrame,
    event_df: pd.DataFrame,
    event_embeddings: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    fusion_result: Optional[FusionResult] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build node/edge CSV files and GraphML from articles and events.

    Args:
        articles: Article dataframe.
        event_df: Event dataframe.
        event_embeddings: Event embedding matrix aligned to `event_df`.
        args: Parsed CLI arguments.
        out_dir: Output directory.
        fusion_result: Optional EventRAG-style fusion artifacts.

    Returns:
        Tuple of node dataframe and edge dataframe.
    """
    nodes: Dict[str, GraphNode] = {}
    edges: Dict[Tuple[str, str, str, str, str], GraphEdge] = {}
    anchor_context = load_anchor_company_context(
        getattr(args, "anchor_tickers", ""),
        getattr(args, "anchor_aliases_json", ""),
    )
    anchor_lookup = build_anchor_lookup(anchor_context)
    unresolved_entities: List[Dict[str, str]] = []

    # Article/date/ticker nodes and provenance edges.
    article_meta = articles.set_index("article_id", drop=False).to_dict(orient="index")
    for _, row in tqdm(articles.iterrows(), total=len(articles), desc="Adding article nodes"):
        article_id = row["article_id"]
        ticker = row["_ticker"]
        date = row["_date"]
        article_node = f"article:{article_id}"
        ticker_node = f"ticker:{ticker}"
        date_node = f"date:{date}"
        add_node(nodes, article_node, "Article", row["_title"][:250] or article_id, ticker=ticker, date=date, attrs={"url": row.get("_url", "")})
        add_node(nodes, ticker_node, "Ticker", ticker, ticker=ticker)
        add_node(nodes, date_node, "Date", date, date=date)
        add_edge(edges, article_node, ticker_node, "ABOUT_TICKER", article_id=article_id)
        add_edge(edges, article_node, date_node, "PUBLISHED_ON", article_id=article_id)

    # Build a canonical company lookup before adding event edges. This lets
    # mentioned entities resolve back to Company/Ticker nodes even when the
    # company appears as the main company in a different event later in the CSV.
    company_records: Dict[str, Dict[str, Any]] = {}
    company_lookup: Dict[str, str] = {}

    def company_node_id(company_name: Any, ticker_value: Any = "") -> str:
        """Create the canonical graph node ID for a company record.

        Args:
            company_name: Company display name or alias.
            ticker_value: Optional ticker used as the stronger identifier.

        Returns:
            Stable company node ID, or an empty string when neither name nor
            ticker can be normalized.
        """
        ticker_norm = normalize_ticker(ticker_value)
        company_norm = normalize_entity_name(company_name)
        if ticker_norm:
            return f"company_ticker:{ticker_norm}"
        if company_norm:
            return "company:" + sha1_short(company_norm.lower(), 16)
        return ""

    def register_company(company_name: Any, ticker_value: Any = "", is_anchor: bool = False) -> str:
        """Register or update a canonical company record for later node creation.

        Args:
            company_name: Raw company name extracted from an article or event.
            ticker_value: Optional ticker associated with the company.
            is_anchor: Whether the company is one of the configured anchor
                forecasting targets.

        Returns:
            Canonical company node ID, or an empty string when the input cannot
            be normalized.
        """
        ticker_norm = normalize_ticker(ticker_value)
        anchor_ticker = resolve_anchor_ticker(company_name, ticker_norm, anchor_context, anchor_lookup)
        if anchor_ticker:
            ticker_norm = anchor_ticker
            company_name = clean_text(anchor_context[anchor_ticker].get("company", "")) or company_name
            is_anchor = True
        company_norm = normalize_entity_name(company_name)
        comp_id = company_node_id(company_norm, ticker_norm)
        if not comp_id:
            return ""
        display_name = company_norm or ticker_norm
        record = company_records.setdefault(comp_id, {"name": display_name, "ticker": ticker_norm, "is_anchor": bool(is_anchor)})
        if is_anchor:
            record["is_anchor"] = True
        if ticker_norm and not record.get("ticker"):
            record["ticker"] = ticker_norm
        if company_norm and (not record.get("name") or record.get("name") == record.get("ticker")):
            record["name"] = company_norm
        for key in entity_resolution_keys(record.get("name", ""), record.get("ticker", "")):
            company_lookup.setdefault(key, comp_id)
        for key in entity_resolution_keys(company_norm, ticker_norm):
            company_lookup.setdefault(key, comp_id)
        return comp_id

    def resolve_company(entity_name: Any) -> str:
        """Resolve a raw entity mention to a registered company node.

        Args:
            entity_name: Raw entity or company mention from an event.

        Returns:
            Matching company node ID, or an empty string when no conservative
            match exists.
        """
        for key in entity_resolution_keys(entity_name):
            comp_id = company_lookup.get(key)
            if comp_id:
                return comp_id
        return ""

    def ensure_company_node(comp_id: str) -> Tuple[str, str]:
        """Materialize a registered company and its ticker edge in the graph.

        Args:
            comp_id: Canonical company node ID previously returned by
                `register_company`.

        Returns:
            Tuple of company display name and normalized ticker.
        """
        record = company_records.get(comp_id, {})
        company_name = record.get("name", comp_id)
        company_ticker = record.get("ticker", "")
        add_node(nodes, comp_id, "Company", company_name, ticker=company_ticker, attrs={"is_anchor": bool(record.get("is_anchor", False))})
        if company_ticker:
            ticker_node_id = f"ticker:{company_ticker}"
            add_node(nodes, ticker_node_id, "Ticker", company_ticker, ticker=company_ticker)
            add_edge(edges, comp_id, ticker_node_id, "HAS_TICKER")
        return company_name, company_ticker

    for anchor_ticker, record in anchor_context.items():
        comp_id = register_company(record.get("company", anchor_ticker), anchor_ticker, is_anchor=True)
        if comp_id:
            ensure_company_node(comp_id)

    for _, ev in event_df.iterrows():
        main_company = normalize_entity_name(ev.get("main_company", ""))
        ticker = normalize_ticker(ev.get("ticker", ""))
        if main_company or ticker:
            register_company(main_company, ticker)

    # Event nodes and edges.
    for idx, ev in tqdm(event_df.iterrows(), total=len(event_df), desc="Adding event nodes"):
        event_id = ev["event_id"]
        article_id = ev["article_id"]
        ticker = normalize_ticker(ev.get("ticker", ""))
        article_date = parse_date(ev.get("article_date", "")) or parse_date(article_meta.get(article_id, {}).get("_date", ""))
        available_date = parse_date(ev.get("available_date", "")) or article_date
        event_start = parse_date(ev.get("event_date_start", "")) or parse_date(ev.get("date", "")) or article_date
        event_end = parse_date(ev.get("event_date_end", "")) or event_start
        date = event_start or article_date
        event_time_role = clean_text(ev.get("event_time_role", "unknown")) or "unknown"
        event_time_granularity = clean_text(ev.get("event_time_granularity", "unknown")) or "unknown"
        etype = ev["event_type"]
        event_node = f"event:{event_id}"
        article_node = f"article:{article_id}"
        ticker_node = f"ticker:{ticker}" if ticker else ""
        date_node = f"date:{date}"
        article_date_node = f"date:{article_date}" if article_date else ""
        available_date_node = f"date:{available_date}" if available_date else ""
        event_end_node = f"date:{event_end}" if event_end else ""
        role_node = f"event_time_role:{event_time_role}" if event_time_role else ""
        etype_node = f"event_type:{etype}"

        attrs = {
            "event_type": etype,
            "trigger": ev.get("event_trigger", ""),
            "importance_score": int(ev.get("importance_score", 1)),
            "confidence": float(ev.get("confidence", 0.0)),
            "main_company": ev.get("main_company", ""),
            "article_title": ev.get("article_title", ""),
            "url": ev.get("url", ""),
            "article_date": article_date,
            "available_date": available_date,
            "event_date_start": event_start,
            "event_date_end": event_end,
            "event_time_role": event_time_role,
            "event_time_granularity": event_time_granularity,
            "event_time_source": ev.get("event_time_source", ""),
            "event_time_confidence": ev.get("event_time_confidence", ""),
            "event_time_normalizer": ev.get("event_time_normalizer", ""),
        }
        add_node(nodes, event_node, "Event", ev["event_description"][:250], ticker=ticker, date=date, attrs=attrs)
        if date:
            add_node(nodes, date_node, "Date", date, date=date)
        if article_date_node:
            add_node(nodes, article_date_node, "Date", article_date, date=article_date)
        if available_date_node:
            add_node(nodes, available_date_node, "Date", available_date, date=available_date)
        if event_end_node:
            add_node(nodes, event_end_node, "Date", event_end, date=event_end)
        if role_node:
            add_node(nodes, role_node, "EventTimeRole", event_time_role)
        add_node(nodes, etype_node, "EventType", etype)
        add_edge(edges, article_node, event_node, "REPORTS", article_id=article_id, event_id=event_id)
        if ticker:
            add_node(nodes, ticker_node, "Ticker", ticker, ticker=ticker)
            add_edge(edges, event_node, ticker_node, "INVOLVES_TICKER", article_id=article_id, event_id=event_id)
        if date:
            add_edge(
                edges,
                event_node,
                date_node,
                "OCCURRED_ON",
                article_id=article_id,
                event_id=event_id,
                attrs={
                    "timeline": "event",
                    "role": event_time_role,
                    "granularity": event_time_granularity,
                    "source_text": ev.get("event_time_source", ""),
                },
            )
            add_edge(
                edges,
                event_node,
                date_node,
                "PERIOD_START",
                article_id=article_id,
                event_id=event_id,
                attrs={"timeline": "event", "role": event_time_role, "granularity": event_time_granularity},
            )
        if event_end_node and event_end != date:
            add_edge(
                edges,
                event_node,
                event_end_node,
                "PERIOD_END",
                article_id=article_id,
                event_id=event_id,
                attrs={"timeline": "event", "role": event_time_role, "granularity": event_time_granularity},
            )
        if article_date_node:
            add_edge(edges, event_node, article_date_node, "REPORTED_ON", article_id=article_id, event_id=event_id, attrs={"timeline": "article"})
        if available_date_node:
            add_edge(edges, event_node, available_date_node, "AVAILABLE_ON", article_id=article_id, event_id=event_id, attrs={"timeline": "availability"})
        if role_node:
            add_edge(edges, event_node, role_node, "HAS_TIME_ROLE", article_id=article_id, event_id=event_id)
        if event_time_role == "deadline" and event_end_node:
            add_edge(edges, event_node, event_end_node, "DEADLINE_ON", article_id=article_id, event_id=event_id)
        elif event_time_role == "guidance_period":
            add_edge(edges, event_node, date_node, "GUIDANCE_PERIOD_START", article_id=article_id, event_id=event_id)
            if event_end_node:
                add_edge(edges, event_node, event_end_node, "GUIDANCE_PERIOD_END", article_id=article_id, event_id=event_id)
        elif event_time_role == "future_effective_period":
            add_edge(edges, event_node, date_node, "EFFECTIVE_START", article_id=article_id, event_id=event_id)
            if event_end_node:
                add_edge(edges, event_node, event_end_node, "EFFECTIVE_END", article_id=article_id, event_id=event_id)
        add_edge(edges, event_node, etype_node, "HAS_TYPE", article_id=article_id, event_id=event_id)

        # Main company if extracted.
        main_company = normalize_entity_name(ev.get("main_company", ""))
        if main_company:
            comp_id = register_company(main_company, ticker)
            ensure_company_node(comp_id)
            company_edge_type = event_company_edge_type(etype)
            add_edge(edges, event_node, comp_id, company_edge_type, article_id=article_id, event_id=event_id)
            add_edge(edges, comp_id, event_node, reverse_edge_type(company_edge_type), article_id=article_id, event_id=event_id)
            if bool(getattr(args, "add_generic_company_edges", False)):
                add_edge(edges, event_node, comp_id, "INVOLVES_COMPANY", article_id=article_id, event_id=event_id)
            if ticker:
                add_edge(edges, comp_id, ticker_node, "HAS_TICKER", article_id=article_id, event_id=event_id)

        # Mentioned entities and participants.
        entities = []
        entities.extend(json_loads_maybe(ev.get("participants_json"), []))
        entities.extend(json_loads_maybe(ev.get("mentioned_entities_json"), []))
        seen = set()
        for ent in entities:
            ent_raw = clean_text(ent)
            ent_norm = normalize_entity_name(ent_raw)
            if not ent_norm or ent_norm.lower() in seen:
                continue
            seen.add(ent_norm.lower())
            ent_id = "entity:" + sha1_short(ent_norm.lower(), 18)
            raw_id = "entity_raw:" + sha1_short(ent_raw.lower(), 18)
            add_node(nodes, ent_id, "Entity", ent_norm, attrs={"canonical": True})
            add_edge(edges, event_node, ent_id, "MENTIONS_ENTITY", article_id=article_id, event_id=event_id)
            if ent_raw != ent_norm:
                add_node(nodes, raw_id, "EntityRawMention", ent_raw)
                add_edge(edges, raw_id, ent_id, "SAME_AS", article_id=article_id, event_id=event_id)
            anchor_ticker = resolve_anchor_ticker(ent_raw or ent_norm, "", anchor_context, anchor_lookup)
            resolved_comp_id = ""
            if anchor_ticker:
                resolved_comp_id = register_company(anchor_context[anchor_ticker].get("company", anchor_ticker), anchor_ticker, is_anchor=True)
            if not resolved_comp_id:
                resolved_comp_id = resolve_company(ent_norm) or resolve_company(ent_raw)
            if resolved_comp_id:
                _, resolved_ticker = ensure_company_node(resolved_comp_id)
                add_edge(
                    edges,
                    ent_id,
                    resolved_comp_id,
                    "RESOLVES_TO_COMPANY",
                    article_id=article_id,
                    event_id=event_id,
                    attrs={"raw_mention": ent_raw},
                )
                add_edge(
                    edges,
                    event_node,
                    resolved_comp_id,
                    "MENTIONS_ANCHOR_COMPANY" if anchor_ticker else "MENTIONS_COMPANY",
                    article_id=article_id,
                    event_id=event_id,
                    attrs={"raw_mention": ent_raw, "resolved_ticker": resolved_ticker},
                )
            else:
                unresolved_entities.append(
                    {
                        "entity_id": ent_id,
                        "event_node": event_node,
                        "entity_name": ent_norm,
                        "raw_mention": ent_raw,
                        "article_id": article_id,
                        "event_id": event_id,
                    }
                )

        metrics = json_loads_maybe(ev.get("financial_metric_mentions_json"), [])
        for metric in metrics[:50]:
            metric_s = clean_text(metric)
            if not metric_s:
                continue
            m_id = "metric_mention:" + sha1_short(metric_s.lower(), 18)
            add_node(nodes, m_id, "FinancialMetricMention", metric_s)
            add_edge(edges, event_node, m_id, "MENTIONS_METRIC", article_id=article_id, event_id=event_id)

        causal = json_loads_maybe(ev.get("explicit_causal_relations_json"), [])
        for c in causal[:20]:
            c_s = clean_text(c)
            if c_s:
                c_id = "causal_statement:" + sha1_short(c_s.lower(), 18)
                add_node(nodes, c_id, "CausalStatement", c_s)
                add_edge(edges, event_node, c_id, "HAS_EXPLICIT_CAUSAL_STATEMENT", article_id=article_id, event_id=event_id)

        anchor_impacts = derive_anchor_impacts_for_event(ev, anchor_context, anchor_lookup)
        for impact in anchor_impacts:
            anchor_ticker = normalize_ticker(impact.get("ticker", ""))
            if not anchor_ticker or anchor_ticker not in anchor_context:
                continue
            comp_id = register_company(anchor_context[anchor_ticker].get("company", anchor_ticker), anchor_ticker, is_anchor=True)
            ensure_company_node(comp_id)
            impact_edge_type = anchor_impact_edge_type(impact.get("impact_type", ""))
            impact_direction = normalize_anchor_impact_direction(impact.get("impact_direction", ""))
            impact_direction_score = anchor_impact_direction_score(impact_direction)
            attrs = {
                "impact_type": normalize_anchor_impact_type(impact.get("impact_type", ""), etype),
                "impact_direction": impact_direction,
                "impact_direction_score": impact_direction_score,
                "evidence": clean_text(impact.get("evidence", "")),
                "reasoning": clean_text(impact.get("reasoning", "")),
                "raw_mention": clean_text(impact.get("raw_mention", "")),
                "anchor_ticker": anchor_ticker,
            }
            add_edge(edges, event_node, comp_id, impact_edge_type, weight=impact_direction_score, article_id=article_id, event_id=event_id, attrs=attrs)
            add_edge(edges, comp_id, event_node, reverse_edge_type(impact_edge_type), weight=impact_direction_score, article_id=article_id, event_id=event_id, attrs=attrs)
            add_edge(edges, article_node, comp_id, "ARTICLE_IMPACTS_ANCHOR", weight=impact_direction_score, article_id=article_id, event_id=event_id, attrs=attrs)
            add_edge(edges, article_node, f"ticker:{anchor_ticker}", "ARTICLE_IMPACTS_ANCHOR_TICKER", weight=impact_direction_score, article_id=article_id, event_id=event_id, attrs=attrs)

    if fusion_result is not None:
        event_rows_by_id = (
            event_df.set_index("event_id", drop=False).to_dict(orient="index")
            if len(event_df) > 0 and "event_id" in event_df.columns
            else {}
        )

        def truthy(value: Any) -> bool:
            """Return a boolean for bool-like CSV/Pandas values."""
            if isinstance(value, bool):
                return value
            return clean_text(value).lower() in {"true", "1", "yes"}

        def entity_cluster_node_id(cluster_id: Any) -> str:
            """Return the graph node id for an entity cluster."""
            return f"entity_cluster:{clean_text(cluster_id)}"

        def event_cluster_node_id(cluster_id: Any) -> str:
            """Return the graph node id for an event cluster."""
            return f"event_cluster:{clean_text(cluster_id)}"

        def add_entity_cluster_node(cluster: Dict[str, Any]) -> str:
            """Materialize one entity-cluster node."""
            cluster_id = clean_text(cluster.get("entity_cluster_id", ""))
            node_id = entity_cluster_node_id(cluster_id)
            if not cluster_id:
                return ""
            add_node(
                nodes,
                node_id,
                "EntityCluster",
                clean_text(cluster.get("representative_name", "")) or cluster_id,
                ticker=normalize_ticker(cluster.get("ticker", "")),
                attrs={
                    "entity_cluster_id": cluster_id,
                    "member_mention_count": int(float(cluster.get("member_mention_count", 0) or 0)),
                    "source_event_count": int(float(cluster.get("source_event_count", 0) or 0)),
                    "member_texts_json": clean_text(cluster.get("member_texts_json", "[]")),
                },
            )
            return node_id

        def add_event_cluster_node(cluster: Dict[str, Any]) -> str:
            """Materialize one event-cluster node."""
            cluster_id = clean_text(cluster.get("event_cluster_id", ""))
            node_id = event_cluster_node_id(cluster_id)
            if not cluster_id:
                return ""
            date = parse_date(cluster.get("event_date_start", ""))
            add_node(
                nodes,
                node_id,
                "EventCluster",
                clean_text(cluster.get("canonical_description", ""))[:250] or cluster_id,
                ticker=normalize_ticker(cluster.get("ticker", "")),
                date=date,
                attrs={
                    "event_cluster_id": cluster_id,
                    "representative_event_id": clean_text(cluster.get("representative_event_id", "")),
                    "cluster_size": int(float(cluster.get("cluster_size", 0) or 0)),
                    "event_type": clean_text(cluster.get("event_type", "")),
                    "member_event_ids_json": clean_text(cluster.get("member_event_ids_json", "[]")),
                },
            )
            return node_id

        def materialize_entity_member_source(member: Dict[str, Any]) -> str:
            """Return the source graph node represented by one entity-cluster member."""
            text = clean_text(member.get("entity_text", ""))
            ticker = normalize_ticker(member.get("ticker", ""))
            source = clean_text(member.get("source", ""))
            if source in {"main_company", "anchor_impact"} and (text or ticker):
                comp_id = register_company(text or ticker, ticker)
                if comp_id:
                    ensure_company_node(comp_id)
                    return comp_id
            if ticker:
                comp_id = register_company(text or ticker, ticker)
                if comp_id:
                    ensure_company_node(comp_id)
                    return comp_id
            ent_norm = normalize_entity_name(text)
            if not ent_norm:
                return ""
            ent_id = "entity:" + sha1_short(ent_norm.lower(), 18)
            add_node(nodes, ent_id, "Entity", ent_norm, attrs={"canonical": True})
            return ent_id

        for _, cluster in fusion_result.entity_clusters.iterrows():
            add_entity_cluster_node(cluster.to_dict())
        seen_entity_sources = set()
        for _, member in fusion_result.entity_cluster_members.iterrows():
            member_dict = member.to_dict()
            cluster_id = clean_text(member_dict.get("entity_cluster_id", ""))
            cluster_node = entity_cluster_node_id(cluster_id)
            source_node = materialize_entity_member_source(member_dict)
            if not cluster_id or not source_node:
                continue
            source_key = (source_node, cluster_node)
            if source_key not in seen_entity_sources:
                seen_entity_sources.add(source_key)
                add_edge(
                    edges,
                    source_node,
                    cluster_node,
                    "FUSED_INTO_ENTITY_CLUSTER",
                    attrs={"entity_cluster_id": cluster_id},
                )
            if truthy(member_dict.get("is_representative", False)):
                add_edge(
                    edges,
                    cluster_node,
                    source_node,
                    "CANONICAL_ENTITY_REPRESENTS",
                    attrs={"entity_cluster_id": cluster_id},
                )
            event_id = clean_text(member_dict.get("event_id", ""))
            if event_id:
                add_edge(
                    edges,
                    f"event:{event_id}",
                    cluster_node,
                    "MENTIONS_ENTITY_CLUSTER",
                    article_id=clean_text(event_rows_by_id.get(event_id, {}).get("article_id", "")),
                    event_id=event_id,
                    attrs={
                        "entity_cluster_id": cluster_id,
                        "source": clean_text(member_dict.get("source", "")),
                        "entity_text": clean_text(member_dict.get("entity_text", "")),
                    },
                )

        for _, cluster in fusion_result.event_clusters.iterrows():
            cluster_dict = cluster.to_dict()
            cluster_node = add_event_cluster_node(cluster_dict)
            if not cluster_node:
                continue
            cluster_id = clean_text(cluster_dict.get("event_cluster_id", ""))
            members = fusion_result.event_cluster_members[
                fusion_result.event_cluster_members["event_cluster_id"].astype(str) == cluster_id
            ]
            for _, member in members.iterrows():
                event_id = clean_text(member.get("event_id", ""))
                if not event_id:
                    continue
                add_edge(
                    edges,
                    f"event:{event_id}",
                    cluster_node,
                    "FUSED_INTO_EVENT_CLUSTER",
                    weight=float(member.get("similarity_to_representative", 1.0) or 1.0),
                    article_id=clean_text(event_rows_by_id.get(event_id, {}).get("article_id", "")),
                    event_id=event_id,
                    attrs={
                        "event_cluster_id": cluster_id,
                        "is_representative": truthy(member.get("is_representative", False)),
                        "merge_reason": clean_text(member.get("merge_reason", "")),
                    },
                )
                if truthy(member.get("is_representative", False)):
                    add_edge(
                        edges,
                        cluster_node,
                        f"event:{event_id}",
                        "CANONICAL_EVENT_REPRESENTS",
                        event_id=event_id,
                        attrs={"event_cluster_id": cluster_id},
                    )

        for _, cluster in fusion_result.event_clusters.iterrows():
            cluster_dict = cluster.to_dict()
            cluster_id = clean_text(cluster_dict.get("event_cluster_id", ""))
            cluster_node = event_cluster_node_id(cluster_id)
            rep_event_id = clean_text(cluster_dict.get("representative_event_id", ""))
            rep_ev = event_rows_by_id.get(rep_event_id, {})
            if not rep_ev:
                continue
            ticker = normalize_ticker(rep_ev.get("ticker", ""))
            etype = clean_text(rep_ev.get("event_type", ""))
            date = parse_date(rep_ev.get("event_date_start", "")) or parse_date(rep_ev.get("date", ""))
            event_end = parse_date(rep_ev.get("event_date_end", "")) or date
            if ticker:
                ticker_node = f"ticker:{ticker}"
                add_node(nodes, ticker_node, "Ticker", ticker, ticker=ticker)
                add_edge(edges, cluster_node, ticker_node, "CLUSTER_INVOLVES_TICKER", attrs={"event_cluster_id": cluster_id})
            if etype:
                etype_node = f"event_type:{etype}"
                add_node(nodes, etype_node, "EventType", etype)
                add_edge(edges, cluster_node, etype_node, "CLUSTER_HAS_TYPE", attrs={"event_cluster_id": cluster_id})
            if date:
                date_node = f"date:{date}"
                add_node(nodes, date_node, "Date", date, date=date)
                add_edge(edges, cluster_node, date_node, "CLUSTER_OCCURRED_ON", attrs={"event_cluster_id": cluster_id})
            if event_end and event_end != date:
                end_node = f"date:{event_end}"
                add_node(nodes, end_node, "Date", event_end, date=event_end)
                add_edge(edges, cluster_node, end_node, "CLUSTER_PERIOD_END", attrs={"event_cluster_id": cluster_id})
            main_company = normalize_entity_name(rep_ev.get("main_company", ""))
            if main_company:
                comp_id = register_company(main_company, ticker)
                ensure_company_node(comp_id)
                company_edge_type = "CLUSTER_" + event_company_edge_type(etype)
                add_edge(edges, cluster_node, comp_id, company_edge_type, attrs={"event_cluster_id": cluster_id})
            primary_entity_cluster = fusion_result.primary_entity_cluster_by_event_id.get(rep_event_id, "")
            if primary_entity_cluster:
                add_edge(
                    edges,
                    cluster_node,
                    entity_cluster_node_id(primary_entity_cluster),
                    "CLUSTER_INVOLVES_ENTITY_CLUSTER",
                    attrs={"event_cluster_id": cluster_id, "entity_cluster_id": primary_entity_cluster},
                )

            member_ids = json_loads_maybe(cluster_dict.get("member_event_ids_json", "[]"), [])
            seen_metrics = set()
            for member_event_id in member_ids:
                member_ev = event_rows_by_id.get(clean_text(member_event_id), {})
                for metric in json_loads_maybe(member_ev.get("financial_metric_mentions_json", "[]"), []):
                    metric_s = clean_text(metric)
                    if not metric_s or metric_s.lower() in seen_metrics:
                        continue
                    seen_metrics.add(metric_s.lower())
                    m_id = "metric_mention:" + sha1_short(metric_s.lower(), 18)
                    add_node(nodes, m_id, "FinancialMetricMention", metric_s)
                    add_edge(edges, cluster_node, m_id, "CLUSTER_MENTIONS_METRIC", attrs={"event_cluster_id": cluster_id})
                for impact in derive_anchor_impacts_for_event(member_ev, anchor_context, anchor_lookup):
                    anchor_ticker = normalize_ticker(impact.get("ticker", ""))
                    if not anchor_ticker or anchor_ticker not in anchor_context:
                        continue
                    comp_id = register_company(anchor_context[anchor_ticker].get("company", anchor_ticker), anchor_ticker, is_anchor=True)
                    ensure_company_node(comp_id)
                    impact_direction = normalize_anchor_impact_direction(impact.get("impact_direction", ""))
                    attrs = {
                        "event_cluster_id": cluster_id,
                        "impact_type": normalize_anchor_impact_type(impact.get("impact_type", ""), etype),
                        "impact_direction": impact_direction,
                        "impact_direction_score": anchor_impact_direction_score(impact_direction),
                        "evidence": clean_text(impact.get("evidence", "")),
                        "raw_mention": clean_text(impact.get("raw_mention", "")),
                        "anchor_ticker": anchor_ticker,
                    }
                    add_edge(
                        edges,
                        cluster_node,
                        comp_id,
                        "CLUSTER_" + anchor_impact_edge_type(impact.get("impact_type", "")),
                        weight=anchor_impact_direction_score(impact_direction),
                        attrs=attrs,
                    )

        tmp_cluster = event_df.copy()
        if len(tmp_cluster) > 0 and "event_id" in tmp_cluster.columns:
            tmp_cluster["_cluster_id"] = tmp_cluster["event_id"].astype(str).map(fusion_result.event_cluster_by_event_id).fillna("")
            tmp_cluster["_cluster_date"] = tmp_cluster.apply(
                lambda r: parse_date(r.get("event_date_start", "")) or parse_date(r.get("date", "")),
                axis=1,
            )
            tmp_cluster = tmp_cluster[
                (tmp_cluster["_cluster_id"].astype(str) != "")
                & (tmp_cluster["ticker"].astype(str).str.strip() != "")
                & (tmp_cluster["_cluster_date"].astype(str) != "")
            ].copy()
            tmp_cluster = tmp_cluster.sort_values(["ticker", "_cluster_date", "_cluster_id", "event_id"])
            for ticker, group in tmp_cluster.groupby("ticker", sort=False):
                cluster_sequence = group.drop_duplicates(subset=["_cluster_id"]).sort_values(["_cluster_date", "_cluster_id"])
                cluster_ids = cluster_sequence["_cluster_id"].tolist()
                dates = cluster_sequence["_cluster_date"].tolist()
                for i in range(len(cluster_ids) - 1):
                    if cluster_ids[i] == cluster_ids[i + 1]:
                        continue
                    add_edge(
                        edges,
                        event_cluster_node_id(cluster_ids[i]),
                        event_cluster_node_id(cluster_ids[i + 1]),
                        "EVENT_CLUSTER_PRECEDES",
                        attrs={
                            "ticker": ticker,
                            "source_date": dates[i],
                            "target_date": dates[i + 1],
                            "timeline": "event_cluster",
                        },
                    )

    if bool(getattr(args, "semantic_entity_resolution", False)):
        add_semantic_entity_company_resolution(
            unresolved_entities,
            company_records,
            nodes,
            edges,
            args,
            ensure_company_node,
        )

    # Temporal chronology edges within each ticker. Event chronology follows the
    # event's normalized semantic time; report chronology follows when the
    # article made the event available to a downstream forecaster.
    if len(event_df) > 0:
        tmp = event_df.copy()
        tmp["_event_timeline_date"] = tmp.apply(
            lambda r: parse_date(r.get("event_date_start", "")) or parse_date(r.get("date", "")),
            axis=1,
        )
        tmp["_report_timeline_date"] = tmp.apply(
            lambda r: parse_date(r.get("available_date", "")) or parse_date(r.get("article_date", "")) or parse_date(r.get("date", "")),
            axis=1,
        )
        tmp = tmp.sort_values(["ticker", "_event_timeline_date", "article_id", "event_id"]).reset_index(drop=True)
        tmp = tmp[tmp["ticker"].astype(str).str.strip() != ""].copy()
        for ticker, group in tmp.groupby("ticker", sort=False):
            group = group.drop_duplicates(subset=["event_id"]).sort_values(["_event_timeline_date", "article_id", "event_id"])
            ids = group["event_id"].tolist()
            dates = group["_event_timeline_date"].tolist()
            for i in range(len(ids) - 1):
                if ids[i] == ids[i + 1]:
                    continue
                attrs = {"ticker": ticker, "source_date": dates[i], "target_date": dates[i + 1], "timeline": "event"}
                add_edge(
                    edges,
                    f"event:{ids[i]}",
                    f"event:{ids[i + 1]}",
                    "EVENT_PRECEDES",
                    article_id="",
                    event_id=ids[i],
                    attrs=attrs,
                )
                add_edge(
                    edges,
                    f"event:{ids[i]}",
                    f"event:{ids[i + 1]}",
                    "PRECEDES",
                    article_id="",
                    event_id=ids[i],
                    attrs=attrs,
                )

            report_group = group.sort_values(["_report_timeline_date", "article_id", "event_id"])
            report_ids = report_group["event_id"].tolist()
            report_dates = report_group["_report_timeline_date"].tolist()
            for i in range(len(report_ids) - 1):
                if report_ids[i] == report_ids[i + 1]:
                    continue
                add_edge(
                    edges,
                    f"event:{report_ids[i]}",
                    f"event:{report_ids[i + 1]}",
                    "REPORT_PRECEDES",
                    article_id="",
                    event_id=report_ids[i],
                    attrs={
                        "ticker": ticker,
                        "source_date": report_dates[i],
                        "target_date": report_dates[i + 1],
                        "timeline": "report",
                    },
                )

    # Semantic SAME_AS / RELATED_TO between event nodes using embeddings.
    if len(event_df) > 1 and event_embeddings.shape[0] == len(event_df):
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import normalize

            event_df_reset = event_df.reset_index(drop=True)
            emb = normalize(event_embeddings)
            for (ticker, etype), idxs in tqdm(event_df_reset.groupby(["ticker", "event_type"]).indices.items(), desc="Adding semantic event links"):
                if not clean_text(ticker):
                    continue
                idxs = list(idxs)
                if len(idxs) < 2:
                    continue
                group_emb = emb[idxs]
                n_neighbors = min(args.related_neighbors + 1, len(idxs))
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
                nn.fit(group_emb)
                distances, neighbors = nn.kneighbors(group_emb)
                for local_i, neighs in enumerate(neighbors):
                    i = idxs[local_i]
                    id_i = event_df_reset.loc[i, "event_id"]
                    date_i = pd.to_datetime(event_df_reset.loc[i, "date"], errors="coerce")
                    for local_j_pos, local_j in enumerate(neighs):
                        j = idxs[local_j]
                        if j <= i:
                            continue
                        sim = 1.0 - float(distances[local_i, local_j_pos])
                        if sim < args.related_threshold:
                            continue
                        date_j = pd.to_datetime(event_df_reset.loc[j, "date"], errors="coerce")
                        if pd.notna(date_i) and pd.notna(date_j):
                            days_apart = abs((date_j - date_i).days)
                            if days_apart > args.semantic_link_max_days:
                                continue
                        id_j = event_df_reset.loc[j, "event_id"]
                        edge_type = "SAME_AS" if sim >= args.same_event_threshold else "RELATED_TO"
                        add_edge(
                            edges,
                            f"event:{id_i}",
                            f"event:{id_j}",
                            edge_type,
                            weight=sim,
                            attrs={"similarity": round(sim, 4), "ticker": ticker, "event_type": etype},
                        )
        except Exception as e:
            print(f"WARNING: Could not add semantic event links: {e}", file=sys.stderr)

    nodes_df = pd.DataFrame([asdict(n) for n in nodes.values()])
    edges_df = pd.DataFrame([asdict(e) for e in edges.values()])
    if not nodes_df.empty and not edges_df.empty:
        node_type_by_id = nodes_df.set_index("node_id")["node_type"].to_dict()
        edges_df["source_node_type"] = edges_df["source"].map(node_type_by_id).fillna("")
        edges_df["target_node_type"] = edges_df["target"].map(node_type_by_id).fillna("")

    nodes_df, node_features = build_and_save_node_features(nodes_df, articles, event_df, event_embeddings, args, out_dir)
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)
    edges_df.to_csv(out_dir / "edges.csv", index=False)

    if bool(getattr(args, "export_hetero_tensors", True)):
        export_heterogeneous_graph_tensors(
            nodes_df,
            edges_df,
            node_features,
            out_dir,
            add_reverse_edges=bool(getattr(args, "hetero_add_reverse_edges", True)),
            export_torch=bool(getattr(args, "export_torch_hetero", False)),
        )

    # Export GraphML.
    try:
        import networkx as nx

        G = nx.MultiDiGraph()
        for _, n in nodes_df.iterrows():
            G.add_node(
                n["node_id"],
                node_type=n["node_type"],
                name=str(n["name"]),
                ticker=str(n.get("ticker", "")),
                date=str(n.get("date", "")),
                feature_row=int(n.get("feature_row", -1)),
                feature_source=str(n.get("feature_source", "")),
                feature_dim=int(n.get("feature_dim", 0)),
                attributes_json=str(n.get("attributes_json", "{}")),
            )
        for _, e in edges_df.iterrows():
            G.add_edge(
                e["source"],
                e["target"],
                edge_type=e["edge_type"],
                weight=float(e.get("weight", 1.0)),
                article_id=str(e.get("article_id", "")),
                event_id=str(e.get("event_id", "")),
                source_node_type=str(e.get("source_node_type", "")),
                target_node_type=str(e.get("target_node_type", "")),
                attributes_json=str(e.get("attributes_json", "{}")),
            )
        nx.write_graphml(G, out_dir / "graph.graphml")
    except Exception as e:
        print(f"WARNING: Could not export graph.graphml: {e}", file=sys.stderr)

    return nodes_df, edges_df


def run_global_build(args: argparse.Namespace, articles: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path) -> None:
    """Build final graph artifacts from already extracted events.

    Args:
        args: Parsed CLI arguments.
        articles: Merged article dataframe.
        event_df: Merged event dataframe.
        out_dir: Final output directory.

    Returns:
        None.
    """
    print(f"Global build input: {len(articles)} articles, {len(event_df)} events")

    # Save merged extraction tables for reproducibility.
    save_articles_snapshot(articles, out_dir)
    event_df.to_csv(out_dir / "events.csv", index=False)

    event_df, event_embeddings = add_event_embeddings(event_df, args, out_dir)
    event_df.to_csv(out_dir / "events.csv", index=False)

    fusion_result = run_fusion(event_df, event_embeddings, args, out_dir)
    print(
        "Fusion: "
        f"{len(fusion_result.event_clusters)} event clusters, "
        f"{len(fusion_result.entity_clusters)} entity clusters"
    )

    nodes_df, edges_df = build_graph_tables(articles, event_df, event_embeddings, args, out_dir, fusion_result=fusion_result)
    print(f"Graph: {len(nodes_df)} nodes, {len(edges_df)} edges")

    save_config(args, out_dir, articles, event_df)
