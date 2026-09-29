from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from financial_ekg.config import ANCHOR_IMPACT_DIRECTIONS, ANCHOR_IMPACT_TYPES, EVENT_TYPES
from financial_ekg.pipeline.evidence import useful_evidence_spans
from financial_ekg.pipeline.stage6 import events_for_anchor_impact_prompt
from financial_ekg.utils.anchors import anchor_context_for_prompt
from financial_ekg.utils.metrics import span_metric_mentions
from financial_ekg.utils.serialization import json_for_prompt, sanitize_for_json
from financial_ekg.utils.text import (
    article_text_for_llm,
    clean_text,
    detect_article_ticker_mentions,
    safe_str,
)
from financial_ekg.pipeline.evidence_prompts import (
    ARTICLE_MAP_PROMPT_TEMPLATE,
    EVENT_BLOCK_PROMPT_TEMPLATE,
    EVIDENCE_ADMISSIBILITY_PROMPT_TEMPLATE,
    FINAL_CONSOLIDATION_PROMPT_TEMPLATE,
    SECTION_CLASSIFICATION_PROMPT_TEMPLATE,
    STAGE6_ANCHOR_IMPACT_PROMPT_TEMPLATE,
    TEMPORAL_INVENTORY_PROMPT_TEMPLATE,
    TEMPORAL_JUDGE_PROMPT_TEMPLATE,
    TEMPORAL_NORMALIZATION_EVIDENCE_PROMPT_TEMPLATE,
    TEMPORAL_ROLE_SELECTION_PROMPT_TEMPLATE,
)


def compact_stage1_for_prompt(stage1_output: Dict[str, Any], max_hints: int = 12) -> Dict[str, Any]:
    """Return a compact article-map payload for follow-up prompts."""
    out = {
        key: sanitize_for_json(value)
        for key, value in stage1_output.items()
        if not safe_str(key).startswith("_") and key != "financial_evidence_spans"
    }
    hints: List[Dict[str, Any]] = []
    for span in useful_evidence_spans(stage1_output)[:max_hints]:
        if not isinstance(span, dict):
            continue
        metrics = span_metric_mentions(span)[:8]
        text = clean_text(span.get("text", ""))
        keywords: List[str] = []
        for word in re.findall(
            r"\b[A-Z][A-Za-z&.\-]{2,}\b|\b(?:revenue|profit|income|stock|shares|portfolio|market cap|subscribers|users|customers|grew|gained|soared|up|down)\b",
            text,
        ):
            cleaned = clean_text(word)
            if cleaned and cleaned.lower() not in {item.lower() for item in keywords}:
                keywords.append(cleaned)
            if len(keywords) >= 8:
                break
        hints.append(
            {
                "span_id": clean_text(span.get("span_id", "")),
                "metrics": metrics,
                "tickers": span.get("tickers", []) if isinstance(span.get("tickers", []), list) else [],
                "keywords": keywords,
            }
        )
    out["financial_evidence_hint_count"] = len(useful_evidence_spans(stage1_output))
    out["financial_evidence_metric_hints"] = hints
    return out


def build_article_map_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    article: str,
    anchor_context: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Build the evidence-first article-map prompt."""
    text = article_text_for_llm(title, article)
    detected_tickers = detect_article_ticker_mentions(article, source_ticker)
    detected_tickers_text = ", ".join(detected_tickers) if detected_tickers else "none"
    return ARTICLE_MAP_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        detected_tickers_text=detected_tickers_text,
        anchors_json=json_for_prompt(anchor_context_for_prompt(anchor_context or {})),
        text=text,
    )


def build_section_classification_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    article: str,
) -> str:
    """Build the evidence-first article-section classifier prompt."""
    return SECTION_CLASSIFICATION_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        text=article_text_for_llm(title, article),
    )


def evidence_span_lookup(stage1_output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return useful evidence spans keyed by span_id."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for span in useful_evidence_spans(stage1_output):
        if not isinstance(span, dict):
            continue
        span_id = clean_text(span.get("span_id", ""))
        if span_id:
            lookup[span_id] = span
    return lookup


def enrich_block_with_evidence_metadata(block: Dict[str, Any], stage1_output: Dict[str, Any]) -> Dict[str, Any]:
    """Add section metadata from annotated Stage 1 spans to an evidence block."""
    span_lookup = evidence_span_lookup(stage1_output)
    enriched_evidence: List[Dict[str, Any]] = []
    for item in block.get("evidence", []):
        if not isinstance(item, dict):
            continue
        span_id = clean_text(item.get("span_id", ""))
        span = span_lookup.get(span_id, {})
        enriched = dict(item)
        for key in [
            "section_id",
            "section_type",
            "allowed_for_event_extraction",
            "section_summary",
            "companies_or_assets",
            "numbers_or_money_mentions",
        ]:
            if key in span:
                enriched[key] = sanitize_for_json(span.get(key))
        enriched_evidence.append(enriched)
    out = dict(block)
    out["evidence"] = enriched_evidence
    return out


def build_event_block_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    stage1_output: Dict[str, Any],
    block: Dict[str, Any],
    block_index: int,
    block_count: int,
    anchor_context: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Build the evidence-first candidate-extraction prompt for one evidence block."""
    return EVENT_BLOCK_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        block_index=block_index,
        block_count=block_count,
        block_id=block.get("block_id", ""),
        allowed_types=", ".join(EVENT_TYPES),
        allowed_impact_types=", ".join(ANCHOR_IMPACT_TYPES),
        allowed_impact_directions=", ".join(ANCHOR_IMPACT_DIRECTIONS),
        article_map_json=json_for_prompt(compact_stage1_for_prompt(stage1_output, max_hints=0)),
        anchors_json=json_for_prompt(anchor_context_for_prompt(anchor_context or {})),
        block_json=json_for_prompt(enrich_block_with_evidence_metadata(block, stage1_output)),
    )


def build_evidence_admissibility_prompt(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    candidate_batch: Sequence[Dict[str, Any]],
) -> str:
    """Build the evidence-admissibility judge prompt."""
    return EVIDENCE_ADMISSIBILITY_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        candidate_batch_json=json_for_prompt({"candidates": list(candidate_batch)}),
    )


def build_final_consolidation_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    stage1_output: Dict[str, Any],
    accepted_candidates: Sequence[Dict[str, Any]],
    anchor_context: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Build the evidence-first final event-consolidation prompt."""
    return FINAL_CONSOLIDATION_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        allowed_types=", ".join(EVENT_TYPES),
        allowed_impact_types=", ".join(ANCHOR_IMPACT_TYPES),
        allowed_impact_directions=", ".join(ANCHOR_IMPACT_DIRECTIONS),
        article_map_json=json_for_prompt(compact_stage1_for_prompt(stage1_output, max_hints=0)),
        anchors_json=json_for_prompt(anchor_context_for_prompt(anchor_context or {})),
        accepted_candidates_json=json_for_prompt({"accepted_candidates": list(accepted_candidates)}),
    )


def build_anchor_impact_classification_prompt(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    stage1_output: Dict[str, Any],
    final_events: Sequence[Dict[str, Any]],
    anchor_context: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Build the prompt for finalized anchor-impact classification."""
    return STAGE6_ANCHOR_IMPACT_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        allowed_impact_types=", ".join(ANCHOR_IMPACT_TYPES),
        allowed_impact_directions=", ".join(ANCHOR_IMPACT_DIRECTIONS),
        article_map_json=json_for_prompt(compact_stage1_for_prompt(stage1_output, max_hints=8)),
        anchors_json=json_for_prompt(anchor_context_for_prompt(anchor_context or {})),
        final_events_json=json_for_prompt({"events": events_for_anchor_impact_prompt(final_events)}),
    )


def build_temporal_inventory_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    events: Sequence[Dict[str, Any]],
) -> str:
    """Build the evidence-first T1 temporal-inventory prompt."""
    return TEMPORAL_INVENTORY_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        events_json=json_for_prompt({"events": list(events)}),
    )


def build_temporal_role_selection_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    events: Sequence[Dict[str, Any]],
    inventories: Sequence[Dict[str, Any]],
) -> str:
    """Build the evidence-first T2 temporal-role-selection prompt."""
    return TEMPORAL_ROLE_SELECTION_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        events_json=json_for_prompt({"events": list(events)}),
        inventories_json=json_for_prompt({"temporal_inventories": list(inventories)}),
    )


def build_temporal_normalization_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    events: Sequence[Dict[str, Any]],
    roles: Sequence[Dict[str, Any]],
) -> str:
    """Build the evidence-first T3 temporal-normalization prompt."""
    return TEMPORAL_NORMALIZATION_EVIDENCE_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        events_json=json_for_prompt({"events": list(events)}),
        roles_json=json_for_prompt({"temporal_roles": list(roles)}),
    )


def build_temporal_judge_prompt_evidence(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    events: Sequence[Dict[str, Any]],
    inventories: Sequence[Dict[str, Any]],
    roles: Sequence[Dict[str, Any]],
    event_times: Sequence[Dict[str, Any]],
) -> str:
    """Build the evidence-first T4 temporal-judge prompt."""
    return TEMPORAL_JUDGE_PROMPT_TEMPLATE.format(
        article_id=article_id,
        source_ticker=source_ticker,
        date=date,
        title=title,
        events_json=json_for_prompt({"events": list(events)}),
        inventories_json=json_for_prompt({"temporal_inventories": list(inventories)}),
        roles_json=json_for_prompt({"temporal_roles": list(roles)}),
        event_times_json=json_for_prompt({"event_times": list(event_times)}),
    )
