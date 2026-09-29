from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from financial_ekg.utils.metrics import extract_metric_mentions_from_text, span_metric_mentions
from financial_ekg.utils.text import (
    clean_text,
    detect_article_ticker_mentions,
    split_article_sentences,
)


def fallback_stage1_article_map(
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    article: str,
    max_spans: Optional[int] = None,
) -> Dict[str, Any]:
    """Build deterministic Stage 1 evidence spans from metric-bearing sentences.

    Args:
        article_id: Stable article identifier.
        source_ticker: Ticker associated with the input article row.
        date: Article publication date.
        title: Article title, retained for signature compatibility.
        article: Full article body.
        max_spans: Optional maximum number of fallback spans to return.

    Returns:
        Stage 1 shaped article-map dictionary containing rule-based financial
        evidence spans and fallback metadata.
    """
    text = clean_text(article)
    sentences = split_article_sentences(text)
    detected_tickers = detect_article_ticker_mentions(article, source_ticker)
    number_pattern = re.compile(
        r"(?:\$?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|trillion|x)?|"
        r"\b(?:twice|double|triple|half)\b)",
        flags=re.I,
    )
    financial_keywords = re.compile(
        r"\b(stock|share|market|revenue|profit|income|earnings|sales|cash|valuation|"
        r"market cap|return|gained|rose|grew|growth|declined|fell|portfolio|holding|"
        r"rank|weight|customers|subscribers|users|margin|guidance|target|dividend|"
        r"buyback|acquisition|merger|lawsuit|regulatory|inflation|interest rate)\b",
        flags=re.I,
    )
    date_keywords = re.compile(r"\b(?:20\d{2}|19\d{2}|Q[1-4]|quarter|fiscal|year|month|week|today|yesterday)\b", flags=re.I)
    causal_keywords = re.compile(r"\b(?:because|due to|driven by|as a result|therefore|despite|led to|resulting in)\b", flags=re.I)

    spans: List[Dict[str, Any]] = []
    for sentence in sentences:
        if max_spans is not None and len(spans) >= max_spans:
            break
        has_number = bool(number_pattern.search(sentence))
        has_financial_language = bool(financial_keywords.search(sentence))
        mentioned = [t for t in detected_tickers if re.search(rf"\b{re.escape(t)}\b", sentence)]
        if not ((has_number and has_financial_language) or (mentioned and has_financial_language)):
            continue
        snippet = clean_text(sentence)
        if not snippet:
            continue
        spans.append(
            {
                "span_id": f"S{len(spans) + 1}",
                "text": snippet,
                "companies_or_assets": [],
                "tickers": mentioned,
                "contains_financial_fact": True,
                "contains_temporal_info": bool(date_keywords.search(sentence)),
                "contains_causal_info": bool(causal_keywords.search(sentence)),
                "numbers_or_money_mentions": [clean_text(m.group(0)) for m in number_pattern.finditer(sentence)][:12],
                "reason": "Fallback financial evidence sentence.",
            }
        )

    return {
        "article_id": article_id,
        "source_ticker": source_ticker,
        "date": date,
        "detected_tickers": detected_tickers,
        "detected_companies": [],
        "detected_entities": [],
        "financial_evidence_spans": spans,
        "article_level_notes": "Fallback evidence map created after Stage 1 parse failure.",
        "no_event_reason": "" if spans else "No fallback financial evidence spans found.",
        "_parse_error": False,
        "_fallback": True,
    }


def useful_evidence_spans(stage1_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter Stage 1 spans to useful financial evidence.

    Args:
        stage1_output: Article-map object returned by Stage 1 or the fallback
            mapper.

    Returns:
        List of span dictionaries that contain text and are marked as financial
        facts.
    """
    spans = stage1_output.get("financial_evidence_spans", [])
    if not isinstance(spans, list):
        return []
    useful: List[Dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = clean_text(span.get("text", ""))
        has_fact = span.get("contains_financial_fact", False)
        if isinstance(has_fact, str):
            has_fact = has_fact.strip().lower() in {"true", "yes", "1"}
        if text and bool(has_fact):
            useful.append(span)
    return useful


def renumber_stage1_spans(stage1_output: Dict[str, Any]) -> Dict[str, Any]:
    """Renumber useful evidence spans as `S1`, `S2`, and so on.

    Args:
        stage1_output: Article-map object with financial evidence spans.

    Returns:
        Copy of the article-map object with normalized span IDs.
    """
    out = dict(stage1_output)
    spans = useful_evidence_spans(stage1_output)
    normalized: List[Dict[str, Any]] = []
    for idx, span in enumerate(spans, start=1):
        new_span = dict(span)
        new_span["span_id"] = f"S{idx}"
        normalized.append(new_span)
    out["financial_evidence_spans"] = normalized
    return out


def merge_stage1_with_rule_based_spans(
    stage1_output: Dict[str, Any],
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    article: str,
) -> Dict[str, Any]:
    """Merge LLM article-map spans with deterministic evidence spans.

    Args:
        stage1_output: Raw Stage 1 article-map output.
        article_id: Stable article identifier.
        source_ticker: Ticker associated with the input article row.
        date: Article publication date.
        title: Article title.
        article: Full article body.

    Returns:
        Stage 1 object whose evidence spans combine useful LLM spans with
        deterministic fallback spans when needed.
    """
    llm_spans = [] if stage1_output.get("_llm_stage1_skipped") else useful_evidence_spans(stage1_output)
    fallback_needed = bool(stage1_output.get("_parse_error") or stage1_output.get("_llm_stage1_skipped") or not llm_spans)
    rule_spans: List[Dict[str, Any]] = []
    if fallback_needed:
        fallback = fallback_stage1_article_map(article_id, source_ticker, date, title, article)
        rule_spans = useful_evidence_spans(fallback)

    merged: List[Dict[str, Any]] = []
    seen_texts: List[str] = []

    def is_duplicate(text: str) -> bool:
        """Check whether a candidate span repeats already accepted evidence.

        Args:
            text: Candidate evidence text.

        Returns:
            True when the text is empty or overlaps a previously accepted span.
        """
        norm = clean_text(text).lower()
        if not norm:
            return True
        for seen in seen_texts:
            if norm == seen or norm in seen or seen in norm:
                return True
        return False

    for span in list(llm_spans) + list(rule_spans):
        text = clean_text(span.get("text", ""))
        if is_duplicate(text):
            continue
        merged.append(dict(span))
        seen_texts.append(text.lower())

    out = dict(stage1_output)
    out["financial_evidence_spans"] = merged
    out["_llm_span_count"] = len(llm_spans)
    out["_rule_based_span_count"] = len(rule_spans)
    out["_stage1_augmented_with_rule_based_spans"] = bool(rule_spans)
    return renumber_stage1_spans(out)


def build_adaptive_evidence_blocks(
    stage1_output: Dict[str, Any],
    article: str,
    target_block_chars: int = 2500,
    max_blocks: int = 4,
) -> List[Dict[str, Any]]:
    """Group article evidence into a bounded number of extraction blocks.

    Args:
        stage1_output: Article-map object containing financial evidence spans.
        article: Full article text used for fallback sentence splitting.
        target_block_chars: Approximate character budget for each block.
        max_blocks: Maximum number of blocks to return.

    Returns:
        List of block dictionaries containing block IDs, span IDs, and compact
        evidence payloads for Stage 2 prompts.
    """
    target_block_chars = max(1500, int(target_block_chars or 2500))
    max_blocks = max(1, int(max_blocks or 1))

    spans = useful_evidence_spans(stage1_output)
    if not spans:
        spans = fallback_stage1_article_map("", "", "", "", article).get("financial_evidence_spans", [])
    if not spans:
        spans = [
            {
                "span_id": f"S{idx + 1}",
                "text": sentence,
                "tickers": [],
                "numbers_or_money_mentions": extract_metric_mentions_from_text(sentence),
                "contains_financial_fact": True,
            }
            for idx, sentence in enumerate(split_article_sentences(article))
            if clean_text(sentence)
        ]

    items: List[Dict[str, Any]] = []
    for idx, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            continue
        text = clean_text(span.get("text", ""))
        if not text:
            continue
        items.append(
            {
                "span_id": clean_text(span.get("span_id", "")) or f"S{idx}",
                "text": text,
                "tickers": span.get("tickers", []) if isinstance(span.get("tickers", []), list) else [],
                "metrics": span_metric_mentions(span)[:10],
            }
        )

    if not items:
        return []

    initial_blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for item in items:
        item_chars = len(json.dumps(item, ensure_ascii=False))
        if current and current_chars + item_chars > target_block_chars:
            initial_blocks.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        initial_blocks.append(current)

    if len(initial_blocks) <= max_blocks:
        grouped = initial_blocks
    else:
        grouped = [[] for _ in range(max_blocks)]
        for idx, block in enumerate(initial_blocks):
            grouped_idx = min(max_blocks - 1, int(idx * max_blocks / len(initial_blocks)))
            grouped[grouped_idx].extend(block)

    blocks: List[Dict[str, Any]] = []
    for idx, block_items in enumerate(grouped, start=1):
        if not block_items:
            continue
        blocks.append(
            {
                "block_id": f"B{idx}",
                "span_count": len(block_items),
                "span_ids": [item["span_id"] for item in block_items],
                "evidence": block_items,
            }
        )
    return blocks
