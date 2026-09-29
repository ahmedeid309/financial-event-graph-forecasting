from __future__ import annotations

import re
from typing import Any, Dict, List

from financial_ekg.utils.text import clean_text

METRIC_MENTION_RE = re.compile(
    r"""
    (?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?\s*
        (?:%|percent|million|billion|trillion|x|times|shares?|subscribers?|users?|customers?)?
    |\b(?:fiscal|FY|Q[1-4]|quarter|year|month)\s+20\d{2}\b
    |\b(?:20\d{2}|19\d{2})\b
    |\b(?:No\.?\s*\d+|\#\d+)\b
    |\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|largest|top\s+\d+)\b
    |\b(?:double|doubled|triple|tripled|twice|half)\b
    """,
    flags=re.I | re.X,
)


def normalize_metric_token(metric: Any) -> str:
    """Normalize a metric string for comparison.

    Args:
        metric: Raw metric text.

    Returns:
        Lowercase comparison key with common approximators and separators
        normalized.
    """
    text = clean_text(metric).lower()
    text = re.sub(r"\b(more than|roughly|nearly|about|approximately|around|over|under|at least|as of)\b", "", text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_metric_mentions_from_text(text: Any) -> List[str]:
    """Extract numeric and period-like metric mentions from text.

    Args:
        text: Text to scan for financial metric tokens.

    Returns:
        Ordered list of unique metric mentions as they appeared in the text.
    """
    out: List[str] = []
    for match in METRIC_MENTION_RE.finditer(clean_text(text)):
        metric = clean_text(match.group(0))
        norm = normalize_metric_token(metric)
        if norm and norm not in {normalize_metric_token(x) for x in out}:
            out.append(metric)
    return out


def span_metric_mentions(span: Dict[str, Any]) -> List[str]:
    """Collect metric mentions from an evidence span.

    Args:
        span: Evidence span dictionary with optional pre-extracted metrics and
            text.

    Returns:
        Deduplicated metric mentions from the span metadata and span text.
    """
    metrics: List[str] = []
    raw = span.get("numbers_or_money_mentions", [])
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for metric in raw:
            cleaned = clean_text(metric)
            if cleaned:
                metrics.append(cleaned)
    metrics.extend(extract_metric_mentions_from_text(span.get("text", "")))
    deduped: List[str] = []
    seen = set()
    for metric in metrics:
        norm = normalize_metric_token(metric)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(metric)
    return deduped
