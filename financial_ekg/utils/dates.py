from __future__ import annotations

import math

from typing import Any, Dict

import pandas as pd

from financial_ekg.config import EVENT_TIME_GRANULARITIES, EVENT_TIME_ROLES
from financial_ekg.utils.text import clean_text

def parse_date(value: Any) -> str:
    """Parse a date-like value into ISO format.

    Args:
        value: Date-like input.

    Returns:
        Date formatted as `YYYY-MM-DD`, or an empty string if parsing fails.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def event_time_dict(
    role: str = "unknown",
    start_date: str = "",
    end_date: str = "",
    granularity: str = "unknown",
    source_text: str = "",
    confidence: float = 0.0,
    normalizer: str = "none",
) -> Dict[str, Any]:
    """Build a normalized event-time payload.

    Args:
        role: Event-time role such as occurrence date or reported period.
        start_date: Raw start date.
        end_date: Raw end date.
        granularity: Date granularity label.
        source_text: Text span supporting the normalized time.
        confidence: Confidence score between 0 and 1.
        normalizer: Name of the component that produced the payload.

    Returns:
        Normalized event-time dictionary with validated role, date range,
        granularity, source text, confidence, and normalizer.
    """
    role = role if role in EVENT_TIME_ROLES else "unknown"
    granularity = granularity if granularity in EVENT_TIME_GRANULARITIES else "unknown"
    start = parse_date(start_date)
    end = parse_date(end_date)
    if start and end:
        try:
            if pd.to_datetime(end) < pd.to_datetime(start):
                start, end = end, start
        except Exception:
            pass
    elif start and not end:
        end = start
    elif end and not start:
        start = end
    try:
        confidence_f = float(confidence)
    except Exception:
        confidence_f = 0.0
    confidence_f = max(0.0, min(1.0, confidence_f))
    return {
        "role": role,
        "start_date": start,
        "end_date": end,
        "granularity": granularity,
        "source_text": clean_text(source_text)[:500],
        "confidence": confidence_f,
        "normalizer": normalizer,
    }


def normalize_llm_event_time(value: Any) -> Dict[str, Any]:
    """Validate an LLM-provided structured event-time object.

    Args:
        value: Raw event-time value from an LLM output.

    Returns:
        Normalized event-time dictionary. Invalid input returns the default
        unknown event-time payload.
    """
    if not isinstance(value, dict):
        return event_time_dict()
    normalizer = clean_text(value.get("normalizer", "")) or "llm"
    return event_time_dict(
        role=clean_text(value.get("role", "unknown")),
        start_date=clean_text(value.get("start_date", "")),
        end_date=clean_text(value.get("end_date", "")),
        granularity=clean_text(value.get("granularity", "unknown")),
        source_text=clean_text(value.get("source_text", "")),
        confidence=value.get("confidence", 0.0),
        normalizer=normalizer,
    )
