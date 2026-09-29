from __future__ import annotations

import json
import math
from typing import Any, List

from financial_ekg.utils.text import clean_text, safe_str


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert values into JSON-safe structures.

    Args:
        obj: Any Python object that may contain numpy or NaN values.

    Returns:
        JSON-serializable version of the object.
    """
    if obj is None:
        return ""
    if isinstance(obj, float) and math.isnan(obj):
        return ""
    try:
        import pandas as pd

        if not isinstance(obj, (list, tuple, dict, set)) and pd.isna(obj):
            return ""
    except Exception:
        pass
    if isinstance(obj, dict):
        return {safe_str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return "" if math.isnan(val) else val
    except Exception:
        pass
    return obj


def json_dumps(obj: Any) -> str:
    """Dump an object to stable JSON.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string with sanitized values.
    """
    return json.dumps(sanitize_for_json(obj), ensure_ascii=False, sort_keys=True, allow_nan=False)


def json_loads_maybe(s: Any, default: Any) -> Any:
    """Parse JSON if possible, otherwise return a default value.

    Args:
        s: JSON string or already parsed value.
        default: Value returned when parsing fails.

    Returns:
        Parsed JSON value or `default`.
    """
    if isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(safe_str(s))
    except Exception:
        return default


def stringify_extracted_list_item(item: Any, field_name: str) -> str:
    """Turn extracted list items into readable strings.

    Args:
        item: Extracted item, often a string or metric dictionary.
        field_name: Name of the event field containing the item.

    Returns:
        Human-readable string representation.
    """
    if isinstance(item, (str, int, float)):
        return clean_text(item)
    if isinstance(item, dict):
        if field_name == "financial_metric_mentions":
            metric = clean_text(item.get("metric", ""))
            value = clean_text(item.get("value", ""))
            period = clean_text(item.get("time_period", ""))
            context = clean_text(item.get("context", "") or item.get("evidence", "") or item.get("text", ""))
            parts = []
            if metric:
                parts.append(metric)
            if value:
                parts.append(value)
            if period:
                parts.append(period)
            text = " | ".join(parts)
            if context:
                text = f"{text} | {context}" if text else context
            if text:
                return text
        for key in ["name", "entity", "company", "person", "organization", "text", "statement", "value"]:
            if clean_text(item.get(key, "")):
                return clean_text(item.get(key, ""))
        return clean_text(json_dumps(item))
    return clean_text(json_dumps(item))


def normalize_extracted_list_field(value: Any, field_name: str) -> List[str]:
    """Normalize an extracted scalar or list field into readable strings.

    Args:
        value: Raw extracted scalar or list value.
        field_name: Name of the event field being normalized.

    Returns:
        List of readable string values for downstream CSV serialization.
    """
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = stringify_extracted_list_item(item, field_name)
        if text:
            out.append(text)
    return out


def json_for_prompt(obj: Any) -> str:
    """Build compact JSON for LLM prompts.

    Args:
        obj: Prior-stage object to include in a prompt.

    Returns:
        Compact JSON string with internal metadata fields removed.
    """

    def strip_internal(value: Any) -> Any:
        """Remove internal metadata keys before prompt serialization.

        Args:
            value: Arbitrary nested value from a prior pipeline stage.

        Returns:
            Sanitized value with dictionary keys beginning with `_` removed.
        """
        if isinstance(value, dict):
            return {
                safe_str(k): strip_internal(v)
                for k, v in value.items()
                if not safe_str(k).startswith("_")
            }
        if isinstance(value, list):
            return [strip_internal(v) for v in value]
        return sanitize_for_json(value)

    return json.dumps(strip_internal(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
