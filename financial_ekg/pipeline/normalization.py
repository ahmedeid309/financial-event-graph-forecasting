from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from financial_ekg.config import MIRRORED_RELATION_DEDUPE_EVENT_TYPES
from financial_ekg.utils.dates import normalize_llm_event_time
from financial_ekg.utils.serialization import json_dumps, normalize_extracted_list_field
from financial_ekg.utils.text import clean_text, normalize_entity_name, normalize_event_type_value, normalize_ticker


def normalize_event_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one event candidate to the supported event schema.

    Args:
        candidate: Raw event-like dictionary from an LLM stage or repair path.

    Returns:
        Event dictionary with normalized scalar fields, list fields, event time,
        and ontology event type.
    """
    impacted_anchor_companies = candidate.get("impacted_anchor_companies", [])
    if not isinstance(impacted_anchor_companies, list):
        impacted_anchor_companies = []
    out = {
        "event_type": normalize_event_type_value(candidate.get("event_type", "other")),
        "event_trigger": clean_text(candidate.get("event_trigger", "")),
        "event_description": clean_text(candidate.get("event_description", "")),
        "main_company": clean_text(candidate.get("main_company", "")),
        "ticker": normalize_ticker(candidate.get("ticker", "")),
        "participants": normalize_extracted_list_field(candidate.get("participants", []), "participants"),
        "mentioned_entities": normalize_extracted_list_field(candidate.get("mentioned_entities", []), "mentioned_entities"),
        "financial_metric_mentions": normalize_extracted_list_field(
            candidate.get("financial_metric_mentions", []),
            "financial_metric_mentions",
        ),
        "temporal_expression": clean_text(candidate.get("temporal_expression", "")),
        "event_time": normalize_llm_event_time(candidate.get("event_time", {})),
        "explicit_causal_relations": normalize_extracted_list_field(
            candidate.get("explicit_causal_relations", []),
            "explicit_causal_relations",
        ),
        "related_event_mentions": normalize_extracted_list_field(
            candidate.get("related_event_mentions", []),
            "related_event_mentions",
        ),
        "impacted_anchor_companies": impacted_anchor_companies,
        "evidence_span_ids": normalize_extracted_list_field(candidate.get("evidence_span_ids", []), "evidence_span_ids"),
        "importance_score": candidate.get("importance_score", 1),
        "confidence": candidate.get("confidence", 0.0),
    }
    evidence_text = clean_text(candidate.get("evidence_text", ""))
    if evidence_text:
        out["evidence_text"] = evidence_text
    return out


def is_publisher_disclosure_or_promo_event(candidate: Dict[str, Any]) -> bool:
    """Detect publisher, author-disclosure, and newsletter-promotion candidates.

    Args:
        candidate: Event-like dictionary to inspect.

    Returns:
        True when the candidate appears to describe promotional or disclosure
        boilerplate rather than an article-supported financial event.
    """
    parts = [
        candidate.get("event_description", ""),
        candidate.get("event_trigger", ""),
        candidate.get("main_company", ""),
        candidate.get("evidence_text", ""),
        " ".join(normalize_extracted_list_field(candidate.get("participants", []), "participants")),
        " ".join(normalize_extracted_list_field(candidate.get("mentioned_entities", []), "mentioned_entities")),
    ]
    text = clean_text(" ".join(clean_text(part) for part in parts)).lower()
    if not text:
        return False

    if any(
        phrase in text
        for phrase in [
            "before you buy",
            "see the 10 stocks",
            "top stock picks",
            "stock advisor returns",
            "affiliate compensation",
            "board of directors",
        ]
    ):
        return True
    if "stock advisor" in text and any(
        phrase in text
        for phrase in [
            "top stock pick",
            "top stock picks",
            "best stocks",
            "10 stocks",
            "ten stocks",
            "wasn't one of them",
            "was not one of them",
            "newsletter",
            "market returns",
            "tripled",
        ]
    ):
        return True
    if "stock advisor service" in text and any(token in text for token in ["return", "tripled", "outperformed"]):
        return True
    if re.search(r"\b(?:has|have|holds?|held|owns?|owned)\s+positions?\s+in\b", text):
        return True
    if re.search(r"\bholds?\s+positions?\s+in\s+and\s+recommends\b", text):
        return True
    if re.search(r"\brecommends?\s+(?:long|short)\b.*\b(?:calls?|puts?)\b", text):
        return True
    if re.search(r"\b(?:former|current)\s+ceo\b.*\bmember\b.*\bboard of directors\b", text):
        return True
    return False


def article_company_ticker_lookup(*collections: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Build an article-local company-name to ticker lookup from events.

    Args:
        *collections: Event-like dictionaries from final events and candidate
            pools.

    Returns:
        Mapping from normalized company-name variants to normalized tickers.
    """
    lookup: Dict[str, str] = {}
    for collection in collections:
        if not isinstance(collection, (list, tuple)):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            ticker = normalize_ticker(item.get("ticker", ""))
            company = clean_text(item.get("main_company", "") or item.get("company", ""))
            if ticker and company:
                lookup.setdefault(company.lower(), ticker)
                normalized = normalize_entity_name(company)
                if normalized:
                    lookup.setdefault(normalized.lower(), ticker)
    return lookup


def expand_multi_company_portfolio_events(
    events: Sequence[Dict[str, Any]],
    candidate_pool: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Duplicate merged portfolio-holding rows for each named company/ticker.

    Args:
        events: Final event dictionaries after Stage 5 validation.
        candidate_pool: Candidate dictionaries used to recover article-local
            company-to-ticker mappings.

    Returns:
        Tuple containing normalized events and the number of additional
        portfolio-holding rows added.
    """
    normalized_events = [
        normalize_event_candidate(event)
        for event in events
        if isinstance(event, dict) and not is_publisher_disclosure_or_promo_event(event)
    ]
    company_lookup = article_company_ticker_lookup(normalized_events, candidate_pool)
    if not company_lookup:
        return normalized_events, 0

    expanded: List[Dict[str, Any]] = []
    seen = set()
    added = 0

    def add_event(event: Dict[str, Any]) -> None:
        """Append an event only once according to the local expansion key.

        Args:
            event: Normalized event dictionary to add.

        Returns:
            None.
        """
        key = (
            normalize_event_type_value(event.get("event_type", "")),
            normalize_ticker(event.get("ticker", "")),
            clean_text(event.get("event_description", "")).lower(),
        )
        if key in seen:
            return
        seen.add(key)
        expanded.append(event)

    for event in normalized_events:
        add_event(event)
        if normalize_event_type_value(event.get("event_type", "")) != "portfolio_holding":
            continue
        desc = clean_text(event.get("event_description", ""))
        if not desc:
            continue
        matched: List[Tuple[str, str]] = []
        for company_key, ticker in company_lookup.items():
            if not ticker or ticker == normalize_ticker(event.get("ticker", "")):
                continue
            pattern = rf"(?<![A-Za-z0-9]){re.escape(company_key)}(?![A-Za-z0-9])"
            if re.search(pattern, desc, flags=re.I):
                matched.append((company_key, ticker))
        for company_key, ticker in sorted(set(matched)):
            cloned = dict(event)
            cloned["ticker"] = ticker
            cloned["main_company"] = normalize_entity_name(company_key).title() or company_key
            add_event(cloned)
            added += 1

    return expanded, added


def repair_final_from_stage2(
    article_id: str,
    source_ticker: str,
    date: str,
    stage2_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a final event object directly from Stage 2 candidates.

    Args:
        article_id: Stable article identifier.
        source_ticker: Source ticker from the article row.
        date: Article publication date.
        stage2_output: Stage 2 output containing `event_candidates`.

    Returns:
        Stage 5 shaped dictionary containing normalized candidate events.
    """
    candidates = stage2_output.get("event_candidates", [])
    events: List[Dict[str, Any]] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            repaired = normalize_event_candidate(candidate)
            if clean_text(repaired.get("event_description", "")):
                events.append(repaired)
    return {"article_id": article_id, "ticker": source_ticker, "date": date, "events": events}


def normalize_event_dedupe_text(value: Any) -> str:
    """Normalize text for conservative event duplicate detection.

    Args:
        value: Raw text-like value to normalize.

    Returns:
        Lowercase normalized text key with punctuation and spacing reduced.
    """
    text = clean_text(value).lower().strip().rstrip(".")
    text = re.sub(r"[^a-z0-9%$., ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def append_unique_list_items(base: Dict[str, Any], other: Dict[str, Any], field_name: str) -> None:
    """Merge string-like list fields without duplicating values.

    Args:
        base: Event dictionary updated in place.
        other: Event dictionary providing additional list values.
        field_name: Name of the list field to merge.

    Returns:
        None.
    """
    items = normalize_extracted_list_field(base.get(field_name, []), field_name)
    seen = {normalize_event_dedupe_text(item) for item in items}
    for item in normalize_extracted_list_field(other.get(field_name, []), field_name):
        key = normalize_event_dedupe_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(item)
    base[field_name] = items


def append_unique_raw_items(base: Dict[str, Any], other: Dict[str, Any], field_name: str) -> None:
    """Merge raw list fields such as `impacted_anchor_companies`.

    Args:
        base: Event dictionary updated in place.
        other: Event dictionary providing additional raw values.
        field_name: Name of the raw list field to merge.

    Returns:
        None.
    """
    base_items = base.get(field_name, [])
    other_items = other.get(field_name, [])
    if not isinstance(base_items, list):
        base_items = []
    if not isinstance(other_items, list):
        other_items = []
    seen = {json_dumps(item) for item in base_items}
    for item in other_items:
        key = json_dumps(item)
        if key in seen:
            continue
        seen.add(key)
        base_items.append(item)
    base[field_name] = base_items


def merge_duplicate_event_metadata(base: Dict[str, Any], other: Dict[str, Any]) -> None:
    """Preserve useful relation metadata when dropping a mirrored duplicate.

    Args:
        base: Event dictionary retained in the deduplicated output.
        other: Duplicate event dictionary whose metadata should be merged.

    Returns:
        None.
    """
    other_main = clean_text(other.get("main_company", ""))
    base_main = clean_text(base.get("main_company", ""))
    if other_main and other_main.lower() != base_main.lower():
        participants = normalize_extracted_list_field(base.get("participants", []), "participants")
        seen = {normalize_event_dedupe_text(item) for item in participants}
        key = normalize_event_dedupe_text(other_main)
        if key and key not in seen:
            participants.append(other_main)
        base["participants"] = participants

    for field_name in [
        "participants",
        "mentioned_entities",
        "financial_metric_mentions",
        "explicit_causal_relations",
        "related_event_mentions",
        "evidence_span_ids",
    ]:
        append_unique_list_items(base, other, field_name)
    append_unique_raw_items(base, other, "impacted_anchor_companies")

    try:
        base["importance_score"] = max(int(base.get("importance_score", 1) or 1), int(other.get("importance_score", 1) or 1))
    except Exception:
        pass
    try:
        base["confidence"] = max(float(base.get("confidence", 0.0) or 0.0), float(other.get("confidence", 0.0) or 0.0))
    except Exception:
        pass


def dedupe_mirrored_relation_events(events: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Drop mirrored relationship events that reuse the same trigger/evidence.

    Args:
        events: Event-like dictionaries after final validation and filtering.

    Returns:
        Tuple containing the deduplicated event list and the number of removed
        mirrored duplicates.
    """
    deduped: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str, Tuple[str, ...]], int] = {}
    removed = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        normalized = normalize_event_candidate(event)
        event_type = normalize_event_type_value(normalized.get("event_type", "other"))
        trigger_key = normalize_event_dedupe_text(normalized.get("event_trigger", ""))
        metrics_key = tuple(
            normalize_event_dedupe_text(item)
            for item in normalize_extracted_list_field(normalized.get("financial_metric_mentions", []), "financial_metric_mentions")
        )

        if event_type in MIRRORED_RELATION_DEDUPE_EVENT_TYPES and len(trigger_key) >= 12:
            key = (event_type, trigger_key, metrics_key)
            if key in seen:
                merge_duplicate_event_metadata(deduped[seen[key]], normalized)
                removed += 1
                continue
            seen[key] = len(deduped)
        deduped.append(normalized)

    return deduped, removed


def dedupe_event_like_dicts(items: Sequence[Dict[str, Any]], description_key: str = "event_description") -> List[Dict[str, Any]]:
    """Deduplicate event-like dictionaries by ticker, type, company, and text.

    Args:
        items: Event-like dictionaries to deduplicate.
        description_key: Field name containing the event description text.

    Returns:
        List of event-like dictionaries with duplicate keys removed.
    """
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = clean_text(item.get(description_key, ""))
        if not desc:
            continue
        key = (
            normalize_ticker(item.get("ticker", "")),
            normalize_event_type_value(item.get("event_type", "other")),
            normalize_entity_name(item.get("main_company", "")).lower(),
            desc.lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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
