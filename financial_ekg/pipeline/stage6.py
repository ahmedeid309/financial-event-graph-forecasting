from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from financial_ekg.utils.anchors import (
    build_anchor_lookup,
    normalize_impacted_anchor_companies,
    resolve_anchor_ticker,
)
from financial_ekg.utils.json_repair import (
    extract_json_objects_from_named_array,
    extract_stage6_event_impacts_from_text,
)
from financial_ekg.utils.serialization import normalize_extracted_list_field
from financial_ekg.utils.text import clean_text, normalize_event_type_value, normalize_ticker


def events_for_anchor_impact_prompt(final_events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build compact finalized events for Stage 6 anchor-impact classification.

    Args:
        final_events: Final event dictionaries produced by Stage 5 and later
            normalization stages.

    Returns:
        Compact event dictionaries safe to include in the Stage 6 prompt.
    """
    def compact_current_impacts(value: Any) -> List[Dict[str, str]]:
        """Compact existing anchor-impact values for prompt context.

        Args:
            value: Raw `impacted_anchor_companies` value from an event.

        Returns:
            List of compact impact dictionaries containing only prompt-relevant
            fields.
        """
        if not isinstance(value, list):
            return []
        impacts: List[Dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            compact = {
                "ticker": normalize_ticker(item.get("ticker", "")),
                "impact_type": clean_text(item.get("impact_type", "")),
                "impact_direction": clean_text(item.get("impact_direction", "")),
                "evidence": clean_text(item.get("evidence", "")),
            }
            if any(compact.values()):
                impacts.append(compact)
        return impacts

    prompt_events: List[Dict[str, Any]] = []
    for idx, event in enumerate(final_events, start=1):
        if not isinstance(event, dict):
            continue
        event_ref = clean_text(event.get("_stage6_event_ref", "")) or f"E{idx}"
        prompt_events.append(
            {
                "event_id": event_ref,
                "event_type": normalize_event_type_value(event.get("event_type", "other")),
                "event_trigger": clean_text(event.get("event_trigger", "")),
                "event_description": clean_text(event.get("event_description", "")),
                "main_company": clean_text(event.get("main_company", "")),
                "ticker": normalize_ticker(event.get("ticker", "")),
                "participants": normalize_extracted_list_field(event.get("participants", []), "participants"),
                "mentioned_entities": normalize_extracted_list_field(event.get("mentioned_entities", []), "mentioned_entities"),
                "financial_metric_mentions": normalize_extracted_list_field(
                    event.get("financial_metric_mentions", []),
                    "financial_metric_mentions",
                ),
                "temporal_expression": clean_text(event.get("temporal_expression", "")),
                "explicit_causal_relations": normalize_extracted_list_field(
                    event.get("explicit_causal_relations", []),
                    "explicit_causal_relations",
                ),
                "related_event_mentions": normalize_extracted_list_field(
                    event.get("related_event_mentions", []),
                    "related_event_mentions",
                ),
                "current_impacted_anchor_companies": compact_current_impacts(event.get("impacted_anchor_companies", [])),
                "evidence_text": clean_text(event.get("evidence_text", "")),
            }
        )
    return prompt_events


def extract_event_impacts_from_stage6_output(stage6_output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read Stage 6 event-to-anchor impact classifications.

    Args:
        stage6_output: Parsed or partially parsed Stage 6 LLM output.

    Returns:
        Tuple of recovered event-impact dictionaries and the possibly annotated
        Stage 6 output dictionary.
    """
    stage6_output = dict(stage6_output)
    event_impacts = stage6_output.get("event_impacts", [])
    if not isinstance(event_impacts, list) or not event_impacts:
        recovered = extract_json_objects_from_named_array(stage6_output.get("_raw_output", ""), "event_impacts")
        if not recovered:
            recovered = extract_stage6_event_impacts_from_text(stage6_output.get("_raw_output", ""))
        if recovered:
            stage6_output["event_impacts"] = recovered
            stage6_output["_recovered_partial_json"] = True
            event_impacts = recovered
    if not isinstance(event_impacts, list):
        event_impacts = []
    return [item for item in event_impacts if isinstance(item, dict)], stage6_output


def apply_stage6_anchor_impacts(
    final_events: Sequence[Dict[str, Any]],
    stage6_event_impacts: Sequence[Dict[str, Any]],
    anchor_context: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Replace finalized events' anchor impacts with Stage 6 classifications.

    Args:
        final_events: Final event dictionaries to update.
        stage6_event_impacts: Stage 6 event-impact records from the LLM.
        anchor_context: Configured anchor-company context keyed by ticker.

    Returns:
        Tuple of updated events, number of events with non-empty anchor impacts,
        and total anchor-impact count.
    """
    anchor_lookup = build_anchor_lookup(anchor_context)
    events = [dict(event) for event in final_events if isinstance(event, dict)]
    ref_to_event = {
        clean_text(event.get("_stage6_event_ref", "")) or f"E{idx}": event
        for idx, event in enumerate(events, start=1)
    }
    for event in events:
        event["impacted_anchor_companies"] = []
        event["_anchor_impacts_source"] = "stage6"

    for item in stage6_event_impacts:
        ref = clean_text(item.get("event_id", "") or item.get("event_ref", ""))
        if not ref and item.get("event_index") is not None:
            try:
                ref = f"E{int(item.get('event_index'))}"
            except Exception:
                ref = ""
        event = ref_to_event.get(ref)
        if event is None:
            continue
        impacts = normalize_impacted_anchor_companies(
            item.get("impacted_anchor_companies", []),
            anchor_context,
            anchor_lookup,
            event_type=event.get("event_type", ""),
        )
        primary_anchor_tickers = {
            primary
            for primary in [
                normalize_ticker(event.get("ticker", "")) if normalize_ticker(event.get("ticker", "")) in anchor_context else "",
                resolve_anchor_ticker(
                    event.get("main_company", ""),
                    normalize_ticker(event.get("ticker", "")),
                    anchor_context,
                    anchor_lookup,
                ),
            ]
            if primary
        }
        impacts = [
            impact
            for impact in impacts
            if normalize_ticker(impact.get("ticker", "")) not in primary_anchor_tickers
        ]
        event["impacted_anchor_companies"] = impacts

    impacted_event_count = sum(1 for event in events if event.get("impacted_anchor_companies"))
    impact_count = sum(len(event.get("impacted_anchor_companies", [])) for event in events)
    return events, impacted_event_count, impact_count
