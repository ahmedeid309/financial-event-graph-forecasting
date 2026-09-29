from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from financial_ekg.utils.dates import event_time_dict, normalize_llm_event_time
from financial_ekg.utils.json_repair import extract_json_objects_from_named_array
from financial_ekg.utils.serialization import normalize_extracted_list_field, sanitize_for_json
from financial_ekg.utils.text import clean_text, normalize_event_type_value, normalize_ticker


def assign_event_refs(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach stable E1/E2 references for evidence-first temporal stages."""
    out: List[Dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        item = dict(event)
        item["_evidence_event_ref"] = clean_text(item.get("_evidence_event_ref", "")) or f"E{idx}"
        out.append(item)
    return out


def events_for_temporal_prompt_evidence(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build compact event records for evidence-first temporal prompts."""
    prompt_events: List[Dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        event_ref = clean_text(event.get("_evidence_event_ref", "")) or f"E{idx}"
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
                "evidence_text": clean_text(event.get("evidence_text", "")),
                "section_type": clean_text(event.get("_evidence_section_type", "") or event.get("section_type", "")),
                "support_level": clean_text(event.get("_evidence_support_level", "") or event.get("support_level", "")),
                "explicit_causal_relations": normalize_extracted_list_field(
                    event.get("explicit_causal_relations", []),
                    "explicit_causal_relations",
                ),
                "related_event_mentions": normalize_extracted_list_field(
                    event.get("related_event_mentions", []),
                    "related_event_mentions",
                ),
            }
        )
    return prompt_events


def extract_temporal_inventories(output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read temporal inventory records from an LLM output."""
    return _extract_named_records(output, "temporal_inventories")


def extract_temporal_roles(output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read temporal role-selection records from an LLM output."""
    return _extract_named_records(output, "temporal_roles")


def extract_temporal_judgments(output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read temporal judge records from an LLM output."""
    return _extract_named_records(output, "temporal_judgments")


def extract_event_times(output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read event_time records from a temporal-normalization output."""
    return _extract_named_records(output, "event_times")


def _extract_named_records(output: Dict[str, Any], key: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read a named list of objects, recovering partial JSON when possible."""
    stage_output = dict(output)
    records = stage_output.get(key, [])
    if not isinstance(records, list) or not records:
        recovered = extract_json_objects_from_named_array(stage_output.get("_raw_output", ""), key)
        if recovered:
            stage_output[key] = recovered
            stage_output["_recovered_partial_json"] = True
            records = recovered
    if not isinstance(records, list):
        records = []
    return [item for item in records if isinstance(item, dict)], stage_output


def apply_temporal_normalization_evidence(
    events: Sequence[Dict[str, Any]],
    event_time_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Attach T3 normalized event_time records to events."""
    updated = [dict(event) for event in events if isinstance(event, dict)]
    ref_to_event = {
        clean_text(event.get("_evidence_event_ref", "")) or f"E{idx}": event
        for idx, event in enumerate(updated, start=1)
    }
    applied = 0
    for record in event_time_records:
        ref = clean_text(record.get("event_id", "") or record.get("event_ref", ""))
        event = ref_to_event.get(ref)
        if event is None:
            continue
        raw_time = record.get("event_time", {})
        if not isinstance(raw_time, dict):
            raw_time = {
                "role": record.get("role", "unknown"),
                "start_date": record.get("start_date", ""),
                "end_date": record.get("end_date", ""),
                "granularity": record.get("granularity", "unknown"),
                "source_text": record.get("source_text", ""),
                "confidence": record.get("confidence", 0.0),
            }
        event_time = normalize_llm_event_time(raw_time)
        event_time["normalizer"] = "llm_temporal_normalizer"
        event["event_time"] = event_time
        event["_temporal_normalization_source"] = "llm_temporal_normalizer"
        if not clean_text(event.get("temporal_expression", "")) and clean_text(event_time.get("source_text", "")):
            event["temporal_expression"] = clean_text(event_time.get("source_text", ""))
        applied += 1
    for event in updated:
        if not isinstance(event.get("event_time", {}), dict):
            event["event_time"] = event_time_dict()
    return updated, applied


def apply_temporal_judgments_evidence(
    events: Sequence[Dict[str, Any]],
    judgments: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Apply T4 judge decisions to event_time fields."""
    updated = [dict(event) for event in events if isinstance(event, dict)]
    ref_to_event = {
        clean_text(event.get("_evidence_event_ref", "")) or f"E{idx}": event
        for idx, event in enumerate(updated, start=1)
    }
    applied = 0
    corrected = 0
    unknown = 0
    for judgment in judgments:
        ref = clean_text(judgment.get("event_id", "") or judgment.get("event_ref", ""))
        event = ref_to_event.get(ref)
        if event is None:
            continue
        old_time = normalize_llm_event_time(event.get("event_time", {}))
        decision = clean_text(judgment.get("decision", "")).lower()
        if decision not in {"accept", "correct", "unknown"}:
            decision = "unknown"
        raw_time = judgment.get("event_time", {})
        if decision == "unknown" or not isinstance(raw_time, dict):
            event_time = event_time_dict()
            event_time["normalizer"] = "llm_temporal_judge"
            unknown += 1
        else:
            event_time = normalize_llm_event_time(raw_time)
            event_time["normalizer"] = "llm_temporal_judge"
            if event_time != {**old_time, "normalizer": "llm_temporal_judge"}:
                corrected += 1
        event["event_time"] = event_time
        event["_temporal_normalization_source"] = "llm_temporal_judge"
        event["_evidence_temporal_judge_decision"] = decision
        event["_evidence_temporal_judge_error_type"] = clean_text(judgment.get("error_type", ""))
        event["_evidence_temporal_judge_reason"] = clean_text(judgment.get("reason", ""))
        if not clean_text(event.get("temporal_expression", "")) and clean_text(event_time.get("source_text", "")):
            event["temporal_expression"] = clean_text(event_time.get("source_text", ""))
        applied += 1
    return updated, applied, corrected, unknown


def compact_temporal_records_by_event(
    records: Sequence[Dict[str, Any]],
    event_ids: Sequence[str],
    key: str,
) -> List[Dict[str, Any]]:
    """Return records matching a batch of event ids."""
    wanted = {clean_text(event_id) for event_id in event_ids if clean_text(event_id)}
    out: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if clean_text(record.get("event_id", "")) in wanted:
            out.append(sanitize_for_json(record))
    return out

