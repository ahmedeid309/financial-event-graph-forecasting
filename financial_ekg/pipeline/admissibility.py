from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from financial_ekg.pipeline.evidence import useful_evidence_spans
from financial_ekg.pipeline.normalization import normalize_event_candidate
from financial_ekg.utils.json_repair import extract_json_objects_from_named_array
from financial_ekg.utils.serialization import normalize_extracted_list_field, sanitize_for_json
from financial_ekg.utils.text import clean_text


ACCEPT_SUPPORT_LEVELS = {"direct", "article_inferred"}


def assign_candidate_refs(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach stable C1/C2 references to candidate events."""
    out: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        item["_candidate_ref"] = clean_text(item.get("_candidate_ref", "")) or f"C{idx}"
        out.append(item)
    return out


def span_lookup(stage1_output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return Stage 1 evidence spans keyed by span_id."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for span in useful_evidence_spans(stage1_output):
        if not isinstance(span, dict):
            continue
        span_id = clean_text(span.get("span_id", ""))
        if span_id:
            lookup[span_id] = span
    return lookup


def candidates_for_admissibility_prompt(
    candidates: Sequence[Dict[str, Any]],
    stage1_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build compact candidate records with their exact evidence spans."""
    spans = span_lookup(stage1_output)
    prompt_candidates: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        ref = clean_text(candidate.get("_candidate_ref", "")) or f"C{idx}"
        evidence_span_ids = normalize_extracted_list_field(candidate.get("evidence_span_ids", []), "evidence_span_ids")
        evidence_payload: List[Dict[str, Any]] = []
        for span_id in evidence_span_ids:
            span = spans.get(clean_text(span_id), {})
            if not span:
                continue
            evidence_payload.append(
                {
                    "span_id": clean_text(span.get("span_id", "")),
                    "text": clean_text(span.get("text", "")),
                    "section_id": clean_text(span.get("section_id", "")),
                    "section_type": clean_text(span.get("section_type", "")),
                    "allowed_for_event_extraction": bool(span.get("allowed_for_event_extraction", True)),
                }
            )
        prompt_candidates.append(
            {
                "candidate_id": ref,
                "event_type": clean_text(candidate.get("event_type", "")),
                "event_trigger": clean_text(candidate.get("event_trigger", "")),
                "event_description": clean_text(candidate.get("event_description", "")),
                "main_company": clean_text(candidate.get("main_company", "")),
                "ticker": clean_text(candidate.get("ticker", "")),
                "participants": sanitize_for_json(candidate.get("participants", [])),
                "mentioned_entities": sanitize_for_json(candidate.get("mentioned_entities", [])),
                "financial_metric_mentions": sanitize_for_json(candidate.get("financial_metric_mentions", [])),
                "temporal_expression": clean_text(candidate.get("temporal_expression", "")),
                "evidence_span_ids": evidence_span_ids,
                "evidence_text": clean_text(candidate.get("evidence_text", "")),
                "evidence_spans": evidence_payload,
            }
        )
    return prompt_candidates


def extract_candidate_decisions(admissibility_output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read evidence-admissibility decisions from an LLM output."""
    output = dict(admissibility_output)
    decisions = output.get("candidate_decisions", [])
    if not isinstance(decisions, list) or not decisions:
        recovered = extract_json_objects_from_named_array(output.get("_raw_output", ""), "candidate_decisions")
        if recovered:
            output["candidate_decisions"] = recovered
            output["_recovered_partial_json"] = True
            decisions = recovered
    if not isinstance(decisions, list):
        decisions = []

    normalized: List[Dict[str, Any]] = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        decision = clean_text(item.get("decision", "")).lower()
        support_level = clean_text(item.get("support_level", "")).lower()
        normalized.append(
            {
                "candidate_id": clean_text(item.get("candidate_id", "") or item.get("candidate_ref", "")),
                "decision": decision if decision in {"accept", "reject"} else "reject",
                "support_level": support_level if support_level else "unsupported",
                "section_type": clean_text(item.get("section_type", "")),
                "canonical_source_quote": clean_text(item.get("canonical_source_quote", "")),
                "reject_reason": clean_text(item.get("reject_reason", "")),
                "confidence": item.get("confidence", 0.0),
            }
        )
    return normalized, output


def apply_candidate_decisions(
    candidates: Sequence[Dict[str, Any]],
    decisions: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Filter candidates using evidence-admissibility decisions."""
    decision_by_ref = {
        clean_text(item.get("candidate_id", "")): item
        for item in decisions
        if isinstance(item, dict) and clean_text(item.get("candidate_id", ""))
    }
    accepted: List[Dict[str, Any]] = []
    rejected = 0
    missing = 0
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        ref = clean_text(candidate.get("_candidate_ref", "")) or f"C{idx}"
        decision = decision_by_ref.get(ref)
        if decision is None:
            missing += 1
            continue
        is_accept = (
            clean_text(decision.get("decision", "")).lower() == "accept"
            and clean_text(decision.get("support_level", "")).lower() in ACCEPT_SUPPORT_LEVELS
        )
        if not is_accept:
            rejected += 1
            continue
        out = normalize_event_candidate(candidate)
        out["_candidate_ref"] = ref
        out["_evidence_decision"] = "accept"
        out["_evidence_support_level"] = clean_text(decision.get("support_level", ""))
        out["_evidence_section_type"] = clean_text(decision.get("section_type", ""))
        out["_evidence_admissibility_confidence"] = decision.get("confidence", 0.0)
        quote = clean_text(decision.get("canonical_source_quote", ""))
        if quote:
            out["evidence_text"] = quote
        accepted.append(out)
    return accepted, rejected, missing
