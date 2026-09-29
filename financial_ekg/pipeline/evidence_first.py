from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from financial_ekg.pipeline.evidence import build_adaptive_evidence_blocks, merge_stage1_with_rule_based_spans, useful_evidence_spans
from financial_ekg.pipeline.normalization import (
    dedupe_event_like_dicts,
    dedupe_mirrored_relation_events,
    expand_multi_company_portfolio_events,
    is_publisher_disclosure_or_promo_event,
    normalize_event_candidate,
    repair_final_from_stage2,
)
from financial_ekg.pipeline.stage6 import apply_stage6_anchor_impacts, extract_event_impacts_from_stage6_output
from financial_ekg.utils.anchors import load_anchor_company_context
from financial_ekg.utils.json_repair import extract_json_objects_from_named_array
from financial_ekg.utils.text import clean_text
from financial_ekg.pipeline.admissibility import (
    apply_candidate_decisions,
    assign_candidate_refs,
    candidates_for_admissibility_prompt,
    extract_candidate_decisions,
)
from financial_ekg.pipeline.evidence_prompt_builders import (
    build_anchor_impact_classification_prompt,
    build_article_map_prompt_evidence,
    build_event_block_prompt_evidence,
    build_evidence_admissibility_prompt,
    build_final_consolidation_prompt_evidence,
    build_section_classification_prompt_evidence,
    build_temporal_inventory_prompt_evidence,
    build_temporal_judge_prompt_evidence,
    build_temporal_normalization_prompt_evidence,
    build_temporal_role_selection_prompt_evidence,
)
from financial_ekg.pipeline.sections import annotate_stage1_with_sections, extract_sections_from_output, stage1_allowed_for_event_blocks
from financial_ekg.pipeline.temporal_evidence import (
    apply_temporal_judgments_evidence,
    apply_temporal_normalization_evidence,
    assign_event_refs,
    events_for_temporal_prompt_evidence,
    extract_event_times,
    extract_temporal_inventories,
    extract_temporal_judgments,
    extract_temporal_roles,
)


def run_evidence_first_extraction_for_article(extractor: Any, row: pd.Series, settings: Any) -> Dict[str, Any]:
    """Run evidence-first extraction for one article."""
    article_id = row["article_id"]
    source_ticker = row["_ticker"]
    date = row["_date"]
    title = row["_title"]
    article = row["_article"]
    anchor_context = load_anchor_company_context(settings.anchor_tickers, settings.anchor_aliases_json)

    errors: List[str] = []
    stage_outputs: Dict[str, Any] = {}
    llm_call_count = 0

    def run_stage(stage_name: str, prompt: str, empty_payload: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal llm_call_count
        print(f"[evidence][{stage_name}] article={article_id} start", file=sys.stderr, flush=True)
        llm_call_count += 1
        try:
            output = extractor(prompt, max_new_tokens=settings.max_new_tokens)
        except Exception as exc:
            output = dict(empty_payload)
            output.update({"_parse_error": True, "_error": str(exc), "_raw_output": ""})
            errors.append(f"{stage_name}: {exc}")
        if output.get("_parse_error"):
            errors.append(f"{stage_name}_parse_error")
        print(
            f"[evidence][{stage_name}] article={article_id} "
            f"parse_error={bool(output.get('_parse_error'))} "
            f"generated_tokens={output.get('_generated_token_count', 'na')} "
            f"truncated={output.get('_output_may_be_truncated', False)}",
            file=sys.stderr,
            flush=True,
        )
        return output

    stage1 = run_stage(
        "evidence_article_map",
        build_article_map_prompt_evidence(article_id, source_ticker, date, title, article, anchor_context),
        {
            "article_id": article_id,
            "source_ticker": source_ticker,
            "date": date,
            "article_topic": "",
            "companies": [],
            "other_entities": [],
            "tracked_anchor_company_mentions": [],
            "event_areas_to_extract": [],
            "financial_evidence_spans": [],
            "article_structure_notes": "",
        },
    )
    stage1 = merge_stage1_with_rule_based_spans(stage1, article_id, source_ticker, date, title, article)
    stage_outputs["_evidence_article_map"] = stage1

    section_output = run_stage(
        "evidence_section_classification",
        build_section_classification_prompt_evidence(article_id, source_ticker, date, title, article),
        {"article_id": article_id, "article_sections": []},
    )
    sections, section_output = extract_sections_from_output(section_output)
    section_output["article_sections"] = sections
    stage_outputs["_evidence_section_map"] = section_output

    annotated_stage1 = annotate_stage1_with_sections(stage1, article, sections)
    stage_outputs["_evidence_annotated_article_map"] = annotated_stage1
    extraction_stage1 = stage1_allowed_for_event_blocks(annotated_stage1)

    evidence_blocks = build_adaptive_evidence_blocks(
        extraction_stage1,
        article,
        target_block_chars=settings.adaptive_block_chars,
        max_blocks=settings.adaptive_max_blocks,
    )
    stage_outputs["_evidence_event_blocks"] = {
        "block_count": len(evidence_blocks),
        "blocks": [
            {
                "block_id": block.get("block_id", ""),
                "span_count": block.get("span_count", 0),
                "span_ids": block.get("span_ids", []),
            }
            for block in evidence_blocks
        ],
    }

    block_outputs: List[Dict[str, Any]] = []
    candidate_pool: List[Dict[str, Any]] = []
    for block_index, block in enumerate(evidence_blocks, start=1):
        block_output = run_stage(
            f"evidence_event_block_{block_index}",
            build_event_block_prompt_evidence(
                article_id,
                source_ticker,
                date,
                title,
                annotated_stage1,
                block,
                block_index,
                len(evidence_blocks),
                anchor_context,
            ),
            {
                "article_id": article_id,
                "source_ticker": source_ticker,
                "date": date,
                "block_id": block.get("block_id", ""),
                "event_candidates": [],
            },
        )
        block_candidates, block_output = extract_candidates_from_event_block_output(block_output)
        normalized_candidates = dedupe_event_like_dicts([normalize_event_candidate(c) for c in block_candidates])
        block_output["event_candidates"] = normalized_candidates
        block_outputs.append(block_output)
        candidate_pool.extend(normalized_candidates)
        print(
            f"[evidence][evidence_event_block_{block_index}] article={article_id} candidates={len(normalized_candidates)}",
            file=sys.stderr,
            flush=True,
        )

    candidate_pool = assign_candidate_refs(dedupe_event_like_dicts(candidate_pool))
    stage_outputs["_evidence_candidate_extraction"] = {
        "candidate_count": len(candidate_pool),
        "blocks": block_outputs,
    }

    accepted_candidates, admissibility_stage = run_evidence_admissibility(
        run_stage,
        article_id,
        source_ticker,
        date,
        title,
        candidate_pool,
        annotated_stage1,
        settings,
    )
    stage_outputs["_evidence_admissibility"] = admissibility_stage

    final_events, final_stage = run_final_consolidation_evidence(
        run_stage,
        article_id,
        source_ticker,
        date,
        title,
        annotated_stage1,
        accepted_candidates,
        candidate_pool,
        anchor_context,
    )
    stage_outputs["_evidence_final_consolidation"] = final_stage

    final_events, temporal_stages = run_temporal_pipeline_evidence(
        run_stage,
        article_id,
        source_ticker,
        date,
        title,
        final_events,
        settings,
    )
    stage_outputs.update(temporal_stages)

    final_events, stage6 = run_stage6_evidence(
        run_stage,
        article_id,
        source_ticker,
        date,
        title,
        annotated_stage1,
        final_events,
        anchor_context,
        settings,
    )
    stage_outputs["_evidence_stage6"] = stage6

    final_obj: Dict[str, Any] = {
        "article_id": article_id,
        "ticker": source_ticker,
        "date": date,
        "events": final_events,
        **stage_outputs,
    }

    stage_dicts = flatten_stage_dicts(stage_outputs)
    parse_error_stage_count = sum(1 for stage in stage_dicts if stage.get("_parse_error"))
    output_truncated_stage_count = sum(1 for stage in stage_dicts if stage.get("_output_may_be_truncated"))
    final_obj["_quality_summary"] = {
        "pipeline": "evidence_temporal",
        "stage1_evidence_span_count": len(useful_evidence_spans(stage1)),
        "section_count": len(sections),
        "evidence_annotated_evidence_span_count": int(annotated_stage1.get("_evidence_section_annotation_count", 0) or 0),
        "evidence_allowed_evidence_span_count": int(annotated_stage1.get("_evidence_allowed_evidence_span_count", 0) or 0),
        "adaptive_evidence_block_count": len(evidence_blocks),
        "candidate_pool_count": len(candidate_pool),
        "admissibility_accepted_candidate_count": len(accepted_candidates),
        "admissibility_rejected_candidate_count": int(admissibility_stage.get("_rejected_candidate_count", 0) or 0),
        "admissibility_missing_decision_count": int(admissibility_stage.get("_missing_decision_count", 0) or 0),
        "final_event_count": len(final_events),
        "temporal_inventory_count": int(stage_outputs.get("_evidence_temporal_inventory", {}).get("_record_count", 0) or 0),
        "temporal_role_count": int(stage_outputs.get("_evidence_temporal_role_selection", {}).get("_record_count", 0) or 0),
        "temporal_normalization_applied_event_count": int(stage_outputs.get("_evidence_temporal_normalization", {}).get("_applied_event_count", 0) or 0),
        "temporal_judge_applied_event_count": int(stage_outputs.get("_evidence_temporal_judge", {}).get("_applied_event_count", 0) or 0),
        "temporal_judge_corrected_event_count": int(stage_outputs.get("_evidence_temporal_judge", {}).get("_corrected_event_count", 0) or 0),
        "temporal_judge_unknown_event_count": int(stage_outputs.get("_evidence_temporal_judge", {}).get("_unknown_event_count", 0) or 0),
        "stage6_anchor_impact_event_count": int(stage6.get("_anchor_impact_event_count", 0) or 0),
        "stage6_anchor_impact_count": int(stage6.get("_anchor_impact_count", 0) or 0),
        "llm_call_count": llm_call_count,
        "parse_error_stage_count": parse_error_stage_count,
        "output_truncated_stage_count": output_truncated_stage_count,
    }
    final_obj["_parse_error"] = bool(parse_error_stage_count)
    if errors:
        final_obj["_error"] = "; ".join(dict.fromkeys(errors))

    print(
        f"[evidence][done] article={article_id} final_events={len(final_events)} "
        f"parse_error_stages={parse_error_stage_count} truncated_stages={output_truncated_stage_count}",
        file=sys.stderr,
        flush=True,
    )
    return final_obj


def extract_candidates_from_event_block_output(output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract event_candidates from an evidence-first event-block stage."""
    stage_output = dict(output)
    candidates = stage_output.get("event_candidates", [])
    if not isinstance(candidates, list) or not candidates:
        recovered = extract_json_objects_from_named_array(stage_output.get("_raw_output", ""), "event_candidates")
        if recovered:
            stage_output["event_candidates"] = recovered
            stage_output["_recovered_partial_json"] = True
            candidates = recovered
    if not isinstance(candidates, list):
        candidates = []
    return [item for item in candidates if isinstance(item, dict)], stage_output


def extract_events_from_final_output(output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract final events from the evidence-first final-consolidation stage."""
    stage_output = dict(output)
    events = stage_output.get("events", [])
    if not isinstance(events, list) or not events:
        recovered = extract_json_objects_from_named_array(stage_output.get("_raw_output", ""), "events")
        if recovered:
            stage_output["events"] = recovered
            stage_output["_recovered_partial_json"] = True
            events = recovered
    if not isinstance(events, list):
        events = []
    return [item for item in events if isinstance(item, dict)], stage_output


def normalize_evidence_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an event while preserving evidence-first audit metadata."""
    normalized = normalize_event_candidate(event)
    for source_key, target_key in [
        ("section_type", "_evidence_section_type"),
        ("support_level", "_evidence_support_level"),
        ("_evidence_section_type", "_evidence_section_type"),
        ("_evidence_support_level", "_evidence_support_level"),
        ("_candidate_ref", "_candidate_ref"),
        ("_evidence_decision", "_evidence_decision"),
    ]:
        value = clean_text(event.get(source_key, ""))
        if value:
            normalized[target_key] = value
    if clean_text(event.get("section_type", "")):
        normalized["section_type"] = clean_text(event.get("section_type", ""))
    if clean_text(event.get("support_level", "")):
        normalized["support_level"] = clean_text(event.get("support_level", ""))
    return normalized


def run_evidence_admissibility(
    run_stage: Any,
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    candidate_pool: Sequence[Dict[str, Any]],
    annotated_stage1: Dict[str, Any],
    settings: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the batched evidence-admissibility judge."""
    if not candidate_pool:
        return [], {
            "article_id": article_id,
            "candidate_decisions": [],
            "_skipped": True,
            "_applied": True,
            "_parse_error": False,
        }

    prompt_candidates = candidates_for_admissibility_prompt(candidate_pool, annotated_stage1)
    batch_size = max(1, int(settings.evidence_judge_batch_size or 12))
    batches: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for batch_start in range(0, len(prompt_candidates), batch_size):
        batch = prompt_candidates[batch_start : batch_start + batch_size]
        batch_index = int(batch_start / batch_size) + 1
        batch_count = int(math.ceil(len(prompt_candidates) / batch_size))
        stage_name = "evidence_admissibility" if batch_count == 1 else f"evidence_admissibility_batch_{batch_index}"
        batch_output = run_stage(
            stage_name,
            build_evidence_admissibility_prompt(article_id, source_ticker, date, title, batch),
            {"article_id": article_id, "candidate_decisions": []},
        )
        batch_decisions, batch_output = extract_candidate_decisions(batch_output)
        decisions.extend(batch_decisions)
        batches.append(batch_output)

    stage = {
        "article_id": article_id,
        "candidate_decisions": decisions,
        "batches": batches,
        "_skipped": False,
        "_batch_size": batch_size,
        "_batch_count": len(batches),
        "_candidate_count": len(candidate_pool),
        "_decision_count": len(decisions),
        "_parse_error": any(batch.get("_parse_error") for batch in batches),
        "_output_may_be_truncated": any(batch.get("_output_may_be_truncated", False) for batch in batches),
        "_generated_token_count": sum(int(batch.get("_generated_token_count", 0) or 0) for batch in batches),
    }
    accepted, rejected, missing = apply_candidate_decisions(candidate_pool, decisions)
    stage["_accepted_candidate_count"] = len(accepted)
    stage["_rejected_candidate_count"] = rejected
    stage["_missing_decision_count"] = missing
    stage["_applied"] = not stage["_parse_error"]
    if stage["_parse_error"]:
        fallback = [normalize_event_candidate(candidate) for candidate in candidate_pool]
        for item in fallback:
            item["_evidence_decision"] = "fallback_after_parse_error"
        return fallback, stage
    return accepted, stage


def run_final_consolidation_evidence(
    run_stage: Any,
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    annotated_stage1: Dict[str, Any],
    accepted_candidates: Sequence[Dict[str, Any]],
    candidate_pool: Sequence[Dict[str, Any]],
    anchor_context: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run final event consolidation and evidence-safe final filters."""
    if not accepted_candidates:
        return [], {
            "article_id": article_id,
            "events": [],
            "_skipped": True,
            "_parse_error": False,
            "_output_may_be_truncated": False,
        }

    final_output = run_stage(
        "evidence_final_consolidation",
        build_final_consolidation_prompt_evidence(
            article_id,
            source_ticker,
            date,
            title,
            annotated_stage1,
            accepted_candidates,
            anchor_context,
        ),
        {"article_id": article_id, "ticker": source_ticker, "date": date, "events": []},
    )
    raw_events, final_output = extract_events_from_final_output(final_output)
    final_events = dedupe_event_like_dicts([normalize_evidence_event(event) for event in raw_events])

    if final_output.get("_parse_error") or not final_events:
        repaired = repair_final_from_stage2(
            article_id,
            source_ticker,
            date,
            {"event_candidates": list(accepted_candidates)},
        )
        final_events = dedupe_event_like_dicts([normalize_evidence_event(event) for event in repaired.get("events", [])])
        final_output["_repaired_from_accepted_candidates"] = bool(final_events)

    before_disclosure_filter = len(final_events)
    final_events = [
        event
        for event in final_events
        if isinstance(event, dict) and not is_publisher_disclosure_or_promo_event(event)
    ]
    final_output["_disclosure_promo_removed_count"] = before_disclosure_filter - len(final_events)

    final_events, mirrored_duplicate_removed_count = dedupe_mirrored_relation_events(final_events)
    final_output["_mirrored_duplicate_removed_count"] = mirrored_duplicate_removed_count

    final_events, expanded_portfolio_count = expand_multi_company_portfolio_events(final_events, candidate_pool)
    final_output["_multi_company_portfolio_expanded_count"] = expanded_portfolio_count
    return final_events, final_output


def run_temporal_pipeline_evidence(
    run_stage: Any,
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    final_events: Sequence[Dict[str, Any]],
    settings: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run evidence-first temporal inventory, role selection, normalization, and judge."""
    events = assign_event_refs(final_events)
    empty_stages = {
        "_evidence_temporal_inventory": _empty_temporal_stage(article_id, "temporal_inventories"),
        "_evidence_temporal_role_selection": _empty_temporal_stage(article_id, "temporal_roles"),
        "_evidence_temporal_normalization": _empty_temporal_stage(article_id, "event_times"),
        "_evidence_temporal_judge": _empty_temporal_stage(article_id, "temporal_judgments"),
    }
    if not events:
        return events, empty_stages

    batch_size = max(1, int(settings.temporal_batch_size or 8))
    inventory_batches: List[Dict[str, Any]] = []
    role_batches: List[Dict[str, Any]] = []
    normalization_batches: List[Dict[str, Any]] = []
    judge_batches: List[Dict[str, Any]] = []
    inventories: List[Dict[str, Any]] = []
    roles: List[Dict[str, Any]] = []
    event_times: List[Dict[str, Any]] = []
    judgments: List[Dict[str, Any]] = []

    for batch_start in range(0, len(events), batch_size):
        batch_events = events[batch_start : batch_start + batch_size]
        prompt_events = events_for_temporal_prompt_evidence(batch_events)
        batch_index = int(batch_start / batch_size) + 1
        batch_count = int(math.ceil(len(events) / batch_size))
        suffix = "" if batch_count == 1 else f"_batch_{batch_index}"

        inv_output = run_stage(
            f"evidence_temporal_inventory{suffix}",
            build_temporal_inventory_prompt_evidence(article_id, source_ticker, date, title, prompt_events),
            {"article_id": article_id, "article_date": date, "temporal_inventories": []},
        )
        batch_inventories, inv_output = extract_temporal_inventories(inv_output)
        inventories.extend(batch_inventories)
        inventory_batches.append(inv_output)

        role_output = run_stage(
            f"evidence_temporal_role_selection{suffix}",
            build_temporal_role_selection_prompt_evidence(
                article_id,
                source_ticker,
                date,
                title,
                prompt_events,
                batch_inventories,
            ),
            {"article_id": article_id, "article_date": date, "temporal_roles": []},
        )
        batch_roles, role_output = extract_temporal_roles(role_output)
        roles.extend(batch_roles)
        role_batches.append(role_output)

        norm_output = run_stage(
            f"evidence_temporal_normalization{suffix}",
            build_temporal_normalization_prompt_evidence(
                article_id,
                source_ticker,
                date,
                title,
                prompt_events,
                batch_roles,
            ),
            {"article_id": article_id, "article_date": date, "event_times": []},
        )
        batch_event_times, norm_output = extract_event_times(norm_output)
        event_times.extend(batch_event_times)
        normalization_batches.append(norm_output)

        judge_output = run_stage(
            f"evidence_temporal_judge{suffix}",
            build_temporal_judge_prompt_evidence(
                article_id,
                source_ticker,
                date,
                title,
                prompt_events,
                batch_inventories,
                batch_roles,
                batch_event_times,
            ),
            {"article_id": article_id, "article_date": date, "temporal_judgments": []},
        )
        batch_judgments, judge_output = extract_temporal_judgments(judge_output)
        judgments.extend(batch_judgments)
        judge_batches.append(judge_output)

    events, temporal_applied_count = apply_temporal_normalization_evidence(events, event_times)
    events, judge_applied_count, judge_corrected_count, judge_unknown_count = apply_temporal_judgments_evidence(events, judgments)

    stages = {
        "_evidence_temporal_inventory": make_batched_stage(article_id, "temporal_inventories", inventories, inventory_batches, batch_size),
        "_evidence_temporal_role_selection": make_batched_stage(article_id, "temporal_roles", roles, role_batches, batch_size),
        "_evidence_temporal_normalization": make_batched_stage(article_id, "event_times", event_times, normalization_batches, batch_size),
        "_evidence_temporal_judge": make_batched_stage(article_id, "temporal_judgments", judgments, judge_batches, batch_size),
    }
    stages["_evidence_temporal_normalization"]["_applied_event_count"] = temporal_applied_count
    stages["_evidence_temporal_judge"]["_applied_event_count"] = judge_applied_count
    stages["_evidence_temporal_judge"]["_corrected_event_count"] = judge_corrected_count
    stages["_evidence_temporal_judge"]["_unknown_event_count"] = judge_unknown_count
    return events, stages


def run_stage6_evidence(
    run_stage: Any,
    article_id: str,
    source_ticker: str,
    date: str,
    title: str,
    annotated_stage1: Dict[str, Any],
    final_events: Sequence[Dict[str, Any]],
    anchor_context: Dict[str, Dict[str, Any]],
    settings: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the existing Stage 6 anchor-impact classifier on evidence-first final events."""
    events = [dict(event) for event in final_events if isinstance(event, dict)]
    if not events:
        return events, {
            "article_id": article_id,
            "event_impacts": [],
            "_skipped": True,
            "_parse_error": False,
            "_output_may_be_truncated": False,
        }

    for event_index, event in enumerate(events, start=1):
        event["_stage6_event_ref"] = f"E{event_index}"

    batch_size = max(1, int(settings.stage6_batch_size or 40))
    batches: List[Dict[str, Any]] = []
    impacts: List[Dict[str, Any]] = []
    for batch_start in range(0, len(events), batch_size):
        batch_events = events[batch_start : batch_start + batch_size]
        batch_index = int(batch_start / batch_size) + 1
        batch_count = int(math.ceil(len(events) / batch_size))
        stage_name = "evidence_stage6_anchor_impact" if batch_count == 1 else f"evidence_stage6_anchor_impact_batch_{batch_index}"
        batch_output = run_stage(
            stage_name,
            build_anchor_impact_classification_prompt(
                article_id,
                source_ticker,
                date,
                title,
                annotated_stage1,
                batch_events,
                anchor_context,
            ),
            {"article_id": article_id, "event_impacts": []},
        )
        batch_impacts, batch_output = extract_event_impacts_from_stage6_output(batch_output)
        impacts.extend(batch_impacts)
        batches.append(batch_output)

    stage = {
        "article_id": article_id,
        "event_impacts": impacts,
        "batches": batches,
        "_batch_size": batch_size,
        "_batch_count": len(batches),
        "_parse_error": any(batch.get("_parse_error") for batch in batches),
        "_output_may_be_truncated": any(batch.get("_output_may_be_truncated", False) for batch in batches),
        "_generated_token_count": sum(int(batch.get("_generated_token_count", 0) or 0) for batch in batches),
    }
    if not stage["_parse_error"]:
        events, impacted_event_count, impact_count = apply_stage6_anchor_impacts(events, impacts, anchor_context)
        stage["_applied"] = True
    else:
        impacted_event_count = 0
        impact_count = 0
        stage["_applied"] = False
    stage["_anchor_impact_event_count"] = impacted_event_count
    stage["_anchor_impact_count"] = impact_count
    return events, stage


def _empty_temporal_stage(article_id: str, key: str) -> Dict[str, Any]:
    """Return a skipped temporal stage object."""
    return {
        "article_id": article_id,
        key: [],
        "_skipped": True,
        "_record_count": 0,
        "_parse_error": False,
        "_output_may_be_truncated": False,
    }


def make_batched_stage(
    article_id: str,
    key: str,
    records: Sequence[Dict[str, Any]],
    batches: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, Any]:
    """Build a standard batched-stage metadata object."""
    return {
        "article_id": article_id,
        key: list(records),
        "batches": list(batches),
        "_skipped": False,
        "_record_count": len(records),
        "_batch_size": batch_size,
        "_batch_count": len(batches),
        "_parse_error": any(batch.get("_parse_error") for batch in batches),
        "_output_may_be_truncated": any(batch.get("_output_may_be_truncated", False) for batch in batches),
        "_generated_token_count": sum(int(batch.get("_generated_token_count", 0) or 0) for batch in batches),
    }


def flatten_stage_dicts(stage_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect stage dictionaries and nested batch dictionaries for summary counts."""
    stages: List[Dict[str, Any]] = []
    for value in stage_outputs.values():
        if isinstance(value, dict):
            stages.append(value)
            batches = value.get("batches", [])
            if isinstance(batches, list):
                stages.extend([batch for batch in batches if isinstance(batch, dict)])
        elif isinstance(value, list):
            stages.extend([item for item in value if isinstance(item, dict)])
    return stages
