from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from financial_ekg.pipeline.evidence import useful_evidence_spans
from financial_ekg.utils.json_repair import extract_json_objects_from_named_array
from financial_ekg.utils.serialization import sanitize_for_json
from financial_ekg.utils.text import clean_text


ALLOWED_SECTION_TYPES = {
    "core_article_body",
    "financial_analysis",
    "unknown",
}

SECTION_TYPES = {
    "core_article_body",
    "financial_analysis",
    "author_disclosure",
    "publisher_promotion",
    "legal_disclaimer",
    "related_links",
    "metadata_or_footer",
    "unknown",
}


def normalize_section_type(value: Any) -> str:
    """Normalize a raw LLM section type to the evidence-first ontology."""
    section_type = clean_text(value).lower()
    return section_type if section_type in SECTION_TYPES else "unknown"


def section_allowed(section: Dict[str, Any]) -> bool:
    """Return whether a section is allowed to support extraction."""
    section_type = normalize_section_type(section.get("section_type", "unknown"))
    raw_allowed = section.get("allowed_for_event_extraction", section_type in ALLOWED_SECTION_TYPES)
    if isinstance(raw_allowed, str):
        raw_allowed = raw_allowed.strip().lower() in {"true", "yes", "1"}
    return bool(raw_allowed) and section_type in ALLOWED_SECTION_TYPES


def extract_sections_from_output(section_output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract article-section records from an LLM output."""
    output = dict(section_output)
    sections = output.get("article_sections", [])
    if not isinstance(sections, list) or not sections:
        recovered = extract_json_objects_from_named_array(output.get("_raw_output", ""), "article_sections")
        if recovered:
            output["article_sections"] = recovered
            output["_recovered_partial_json"] = True
            sections = recovered
    if not isinstance(sections, list):
        sections = []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(sections, start=1):
        if not isinstance(item, dict):
            continue
        section_type = normalize_section_type(item.get("section_type", "unknown"))
        section = {
            "section_id": clean_text(item.get("section_id", "")) or f"SEC{idx}",
            "section_type": section_type,
            "allowed_for_event_extraction": section_allowed({**item, "section_type": section_type}),
            "start_quote": clean_text(item.get("start_quote", "")),
            "end_quote": clean_text(item.get("end_quote", "")),
            "section_summary": clean_text(item.get("section_summary", "")),
        }
        normalized.append(section)
    return normalized, output


def _find_quote_bounds(article: str, start_quote: str, end_quote: str) -> Optional[Tuple[int, int]]:
    """Find approximate section character bounds from short anchor quotes."""
    article_text = clean_text(article)
    start = -1
    end = -1
    if start_quote:
        start = article_text.find(clean_text(start_quote))
    if end_quote:
        end_quote_clean = clean_text(end_quote)
        search_start = start if start >= 0 else 0
        end = article_text.find(end_quote_clean, search_start)
        if end >= 0:
            end += len(end_quote_clean)
    if start >= 0 and end >= start:
        return start, end
    if start >= 0:
        return start, min(len(article_text), start + 4000)
    if end >= 0:
        return max(0, end - 4000), end
    return None


def sections_with_bounds(sections: Sequence[Dict[str, Any]], article: str) -> List[Dict[str, Any]]:
    """Attach approximate text bounds to classifier sections."""
    bounded: List[Dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        out = dict(section)
        bounds = _find_quote_bounds(
            article,
            clean_text(out.get("start_quote", "")),
            clean_text(out.get("end_quote", "")),
        )
        if bounds is not None:
            out["_start_char"], out["_end_char"] = bounds
        bounded.append(out)
    return bounded


def _find_span_bounds(article: str, span_text: str) -> Optional[Tuple[int, int]]:
    """Find a span in article text, with short-prefix fallback."""
    article_text = clean_text(article)
    text = clean_text(span_text)
    if not article_text or not text:
        return None
    start = article_text.find(text)
    if start >= 0:
        return start, start + len(text)
    prefix = text[:160]
    if len(prefix) >= 40:
        start = article_text.find(prefix)
        if start >= 0:
            return start, start + len(text)
    return None


def _overlap_size(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Return character overlap between two half-open intervals."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def find_section_for_span(
    article: str,
    sections: Sequence[Dict[str, Any]],
    span_text: str,
) -> Dict[str, Any]:
    """Find the best section for a Stage 1 evidence span."""
    bounded = sections_with_bounds(sections, article)
    span_bounds = _find_span_bounds(article, span_text)
    if span_bounds is not None:
        best: Optional[Dict[str, Any]] = None
        best_overlap = 0
        for section in bounded:
            if "_start_char" not in section or "_end_char" not in section:
                continue
            overlap = _overlap_size(span_bounds, (int(section["_start_char"]), int(section["_end_char"])))
            if overlap > best_overlap:
                best = section
                best_overlap = overlap
        if best is not None and best_overlap > 0:
            return best

    lower_span = clean_text(span_text).lower()
    for section in bounded:
        start_quote = clean_text(section.get("start_quote", "")).lower()
        end_quote = clean_text(section.get("end_quote", "")).lower()
        if start_quote and (start_quote in lower_span or lower_span in start_quote):
            return section
        if end_quote and (end_quote in lower_span or lower_span in end_quote):
            return section

    return {
        "section_id": "UNKNOWN",
        "section_type": "unknown",
        "allowed_for_event_extraction": True,
        "section_summary": "No classifier section matched this evidence span.",
    }


def annotate_stage1_with_sections(
    stage1_output: Dict[str, Any],
    article: str,
    sections: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Copy Stage 1 and add section metadata to each useful evidence span."""
    out = dict(stage1_output)
    annotated: List[Dict[str, Any]] = []
    for span in useful_evidence_spans(stage1_output):
        span_text = clean_text(span.get("text", ""))
        section = find_section_for_span(article, sections, span_text)
        item = dict(span)
        item["section_id"] = clean_text(section.get("section_id", "UNKNOWN")) or "UNKNOWN"
        item["section_type"] = normalize_section_type(section.get("section_type", "unknown"))
        item["allowed_for_event_extraction"] = section_allowed(section)
        item["section_summary"] = clean_text(section.get("section_summary", ""))
        annotated.append(item)
    out["financial_evidence_spans"] = annotated
    out["_evidence_section_annotation_count"] = len(annotated)
    out["_evidence_allowed_evidence_span_count"] = sum(1 for item in annotated if item.get("allowed_for_event_extraction"))
    return out


def stage1_allowed_for_event_blocks(stage1_output: Dict[str, Any]) -> Dict[str, Any]:
    """Return a Stage 1 copy containing only extraction-allowed evidence spans."""
    out = dict(stage1_output)
    allowed: List[Dict[str, Any]] = []
    for span in useful_evidence_spans(stage1_output):
        if bool(span.get("allowed_for_event_extraction", True)):
            allowed.append(sanitize_for_json(span))
    out["financial_evidence_spans"] = allowed
    out["_evidence_disallowed_evidence_span_count"] = len(useful_evidence_spans(stage1_output)) - len(allowed)
    return out
