from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

from financial_ekg.utils.text import clean_text


def normalize_llm_output_text(text: str) -> str:
    """Make raw decoded model output readable and easier to parse.

    Some reasoning models can return byte-level tokenizer artifacts when special
    tokens are preserved during decoding. DeepSeek-R1-Distill-Qwen, for example,
    may contain `Ċ` for newlines and `Ġ` for spaces around an otherwise valid
    JSON object.

    Args:
        text: Raw model output.

    Returns:
        Output with common tokenizer artifacts and terminal special tokens
        normalized.
    """
    if not text:
        return ""
    replacements = {
        "Ċ": "\n",
        "Ġ": " ",
        "ĉ": "\t",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"<\|endoftext\|>|<｜end of sentence｜>|<｜end▁of▁sentence｜>", "", text)
    return text.strip()


def strip_reasoning_for_json_search(text: str) -> str:
    """Prefer the final answer section after a reasoning block.

    Args:
        text: Raw or normalized model output.

    Returns:
        The likely JSON answer section.
    """
    text = normalize_llm_output_text(text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def iter_fenced_json_candidates(text: str) -> Iterable[str]:
    """Yield JSON-looking code fence bodies from model output.

    Args:
        text: Raw or normalized model output.

    Returns:
        Iterable of content inside markdown code fences marked as JSON or
        untyped.
    """
    text = normalize_llm_output_text(text)
    fence_pattern = re.compile(r"```(?:json|JSON)?\s*(.*?)```", flags=re.S)
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate


def iter_balanced_json_object_strings(text: str) -> Iterable[str]:
    """Yield balanced JSON object substrings while respecting quoted strings.

    Args:
        text: Text that may contain one or more JSON objects.

    Returns:
        Iterable of complete balanced object substrings.
    """
    start: Optional[int] = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def remove_trailing_json_commas(text: str) -> str:
    """Remove trailing commas before JSON object or array closers.

    Args:
        text: JSON-like string to repair.

    Returns:
        String with commas before `}` or `]` removed.
    """
    return re.sub(r",(\s*[}\]])", r"\1", text)


def next_token_is_json_key(text: str, pos: int) -> bool:
    """Return whether a position starts a quoted JSON object key.

    Args:
        text: JSON-like string to inspect.
        pos: Character offset expected to point at a quote.

    Returns:
        True when the quoted token is followed by a colon after optional
        whitespace.
    """
    if pos >= len(text) or text[pos] != '"':
        return False
    i = pos + 1
    escape = False
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            return j < len(text) and text[j] == ":"
        i += 1
    return False


def json_quote_looks_structural(text: str, pos: int) -> bool:
    """Heuristically decide whether a quote closes a JSON string.

    Args:
        text: JSON-like string to inspect.
        pos: Character offset of the quote being classified.

    Returns:
        True when the surrounding characters indicate a structural closing
        quote rather than an unescaped quote inside string content.
    """
    j = pos + 1
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text):
        return True
    nxt = text[j]
    if nxt in ":}]":
        return True
    if nxt == '"':
        return next_token_is_json_key(text, j)
    if nxt != ",":
        return False
    k = j + 1
    while k < len(text) and text[k].isspace():
        k += 1
    if k >= len(text):
        return True
    return text[k] in "\"{[-0123456789tfn"


def insert_missing_json_commas(text: str) -> str:
    """Insert missing commas between completed values and following keys.

    Args:
        text: JSON-like string to repair.

    Returns:
        String with commas inserted before obvious subsequent object keys.
    """
    out: List[str] = []
    in_string = False
    escape = False
    i = 0

    def last_nonspace() -> str:
        """Return the most recent non-whitespace output character.

        Args:
            None.

        Returns:
            Last non-whitespace character already written, or an empty string.
        """
        for ch in reversed(out):
            if not ch.isspace():
                return ch
        return ""

    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            in_string = True
            escape = False
            i += 1
            continue

        if ch.isspace():
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            prev = last_nonspace()
            if (
                j < len(text)
                and next_token_is_json_key(text, j)
                and prev
                and prev not in "{[,:"
            ):
                out.append(",")
            out.append(text[i:j])
            i = j
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def escape_invalid_json_string_chars(text: str) -> str:
    """Repair common LLM JSON string issues without changing structure.

    The local model sometimes emits otherwise valid JSON with unescaped quotes
    inside string values, for example `"text": "the answer is "no" ..."` or
    raw newlines inside strings. This state machine escapes those content
    characters while preserving structural quotes.

    Args:
        text: JSON-like string to repair.

    Returns:
        String with invalid string-content characters escaped where possible.
    """
    out: List[str] = []
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == '"':
                if json_quote_looks_structural(text, i):
                    out.append(ch)
                    in_string = False
                else:
                    out.append('\\"')
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 32:
                out.append(" ")
            else:
                out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escape = False
    return "".join(out)


def json_loads_lenient(text: str) -> Any:
    """Load JSON with deterministic repairs for common LLM formatting errors.

    Args:
        text: JSON-like text to parse.

    Returns:
        Parsed JSON value produced by the first successful repair candidate.
    """
    candidates: List[str] = []
    base = normalize_llm_output_text(text).strip()
    if base:
        candidates.append(base)
        candidates.append(remove_trailing_json_commas(base))
        repaired = escape_invalid_json_string_chars(base)
        candidates.append(repaired)
        candidates.append(remove_trailing_json_commas(repaired))
        comma_repaired = insert_missing_json_commas(repaired)
        candidates.append(comma_repaired)
        candidates.append(remove_trailing_json_commas(comma_repaired))

    seen: set[str] = set()
    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("empty JSON input", text, 0)


def extract_json_string_value(text: str, key: str) -> str:
    """Extract one JSON-like string value by key.

    Args:
        text: JSON-like text containing object fields.
        key: Field name whose string value should be recovered.

    Returns:
        Recovered string value, or an empty string when the key is absent or
        unrecoverable.
    """
    key_match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not key_match:
        return ""
    i = key_match.end()
    out: List[str] = []
    escape = False
    while i < len(text):
        ch = text[i]
        if escape:
            out.append(ch)
            escape = False
        elif ch == "\\":
            out.append(ch)
            escape = True
        elif ch == '"':
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] in ",}]":
                break
            out.append('\\"')
        else:
            out.append(ch)
        i += 1
    try:
        return str(json_loads_lenient('"' + "".join(out) + '"'))
    except Exception:
        return clean_text("".join(out).replace('\\"', '"'))


def extract_json_strings_from_named_array(text: str, array_key: str) -> List[str]:
    """Recover complete string values from a named JSON array.

    Args:
        text: JSON-like text containing the target array.
        array_key: Name of the array field to scan.

    Returns:
        List of recovered string values.
    """
    if not text:
        return []
    text = strip_reasoning_for_json_search(text)
    key_match = re.search(rf'"{re.escape(array_key)}"\s*:\s*\[', text)
    if not key_match:
        return []
    i = key_match.end()
    out: List[str] = []
    while i < len(text):
        while i < len(text) and text[i] not in '"]':
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        start = i
        i += 1
        escape = False
        while i < len(text):
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                candidate = text[start : i + 1]
                try:
                    value = json_loads_lenient(candidate)
                    if isinstance(value, str):
                        out.append(value)
                except Exception:
                    pass
                i += 1
                break
            i += 1
    return out


def extract_partial_string_object_fields(text: str, keys: Sequence[str]) -> Dict[str, str]:
    """Recover string-valued fields from a possibly incomplete JSON object.

    Args:
        text: JSON-like object fragment.
        keys: Field names to recover.

    Returns:
        Dictionary of recovered string fields keyed by requested field name.
    """
    out: Dict[str, str] = {}
    for key in keys:
        value = extract_json_string_value(text, key)
        if value:
            out[key] = value
    return out


def extract_partial_impacted_anchor_companies(text: str) -> List[Dict[str, Any]]:
    """Recover Stage 6 impacted-anchor objects from partial JSON.

    Args:
        text: JSON-like Stage 6 output or object fragment.

    Returns:
        List of recovered impacted-anchor dictionaries.
    """
    complete = extract_json_objects_from_named_array(text, "impacted_anchor_companies")
    if complete:
        return complete

    key_match = re.search(r'"impacted_anchor_companies"\s*:\s*\[', text)
    if not key_match:
        return []
    array_region = text[key_match.end() :]
    starts = list(re.finditer(r'\{\s*"ticker"\s*:', array_region))
    if not starts:
        return []

    fields = [
        "ticker",
        "impact_type",
        "impact_direction",
        "evidence",
        # Kept for backward compatibility with older saved model outputs.
        "company",
        "reasoning",
        "raw_mention",
    ]
    recovered: List[Dict[str, Any]] = []
    for idx, match in enumerate(starts):
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(array_region)
        item_text = array_region[start:end]
        item = extract_partial_string_object_fields(item_text, fields)
        if item.get("ticker") or item.get("company") or item.get("evidence"):
            recovered.append(item)
    return recovered


def extract_stage6_event_impacts_from_text(text: str) -> List[Dict[str, Any]]:
    """Recover Stage 6 event-impact entries from malformed or truncated JSON.

    Args:
        text: Raw Stage 6 model output.

    Returns:
        List of recovered event-impact dictionaries containing event references
        and impacted anchors.
    """
    if not text:
        return []
    text = strip_reasoning_for_json_search(text)
    key_match = re.search(r'"event_impacts"\s*:\s*\[', text)
    if not key_match:
        return []

    region = text[key_match.end() :]
    event_matches = list(re.finditer(r'"event_id"\s*:\s*"', region))
    if not event_matches:
        return []

    recovered: List[Dict[str, Any]] = []
    for idx, match in enumerate(event_matches):
        start = match.start()
        end = event_matches[idx + 1].start() if idx + 1 < len(event_matches) else len(region)
        item_text = region[start:end]
        event_id = extract_json_string_value("{" + item_text, "event_id")
        impacts = extract_partial_impacted_anchor_companies(item_text)
        if event_id and impacts:
            recovered.append({"event_id": event_id, "impacted_anchor_companies": impacts})
    return recovered


def recover_partial_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Recover useful fields from an incomplete or malformed LLM JSON object.

    Args:
        text: Raw or normalized model output.

    Returns:
        Partial object dictionary when recoverable fields are found, otherwise
        `None`.
    """
    if not text:
        return None
    text = strip_reasoning_for_json_search(text)
    recovered: Dict[str, Any] = {}

    for key in [
        "article_id",
        "source_ticker",
        "ticker",
        "date",
        "block_id",
        "article_topic",
        "article_structure_notes",
    ]:
        value = extract_json_string_value(text, key)
        if value:
            recovered[key] = value

    object_arrays = [
        "companies",
        "other_entities",
        "tracked_anchor_company_mentions",
        "financial_evidence_spans",
        "event_candidates",
        "events",
        "event_impacts",
        "event_times",
    ]
    recovered_array_count = 0
    for key in object_arrays:
        items = extract_json_objects_from_named_array(text, key)
        if items:
            recovered[key] = items
            recovered_array_count += len(items)

    if not recovered.get("event_impacts"):
        event_impacts = extract_stage6_event_impacts_from_text(text)
        if event_impacts:
            recovered["event_impacts"] = event_impacts
            recovered_array_count += len(event_impacts)

    string_items = extract_json_strings_from_named_array(text, "event_areas_to_extract")
    if string_items:
        recovered["event_areas_to_extract"] = string_items

    if recovered_array_count or string_items:
        recovered["_recovered_partial_json"] = True
        return recovered
    return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first valid JSON object from an LLM response.

    Args:
        text: Raw model output.

    Returns:
        Parsed JSON object, or `None` if no valid object is found.
    """
    if not text or not text.strip():
        return None

    normalized = normalize_llm_output_text(text)
    answer_section = strip_reasoning_for_json_search(normalized)

    search_regions: List[str] = []
    search_regions.extend(iter_fenced_json_candidates(answer_section))
    search_regions.extend(iter_fenced_json_candidates(normalized))
    search_regions.extend([answer_section, normalized])

    seen: set[str] = set()
    for region in search_regions:
        region = region.strip()
        if not region or region in seen:
            continue
        seen.add(region)
        try:
            obj = json_loads_lenient(region)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        for candidate in iter_balanced_json_object_strings(region):
            try:
                obj = json_loads_lenient(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return recover_partial_json_object(normalized)


def extract_json_objects_from_named_array(text: str, array_key: str) -> List[Dict[str, Any]]:
    """Recover complete objects from a named JSON array.

    Args:
        text: Raw model output.
        array_key: JSON array key to recover from.

    Returns:
        List of complete JSON objects recovered from the array.
    """
    if not text:
        return []
    text = strip_reasoning_for_json_search(text)
    key_match = re.search(rf'"{re.escape(array_key)}"\s*:\s*\[', text)
    if not key_match:
        return []
    i = key_match.end()
    out: List[Dict[str, Any]] = []
    while i < len(text):
        while i < len(text) and text[i] not in "{]":
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        start = i
        depth = 0
        in_string = False
        escape = False
        while i < len(text):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"' and json_quote_looks_structural(text, i):
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json_loads_lenient(candidate)
                            if isinstance(obj, dict):
                                out.append(obj)
                        except Exception:
                            pass
                        i += 1
                        break
            i += 1
        if depth != 0:
            break
    return out


def print_parse_failure_visualization(raw_output: str) -> None:
    """Print a readable parse-debug view for failed LLM generations.

    Args:
        raw_output: Raw decoded model output.

    Returns:
        None.
    """
    readable = normalize_llm_output_text(raw_output)
    json_search_region = strip_reasoning_for_json_search(readable)
    print("\n" + "=" * 36 + " PARSE DEBUG: READABLE MODEL OUTPUT " + "=" * 36, file=sys.stderr)
    print(readable, file=sys.stderr)
    if json_search_region and json_search_region != readable:
        print("\n" + "-" * 35 + " JSON SEARCH REGION AFTER REASONING " + "-" * 35, file=sys.stderr)
        print(json_search_region, file=sys.stderr)
    print("=" * 112 + "\n", file=sys.stderr)
