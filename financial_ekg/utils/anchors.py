from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from financial_ekg.config import (
    ANCHOR_IMPACT_DIRECTION_SCORE,
    ANCHOR_IMPACT_DIRECTIONS,
    ANCHOR_IMPACT_EDGE_BY_TYPE,
    ANCHOR_IMPACT_TYPES,
    DEFAULT_ANCHOR_COMPANIES,
    EVENT_COMPANY_EDGE_BY_TYPE,
    TEXT_AMBIGUOUS_ANCHOR_TICKERS,
)
from financial_ekg.utils.serialization import json_loads_maybe
from financial_ekg.utils.text import (
    clean_text,
    entity_resolution_keys,
    normalize_entity_name,
    normalize_event_type_value,
    normalize_ticker,
    safe_str,
)

def parse_anchor_tickers(value: Any) -> List[str]:
    """Parse configured tracked tickers.

    Args:
        value: Comma-separated ticker string or iterable.

    Returns:
        Ordered normalized tickers.
    """
    raw_values: List[Any]
    if isinstance(value, str):
        raw_values = re.split(r"[,;\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = []

    tickers: List[str] = []
    seen = set()
    for raw in raw_values:
        ticker = normalize_ticker(raw)
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def load_anchor_company_context(anchor_tickers: Any, aliases_json: str = "") -> Dict[str, Dict[str, Any]]:
    """Load the canonical tracked-company context used for anchor resolution.

    The context is intentionally ticker-first. Name cleanup is still useful for
    display and fallback matching, but the ticker is the canonical graph ID for
    tracked forecasting targets.

    Args:
        anchor_tickers: Comma-separated ticker string or iterable of tickers.
        aliases_json: Optional JSON file containing company and alias overrides.

    Returns:
        Dictionary keyed by anchor ticker with company name and aliases.
    """
    tickers = parse_anchor_tickers(anchor_tickers)
    if not tickers:
        return {}

    configured: Dict[str, Any] = {}
    aliases_path = Path(safe_str(aliases_json).strip()) if clean_text(aliases_json) else None
    if aliases_path:
        try:
            with aliases_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                configured = loaded
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict) and normalize_ticker(item.get("ticker", "")):
                        configured[normalize_ticker(item.get("ticker", ""))] = item
        except Exception as exc:
            print(f"WARNING: Could not load anchor alias JSON {aliases_path}: {exc}", file=sys.stderr)

    context: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        base = DEFAULT_ANCHOR_COMPANIES.get(ticker, {"company": ticker, "aliases": [ticker]})
        override = configured.get(ticker, {})
        if isinstance(override, list):
            override = {"aliases": override}
        if not isinstance(override, dict):
            override = {}

        company = clean_text(override.get("company", "")) or clean_text(base.get("company", "")) or ticker
        aliases: List[str] = []
        alias_items: List[Any] = [company, *list(base.get("aliases", [])), *list(override.get("aliases", []))]
        if ticker not in TEXT_AMBIGUOUS_ANCHOR_TICKERS:
            alias_items.insert(0, ticker)
        for item in alias_items:
            alias = clean_text(item)
            if alias and alias.lower() not in {x.lower() for x in aliases}:
                aliases.append(alias)
        context[ticker] = {"ticker": ticker, "company": company, "aliases": aliases}
    return context


def anchor_context_for_prompt(anchor_context: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact anchor context for extraction prompts.

    Args:
        anchor_context: Full anchor-company context keyed by ticker.

    Returns:
        Sorted list of prompt-safe anchor dictionaries with ticker, company,
        and aliases.
    """
    return [
        {
            "ticker": ticker,
            "company": clean_text(record.get("company", "")) or ticker,
            "aliases": [clean_text(alias) for alias in record.get("aliases", []) if clean_text(alias)][:12],
        }
        for ticker, record in sorted(anchor_context.items())
    ]


def build_anchor_lookup(anchor_context: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Build conservative alias keys mapping raw text to tracked tickers.

    Args:
        anchor_context: Full anchor-company context keyed by ticker.

    Returns:
        Lookup mapping lowercase alias keys to canonical anchor tickers.
    """
    lookup: Dict[str, str] = {}

    def add_key(key: Any, ticker: str) -> None:
        """Add one cleaned alias key to the anchor lookup.

        Args:
            key: Raw alias or company-name value.
            ticker: Canonical anchor ticker for the alias.

        Returns:
            None.
        """
        cleaned = clean_text(key).lower()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,")
        if cleaned:
            lookup.setdefault(cleaned, ticker)

    for ticker, record in anchor_context.items():
        if ticker not in TEXT_AMBIGUOUS_ANCHOR_TICKERS:
            add_key(ticker, ticker)
        add_key(record.get("company", ""), ticker)
        for alias in record.get("aliases", []):
            add_key(alias, ticker)
            for key in entity_resolution_keys(alias, ticker):
                if ticker in TEXT_AMBIGUOUS_ANCHOR_TICKERS and key == ticker.lower():
                    continue
                add_key(key, ticker)
    return lookup


def detect_anchor_tickers_in_text(text: Any, anchor_context: Dict[str, Dict[str, Any]]) -> List[str]:
    """Detect tracked-company aliases in free text.

    Args:
        text: Text to scan for anchor aliases.
        anchor_context: Full anchor-company context keyed by ticker.

    Returns:
        Ordered list of anchor tickers explicitly mentioned in the text.
    """
    haystack = clean_text(text)
    if not haystack:
        return []
    found: List[str] = []
    for ticker, record in anchor_context.items():
        aliases = [record.get("company", ""), *record.get("aliases", [])]
        if ticker not in TEXT_AMBIGUOUS_ANCHOR_TICKERS:
            aliases.insert(0, ticker)
        for alias in aliases:
            alias_s = clean_text(alias)
            if not alias_s:
                continue
            pattern = rf"(?<![A-Za-z0-9]){re.escape(alias_s)}(?![A-Za-z0-9])"
            if re.search(pattern, haystack, flags=re.I):
                if ticker not in found:
                    found.append(ticker)
                break
    return found


def text_mentions_entity(text: Any, entity: Any) -> bool:
    """Return whether text explicitly names an extracted entity.

    Args:
        text: Text to search.
        entity: Raw entity mention to look for.

    Returns:
        True when the text contains the raw or normalized entity mention.
    """
    haystack = clean_text(text)
    needle = clean_text(entity)
    if not haystack or not needle:
        return False
    candidates = [needle]
    normalized = normalize_entity_name(needle)
    if normalized and normalized.lower() != needle.lower():
        candidates.append(normalized)
    for candidate in candidates:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])"
        if re.search(pattern, haystack, flags=re.I):
            return True
    return False


def text_mentions_anchor(text: Any, ticker: str, anchor_context: Dict[str, Dict[str, Any]]) -> bool:
    """Return whether text explicitly names a tracked anchor company.

    Args:
        text: Text to search.
        ticker: Anchor ticker to test.
        anchor_context: Full anchor-company context keyed by ticker.

    Returns:
        True when the text contains an alias for the requested anchor ticker.
    """
    return ticker in detect_anchor_tickers_in_text(text, anchor_context)


def resolve_anchor_ticker(
    name: Any,
    ticker: Any,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
) -> str:
    """Resolve a raw name or ticker to a tracked anchor ticker when possible.

    Args:
        name: Raw company/entity name.
        ticker: Raw ticker value associated with the entity.
        anchor_context: Full anchor-company context keyed by ticker.
        anchor_lookup: Alias lookup produced by `build_anchor_lookup`.

    Returns:
        Canonical anchor ticker, or an empty string when no anchor matches.
    """
    ticker_norm = normalize_ticker(ticker)
    if ticker_norm in anchor_context:
        return ticker_norm
    for key in entity_resolution_keys(name, ticker_norm):
        if key in anchor_lookup:
            return anchor_lookup[key]
    detected = detect_anchor_tickers_in_text(name, anchor_context)
    return detected[0] if detected else ""


def normalize_anchor_impact_type(value: Any, event_type: Any = "") -> str:
    """Map free-text impact labels into the anchor-impact ontology.

    Args:
        value: Raw impact type from the LLM or a heuristic.
        event_type: Optional event type used as contextual evidence.

    Returns:
        Normalized anchor-impact type from `ANCHOR_IMPACT_TYPES`.
    """
    key = re.sub(r"[^a-z0-9]+", "_", clean_text(value).lower()).strip("_")
    etype = normalize_event_type_value(event_type)
    text = f"{key} {etype}".lower()

    if any(token in text for token in ["direct", "involves_ticker", "same_company"]):
        return "direct"
    if any(token in text for token in ["compet", "rival", "market_share", "peer"]):
        return "competitor" if "market_share" not in text and "peer" not in text else "peer_comparison"
    if any(token in text for token in ["supplier", "customer", "supply_chain", "vendor", "client"]):
        return "supplier_customer"
    if any(token in text for token in ["partner", "partnership"]):
        return "partner"
    if any(token in text for token in ["ecosystem", "platform", "segment", "product"]):
        return "ecosystem"
    if any(token in text for token in ["portfolio", "holding", "weight"]):
        return "portfolio"
    if any(token in text for token in ["macro", "inflation", "interest", "recession", "market_commentary"]):
        return "macro"
    if any(token in text for token in ["regulatory", "lawsuit", "legal"]):
        return "regulatory"
    if any(token in text for token in ["sentiment", "stock_performance", "options"]):
        return "market_sentiment"
    if any(token in text for token in ["mention", "related"]):
        return "mentioned"
    return key if key in ANCHOR_IMPACT_TYPES else "other"


def normalize_anchor_impact_direction(value: Any) -> str:
    """Map free-text anchor impact direction into a strict vocabulary.

    Args:
        value: Raw direction string from the LLM or a heuristic.

    Returns:
        One of `positive`, `negative`, or `neutral`.
    """
    key = re.sub(r"[^a-z0-9]+", "_", clean_text(value).lower()).strip("_")
    if key in ANCHOR_IMPACT_DIRECTIONS:
        return key
    if not key:
        return "neutral"

    if any(
        token in key
        for token in [
            "neutral",
            "mixed",
            "uncertain",
            "unknown",
            "unclear",
            "ambiguous",
            "indirect",
            "none",
            "no_clear",
        ]
    ):
        return "neutral"
    if any(
        token in key
        for token in [
            "positive",
            "up",
            "upward",
            "increase",
            "increased",
            "benefit",
            "beneficial",
            "good",
            "favorable",
            "favourable",
            "tailwind",
            "bullish",
            "opportunity",
            "gain",
            "growth",
            "improve",
        ]
    ):
        return "positive"
    if any(
        token in key
        for token in [
            "negative",
            "down",
            "downward",
            "decrease",
            "decreased",
            "bad",
            "harm",
            "harmful",
            "adverse",
            "unfavorable",
            "unfavourable",
            "headwind",
            "bearish",
            "risk",
            "pressure",
            "decline",
            "loss",
        ]
    ):
        return "negative"
    return "neutral"


def normalize_anchor_impact_direction_with_context(
    value: Any,
    ticker: str,
    company: str,
    impact_type: str,
    evidence: str,
) -> str:
    """Normalize anchor impact direction with supplier/customer context.

    Args:
        value: Raw direction value.
        ticker: Anchor ticker receiving the impact.
        company: Anchor company display name.
        impact_type: Raw or normalized impact type.
        evidence: Evidence text supporting the impact.

    Returns:
        Normalized direction, including the existing supplier-shortage
        correction heuristic when applicable.
    """
    direction = normalize_anchor_impact_direction(value)
    normalized_type = normalize_anchor_impact_type(impact_type)
    if normalized_type != "supplier_customer":
        return direction

    evidence_text = clean_text(evidence).lower()
    company_text = clean_text(company).lower()
    if not evidence_text or not company_text or company_text not in evidence_text:
        return direction

    demand_signal = any(
        phrase in evidence_text
        for phrase in [
            "high demand",
            "strong demand",
            "overwhelming demand",
            "demand from",
            "demand-driven",
            "backlog",
            "line out the door",
            "many other customers",
            "can't keep up with demand",
            "cannot keep up with demand",
            "can not keep up with demand",
            "not able to keep up with demand",
        ]
    )
    supplier_shortage_signal = any(
        phrase in evidence_text
        for phrase in [
            "cannot supply enough",
            "can't supply enough",
            "can not supply enough",
            "cannot deliver enough",
            "can't deliver enough",
            "can not deliver enough",
            "not able to get enough",
            "not being able to get enough",
            "can't keep up",
            "cannot keep up",
            "can not keep up",
        ]
    )
    explicit_supplier_harm = any(
        phrase in evidence_text
        for phrase in [
            "lost customer",
            "lost customers",
            "lost revenue",
            "lost sales",
            "cancelled orders",
            "canceled orders",
            "order cancellations",
            "penalty",
            "penalties",
            "reputational harm",
            "reputation damage",
            "demand weakness",
            "weak demand",
            "revenue decline",
            "sales decline",
        ]
    )
    company_is_supplier_subject = bool(
        re.search(
            rf"\b{re.escape(company_text)}\b[^.]*\b(?:cannot|can't|can not|not able to|isn't able to|unable to)\b[^.]*\b(?:supply|deliver|keep up|get enough|meet)",
            evidence_text,
        )
        or bool(
            re.search(
                rf"\b{re.escape(company_text)}\b[^.]*\bnot being able to\b[^.]*\b(?:supply|deliver|keep up|get enough|meet)",
                evidence_text,
            )
        )
    )
    if company_is_supplier_subject and demand_signal and supplier_shortage_signal and not explicit_supplier_harm:
        return "positive"
    return direction


def anchor_impact_direction_score(value: Any) -> float:
    """Convert an anchor-impact direction to a numeric edge weight.

    Args:
        value: Raw or normalized impact direction.

    Returns:
        Floating-point score used as the signed graph edge weight.
    """
    direction = normalize_anchor_impact_direction(value)
    return float(ANCHOR_IMPACT_DIRECTION_SCORE[direction])


def normalize_impacted_anchor_companies(
    raw: Any,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
    event_type: Any = "",
) -> List[Dict[str, str]]:
    """Normalize extracted anchor-impact objects to ticker-keyed dictionaries.

    Args:
        raw: Raw impacted-anchor field from an event or LLM output.
        anchor_context: Full anchor-company context keyed by ticker.
        anchor_lookup: Alias lookup produced by `build_anchor_lookup`.
        event_type: Optional event type used to normalize impact labels.

    Returns:
        List of normalized impacted-anchor dictionaries keyed by ticker.
    """
    if not anchor_context:
        return []
    if isinstance(raw, str):
        parsed = json_loads_maybe(raw, [])
        raw_items = parsed if isinstance(parsed, list) else [raw]
    elif isinstance(raw, list):
        raw_items = raw
    elif isinstance(raw, dict):
        raw_items = [raw]
    else:
        raw_items = []

    impacts: List[Dict[str, str]] = []
    seen = set()
    for item in raw_items:
        ticker = ""
        company = ""
        impact_type = ""
        impact_direction = ""
        evidence = ""
        reasoning = ""
        raw_mention = ""

        if isinstance(item, dict):
            ticker = normalize_ticker(item.get("ticker", ""))
            company = clean_text(item.get("company", "") or item.get("name", "") or item.get("entity", ""))
            raw_mention = clean_text(item.get("raw_mention", "") or company)
            impact_type = clean_text(item.get("impact_type", "") or item.get("relation", ""))
            impact_direction = clean_text(item.get("impact_direction", "") or item.get("direction", ""))
            evidence = clean_text(item.get("evidence", "") or item.get("reason", "") or item.get("reasoning", ""))
            reasoning = clean_text(item.get("reasoning", "") or item.get("reason", ""))
        else:
            raw_mention = clean_text(item)
            company = raw_mention

        if ticker not in anchor_context:
            ticker = resolve_anchor_ticker(company or raw_mention, ticker, anchor_context, anchor_lookup)
        if not ticker or ticker not in anchor_context:
            for detected in detect_anchor_tickers_in_text(raw_mention or company, anchor_context):
                ticker = detected
                break
        if not ticker or ticker not in anchor_context:
            continue

        impact_type = normalize_anchor_impact_type(impact_type or "mentioned", event_type)
        company = clean_text(anchor_context[ticker].get("company", "")) or ticker
        base_impact_direction = normalize_anchor_impact_direction(impact_direction)
        impact_direction = normalize_anchor_impact_direction_with_context(
            impact_direction,
            ticker,
            company,
            impact_type,
            evidence,
        )
        if impact_direction != base_impact_direction:
            reasoning = (
                "Direction normalized to positive because the evidence describes a demand-driven "
                "supplier shortage for the named anchor supplier without explicit supplier-side harm."
            )
        key = (ticker, impact_type, evidence.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized = {
            "ticker": ticker,
            "company": company,
            "impact_type": impact_type,
            "impact_direction": impact_direction,
            "evidence": evidence,
            "raw_mention": raw_mention,
        }
        if reasoning:
            normalized["reasoning"] = reasoning
        impacts.append(normalized)
    return impacts


def derive_anchor_impacts_for_event(
    ev: Any,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
) -> List[Dict[str, str]]:
    """Derive supported cross-anchor impacts from extracted event fields.

    Main-company/ticker involvement is represented by the normal event-company
    edges, so this function only preserves or derives additional impacted anchor
    companies when the event text states a supported relation.

    Args:
        ev: Event row or dictionary containing extracted event fields.
        anchor_context: Full anchor-company context keyed by ticker.
        anchor_lookup: Alias lookup produced by `build_anchor_lookup`.

    Returns:
        List of supported impacted-anchor dictionaries for non-primary anchors.
    """
    if not anchor_context:
        return []

    event_type = normalize_event_type_value(ev.get("event_type", ""))
    event_ticker = normalize_ticker(ev.get("ticker", ""))
    main_company = clean_text(ev.get("main_company", ""))
    resolved_main = resolve_anchor_ticker(main_company, event_ticker, anchor_context, anchor_lookup)
    primary_anchor_tickers = {
        ticker
        for ticker in [event_ticker if event_ticker in anchor_context else "", resolved_main]
        if ticker
    }
    desc = clean_text(ev.get("event_description", ""))
    support_text = " ".join(
        [
            desc,
            clean_text(ev.get("event_trigger", "")),
            *[clean_text(item) for item in json_loads_maybe(ev.get("explicit_causal_relations_json"), [])],
            *[clean_text(item) for item in json_loads_maybe(ev.get("related_event_mentions_json"), [])],
        ]
    )
    supported_anchor_tickers = set(detect_anchor_tickers_in_text(support_text, anchor_context))

    normalized_llm_impacts = normalize_impacted_anchor_companies(
        json_loads_maybe(ev.get("impacted_anchor_companies_json", "[]"), []),
        anchor_context,
        anchor_lookup,
        event_type=event_type,
    )
    impacts: List[Dict[str, str]] = [
        impact
        for impact in normalized_llm_impacts
        if normalize_ticker(impact.get("ticker", "")) not in primary_anchor_tickers
    ]
    seen = {
        (impact.get("ticker", ""), impact.get("impact_type", ""), impact.get("evidence", ""))
        for impact in impacts
    }

    def add_impact(ticker: str, impact_type: str, evidence: str, raw_mention: str = "") -> None:
        """Append one derived impact when all support checks pass.

        Args:
            ticker: Anchor ticker to add as impacted.
            impact_type: Raw or inferred impact type.
            evidence: Evidence text supporting the impact.
            raw_mention: Optional raw entity mention that resolved to ticker.

        Returns:
            None.
        """
        if not ticker or ticker not in anchor_context:
            return
        if ticker in primary_anchor_tickers:
            return
        if any(impact.get("ticker") == ticker for impact in impacts):
            return
        if ticker not in supported_anchor_tickers:
            return
        normalized_type = normalize_anchor_impact_type(impact_type, event_type)
        key = (ticker, normalized_type, evidence)
        if key in seen:
            return
        seen.add(key)
        impacts.append(
            {
                "ticker": ticker,
                "company": clean_text(anchor_context[ticker].get("company", "")) or ticker,
                "impact_type": normalized_type,
                "impact_direction": "neutral",
                "evidence": evidence,
                "raw_mention": raw_mention,
            }
        )

    def inferred_impact_type() -> str:
        """Infer an anchor-impact type from the current event type.

        Args:
            None.

        Returns:
            Anchor-impact type used for heuristic impacts derived from mentions.
        """
        if event_type == "supply_chain_event":
            return "supplier_customer"
        if event_type in {"competitive_position", "market_share"}:
            return "peer_comparison"
        if event_type == "partnership":
            return "partner"
        if event_type in {"macroeconomic_event", "recession_inflation_interest_rate"}:
            return "macro"
        if event_type in {"lawsuit", "regulatory_investigation"}:
            return "regulatory"
        if event_type == "portfolio_holding":
            return "portfolio"
        if event_type in {"stock_performance", "options_activity", "analyst_upgrade", "analyst_downgrade", "price_target_change"}:
            return "market_sentiment"
        if event_type in {"product_launch", "business_segment_performance"}:
            return "ecosystem"
        return "mentioned"

    entities: List[str] = []
    entities.extend(json_loads_maybe(ev.get("participants_json"), []))
    entities.extend(json_loads_maybe(ev.get("mentioned_entities_json"), []))
    for entity in entities:
        entity_text = clean_text(entity)
        resolved = resolve_anchor_ticker(entity_text, "", anchor_context, anchor_lookup)
        if resolved and text_mentions_anchor(support_text, resolved, anchor_context):
            add_impact(resolved, inferred_impact_type(), desc, entity_text)

    # Fallback scan over the event description catches older rows where an
    # anchor was only included in the prose and not in structured participants.
    for ticker in supported_anchor_tickers:
        add_impact(ticker, inferred_impact_type(), desc, "")
    return impacts


def event_company_edge_type(event_type: Any) -> str:
    """Return a specific Event-to-Company relation for an event type.

    Args:
        event_type: Raw or normalized event type.

    Returns:
        Graph edge type for the event-to-company relation.
    """
    etype = normalize_event_type_value(event_type)
    return EVENT_COMPANY_EDGE_BY_TYPE.get(etype, "EVENT_INVOLVES_COMPANY")


def anchor_impact_edge_type(impact_type: Any) -> str:
    """Return a specific Event-to-anchor relation for an impact type.

    Args:
        impact_type: Raw or normalized anchor-impact type.

    Returns:
        Graph edge type for the event-to-anchor relation.
    """
    normalized = normalize_anchor_impact_type(impact_type)
    return ANCHOR_IMPACT_EDGE_BY_TYPE.get(normalized, "RELATED_TO_ANCHOR")


def reverse_edge_type(edge_type: Any) -> str:
    """Create a readable reverse edge type for heterogeneous message passing.

    Args:
        edge_type: Forward edge type.

    Returns:
        Reverse edge type prefixed with `REV_`.
    """
    clean = re.sub(r"[^A-Z0-9_]+", "_", safe_str(edge_type).upper()).strip("_")
    return f"REV_{clean}" if clean else "REV_RELATED_TO"


def filter_supported_participants(
    participants: Sequence[str],
    support_text: str,
    anchor_context: Dict[str, Dict[str, Any]],
    anchor_lookup: Dict[str, str],
) -> List[str]:
    """Keep participants explicitly supported by final event text.

    Args:
        participants: Candidate participant strings from an event.
        support_text: Event text used as evidence for participant support.
        anchor_context: Full anchor-company context keyed by ticker.
        anchor_lookup: Alias lookup produced by `build_anchor_lookup`.

    Returns:
        Deduplicated list of supported participant strings.
    """
    if not participants:
        return []
    detected_anchors = set(detect_anchor_tickers_in_text(support_text, anchor_context))
    filtered: List[str] = []
    seen: set[str] = set()
    for participant in participants:
        participant_text = clean_text(participant)
        if not participant_text:
            continue
        resolved_anchor = resolve_anchor_ticker(participant_text, "", anchor_context, anchor_lookup)
        supported = text_mentions_entity(support_text, participant_text) or (
            bool(resolved_anchor) and resolved_anchor in detected_anchors
        )
        if not supported:
            continue
        key = participant_text.lower()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(participant_text)
    return filtered
