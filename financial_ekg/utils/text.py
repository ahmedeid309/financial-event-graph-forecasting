from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from financial_ekg.config import (
    EVENT_TYPE_ALIASES,
    EVENT_TYPES,
)

def sha1_short(text: str, n: int = 16) -> str:
    """Create a short deterministic SHA-1 identifier.

    Args:
        text: Text to hash.
        n: Number of hex characters to return.

    Returns:
        Short hexadecimal hash string.
    """
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed.

    Args:
        path: Directory path.

    Returns:
        The directory as a `Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_str(x: Any) -> str:
    """Convert values to strings while treating null-like values as empty.

    Args:
        x: Any scalar value.

    Returns:
        Clean string representation or an empty string for null values.
    """
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    try:
        import pandas as pd

        if not isinstance(x, (list, tuple, dict, set)) and pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def clean_text(text: Any) -> str:
    """Normalize text whitespace.

    Args:
        text: Input text-like value.

    Returns:
        String with collapsed whitespace and trimmed ends.
    """
    s = safe_str(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_ticker(ticker: Any) -> str:
    """Normalize a ticker symbol.

    Args:
        ticker: Raw ticker-like value.

    Returns:
        Uppercase ticker with unsupported characters removed.
    """
    s = safe_str(ticker).strip().upper()
    s = re.sub(r"[^A-Z0-9.\-]", "", s)
    return s


def normalize_event_type_value(event_type: Any) -> str:
    """Normalize an LLM event type to the supported ontology.

    Args:
        event_type: Raw event type from extraction.

    Returns:
        A value from `EVENT_TYPES` when possible, otherwise the cleaned input.
    """
    s = safe_str(event_type).strip()
    key = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    mapped = EVENT_TYPE_ALIASES.get(key, key)
    return mapped if mapped in EVENT_TYPES else s


def normalize_entity_name(name: Any) -> str:
    """Normalize an entity name for graph node deduplication.

    Args:
        name: Raw entity/company name.

    Returns:
        Cleaned entity name.
    """
    s = safe_str(name).strip()
    s = re.sub(r"[\"“”‘’]+", "", s)
    s = re.sub(r"\s*\((?:NASDAQ|NYSE|AMEX|OTC|OTCMKTS|NYSEARCA)\s*:\s*[A-Z0-9.\-]+\)\s*", " ", s, flags=re.I)
    s = re.sub(r"\s*\b(?:NASDAQ|NYSE|AMEX|OTC|OTCMKTS|NYSEARCA)\s*:\s*[A-Z0-9.\-]+\b\s*", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"'s\b", "", s, flags=re.I)
    legal_suffix = re.compile(
        r"(?:,\s*)?\b(?:"
        r"incorporated|corporation|company|limited|holdings|holding|"
        r"inc|corp|co|ltd|plc|llc|lp|sa|ag|nv"
        r")\.?$",
        flags=re.I,
    )
    previous = None
    while previous != s:
        previous = s
        s = legal_suffix.sub("", s).strip()
    s = re.sub(r"\b(Class A|Class B)\b", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9&.\- ]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\s.,-]+$", "", s)
    return s


def entity_resolution_keys(name: Any, ticker: Any = "") -> List[str]:
    """Create conservative lookup keys for resolving entities to companies.

    Args:
        name: Entity or company name.
        ticker: Optional ticker alias.

    Returns:
        Ordered normalized lookup keys.
    """
    keys: List[str] = []

    def add_key(value: Any) -> None:
        """Append a normalized lookup key if it is non-empty and new.

        Args:
            value: Raw text value to convert into an entity-resolution key.

        Returns:
            None.
        """
        key = safe_str(value).strip().lower()
        key = re.sub(r"\s+", " ", key)
        key = re.sub(r"[\s.,-]+$", "", key)
        if key and key not in keys:
            keys.append(key)

    normalized = normalize_entity_name(name)
    add_key(normalized)
    add_key(re.sub(r"\.com$", "", normalized, flags=re.I))
    add_key(re.sub(r"\s+", "", normalized))
    ticker_norm = normalize_ticker(ticker)
    if ticker_norm:
        add_key(ticker_norm)
    return keys


def article_text_for_llm(title: str, article: str) -> str:
    """Build the full article text sent to the LLM.

    Args:
        title: Article title from the CSV.
        article: Full article body from the CSV.

    Returns:
        A single text string containing the title and full article body.
    """
    title = clean_text(title)
    article = clean_text(article)
    if title:
        return f"Title: {title}\n\nArticle: {article}"
    return f"Article: {article}"


def detect_article_ticker_mentions(article: str, source_ticker: str) -> List[str]:
    """Detect ticker symbols mentioned in article text.

    Args:
        article: Full article body.
        source_ticker: Ticker from the CSV row, used as provenance.

    Returns:
        Ordered list of detected tickers, always including the source ticker
        when it is available.
    """
    text = safe_str(article)
    tickers = []
    seen = set()
    patterns = [
        r"\((?:NASDAQ|NYSE|AMEX|OTC|OTCMKTS|NYSEARCA)\s*:\s*([A-Z][A-Z0-9.\-]{0,9})\)",
        r"\b(?:NASDAQ|NYSE|AMEX|OTC|OTCMKTS|NYSEARCA)\s*:\s*([A-Z][A-Z0-9.\-]{0,9})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            ticker = normalize_ticker(match.group(1))
            if ticker and ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    source = normalize_ticker(source_ticker)
    if source and source not in seen:
        tickers.insert(0, source)
    return tickers


def split_article_sentences(text: str) -> List[str]:
    """Split article text into sentences while preserving common abbreviations.

    Args:
        text: Raw article text.

    Returns:
        List of sentence-like strings with protected abbreviations restored.
    """
    protected = clean_text(text)
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Inc.", "Corp.", "Co.", "Ltd.", "LLC.",
        "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.",
        "Oct.", "Nov.", "Dec.", "U.S.", "U.K.",
    ]
    placeholders: Dict[str, str] = {}
    for idx, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR_DOT_{idx}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, abbr.replace(".", placeholder))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", protected) if s.strip()]
    restored: List[str] = []
    for sentence in sentences:
        for placeholder in placeholders:
            sentence = sentence.replace(placeholder, ".")
        restored.append(sentence.strip())
    return restored
