from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from financial_ekg.config import EVENT_TEMPORAL_COLUMNS
from financial_ekg.utils.dates import normalize_llm_event_time, parse_date
from financial_ekg.utils.serialization import json_dumps, json_loads_maybe
from financial_ekg.utils.text import clean_text, sha1_short

def discover_extraction_dirs(chunks_parent_dir: str) -> List[str]:
    """Find extraction chunk directories under one parent directory.

    Args:
        chunks_parent_dir: Parent directory containing chunk outputs.

    Returns:
        Sorted list of directories that contain `events.csv`.
    """
    parent = Path(chunks_parent_dir)
    if not parent.exists():
        raise FileNotFoundError(f"Chunk parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Chunk parent path is not a directory: {parent}")

    def sort_key(path: Path) -> Tuple[str, int, str]:
        """Create a stable sort key for chunk directories.

        Args:
            path: Candidate chunk directory path.

        Returns:
            Tuple of text prefix, numeric suffix, and full directory name.
        """
        match = re.search(r"(\d+)$", path.name)
        numeric_suffix = int(match.group(1)) if match else -1
        prefix = re.sub(r"\d+$", "", path.name)
        return (prefix, numeric_suffix, path.name)

    dirs = [
        str(p)
        for p in sorted(parent.iterdir(), key=sort_key)
        if p.is_dir() and (p / "articles.csv").exists() and (p / "events.csv").exists()
    ]
    if not dirs:
        raise FileNotFoundError(f"No chunk folders with articles.csv and events.csv found in {parent}")
    return dirs


def ensure_event_temporal_columns(events: pd.DataFrame, articles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Ensure events contain separate article availability time and event time.

    Older extraction chunks used `date` for the article publication date. This
    function upgrades them by parsing temporal_expression/event_description into
    event_date_start/event_date_end and then setting `date` to the semantic event
    start date when one is available.

    Args:
        events: Event dataframe loaded from one or more extraction chunks.
        articles: Optional article dataframe used to recover publication dates.

    Returns:
        Copy of the event dataframe with all temporal columns populated and
        normalized where possible.
    """
    if events is None:
        return pd.DataFrame()
    df = events.copy()
    for col in EVENT_TEMPORAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if df.empty:
        return df

    article_dates: Dict[str, str] = {}
    if articles is not None and len(articles) > 0 and "article_id" in articles.columns:
        date_col = "_date" if "_date" in articles.columns else ("date" if "date" in articles.columns else "")
        if date_col:
            article_dates = {
                clean_text(row.get("article_id", "")): parse_date(row.get(date_col, ""))
                for _, row in articles.iterrows()
                if clean_text(row.get("article_id", ""))
            }

    for idx, row in df.iterrows():
        event_id = clean_text(row.get("event_id", ""))
        article_id = clean_text(row.get("article_id", ""))
        legacy_date = parse_date(row.get("date", ""))
        article_date = (
            parse_date(row.get("article_date", ""))
            or parse_date(row.get("available_date", ""))
            or article_dates.get(article_id, "")
            or legacy_date
        )
        available_date = parse_date(row.get("available_date", "")) or article_date
        df.at[idx, "article_date"] = article_date
        df.at[idx, "available_date"] = available_date

        existing_start = parse_date(row.get("event_date_start", ""))
        existing_end = parse_date(row.get("event_date_end", ""))
        if not existing_start:
            event_time_payload = json_loads_maybe(row.get("event_time_json", ""), {})
            normalized_time = normalize_llm_event_time(event_time_payload if isinstance(event_time_payload, dict) else {})
            existing_start = parse_date(normalized_time.get("start_date", ""))
            existing_end = parse_date(normalized_time.get("end_date", ""))
            df.at[idx, "event_date_start"] = existing_start
            df.at[idx, "event_date_end"] = existing_end
            df.at[idx, "event_time_role"] = clean_text(normalized_time.get("role", "unknown"))
            df.at[idx, "event_time_granularity"] = clean_text(normalized_time.get("granularity", "unknown"))
            df.at[idx, "event_time_source"] = clean_text(normalized_time.get("source_text", ""))
            df.at[idx, "event_time_confidence"] = float(normalized_time.get("confidence", 0.0) or 0.0)
            df.at[idx, "event_time_normalizer"] = clean_text(normalized_time.get("normalizer", ""))
            df.at[idx, "event_time_json"] = json_dumps(normalized_time)
        else:
            df.at[idx, "event_date_start"] = existing_start
            df.at[idx, "event_date_end"] = existing_end or existing_start
            if not clean_text(row.get("event_time_role", "")):
                df.at[idx, "event_time_role"] = "unknown"
            if not clean_text(row.get("event_time_granularity", "")):
                df.at[idx, "event_time_granularity"] = "unknown"

        semantic_event_date = parse_date(df.at[idx, "event_date_start"]) or article_date or legacy_date
        df.at[idx, "date"] = semantic_event_date
        if not event_id:
            event_key = (
                f"{article_id}|{clean_text(row.get('ticker', ''))}|{article_date}|{semantic_event_date}|"
                f"{clean_text(row.get('event_date_end', ''))}|{clean_text(row.get('event_type', ''))}|"
                f"{clean_text(row.get('event_description', ''))[:300]}"
            )
            df.at[idx, "event_id"] = "EVT_" + sha1_short(event_key, 18)
    return df


def load_extracted_chunks(extraction_dirs: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge chunk-level extraction outputs.

    Args:
        extraction_dirs: Directories produced by extract mode.

    Returns:
        Tuple of merged articles dataframe and merged events dataframe.
    """
    article_frames: List[pd.DataFrame] = []
    event_frames: List[pd.DataFrame] = []
    for d in extraction_dirs:
        dpath = Path(d)
        articles_path = dpath / "articles.csv"
        events_path = dpath / "events.csv"
        if not articles_path.exists():
            raise FileNotFoundError(f"Missing {articles_path}. Run --mode extract for this chunk first.")
        if not events_path.exists():
            raise FileNotFoundError(f"Missing {events_path}. Run --mode extract for this chunk first.")
        article_frames.append(pd.read_csv(articles_path, dtype=str, low_memory=False, keep_default_na=False))
        event_frames.append(pd.read_csv(events_path, dtype=str, low_memory=False, keep_default_na=False))

    articles = pd.concat(article_frames, ignore_index=True) if article_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    if len(articles) > 0:
        articles = articles.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
        # Ensure expected columns exist.
        for col in ["_source_row", "_date", "_ticker", "_title", "_article", "_url"]:
            if col not in articles.columns:
                articles[col] = ""

    if len(events) > 0:
        events = events.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
        if "impacted_anchor_companies_json" not in events.columns:
            events["impacted_anchor_companies_json"] = "[]"
        events = ensure_event_temporal_columns(events, articles)
        # Restore numeric types used by feature construction.
        if "importance_score" in events.columns:
            events["importance_score"] = pd.to_numeric(events["importance_score"], errors="coerce").fillna(1).astype(int)
        if "confidence" in events.columns:
            events["confidence"] = pd.to_numeric(events["confidence"], errors="coerce").fillna(0.0).astype(float)

    return articles, events
