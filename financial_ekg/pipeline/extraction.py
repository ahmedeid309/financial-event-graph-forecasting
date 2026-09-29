from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x

from financial_ekg.config import EVENT_TYPES
from financial_ekg.extractors.hf import HFExtractor
from financial_ekg.extractors.llama_cpp import LlamaCppExtractor
from financial_ekg.pipeline.evidence_first import run_evidence_first_extraction_for_article
from financial_ekg.utils.anchors import (
    build_anchor_lookup,
    derive_anchor_impacts_for_event,
    filter_supported_participants,
    load_anchor_company_context,
    normalize_impacted_anchor_companies,
    resolve_anchor_ticker,
)
from financial_ekg.utils.dates import normalize_llm_event_time
from financial_ekg.utils.serialization import json_dumps, stringify_extracted_list_item
from financial_ekg.utils.text import clean_text, normalize_ticker, safe_str, sha1_short


def validate_and_flatten_events(
    extraction: Dict[str, Any],
    article_id: str,
    ticker: str,
    date: str,
    title: str,
    url: str,
    anchor_context: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Validate extracted events and convert them into CSV rows.

    Args:
        extraction: Raw extraction object with an `events` list.
        article_id: Stable article identifier.
        ticker: Source row ticker.
        date: Article date.
        title: Article title.
        url: Article URL.
        anchor_context: Optional anchor-company context used to normalize and
            derive impacted anchor companies.

    Returns:
        List of event rows ready for `events.csv`.
    """
    rows = []
    anchor_context = anchor_context or {}
    anchor_lookup = build_anchor_lookup(anchor_context)
    events = extraction.get("events", [])
    if not isinstance(events, list):
        return rows
    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            continue
        event_type = safe_str(ev.get("event_type", "other")).strip()
        if event_type not in EVENT_TYPES:
            event_type = "other"
        desc = clean_text(ev.get("event_description", ""))
        if not desc:
            continue
        ev_ticker = normalize_ticker(ev.get("ticker", ""))
        importance = ev.get("importance_score", 1)
        try:
            importance = int(importance)
        except Exception:
            importance = 1
        importance = max(1, min(3, importance))
        confidence = ev.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        def list_field(name: str) -> List[str]:
            """Normalize one extracted event list field to strings.

            Args:
                name: Field name in the raw event dictionary.

            Returns:
                List of readable string values for the requested field.
            """
            val = ev.get(name, [])
            if isinstance(val, str):
                return [clean_text(val)] if clean_text(val) else []
            if isinstance(val, list):
                out = []
                for item in val:
                    s = stringify_extracted_list_item(item, name)
                    if s:
                        out.append(s)
                return out
            return []

        impacted_anchors = normalize_impacted_anchor_companies(
            ev.get("impacted_anchor_companies", []),
            anchor_context,
            anchor_lookup,
            event_type=event_type,
        )
        event_trigger = clean_text(ev.get("event_trigger", ""))
        participants = list_field("participants")
        mentioned_entities = list_field("mentioned_entities")
        financial_metric_mentions = list_field("financial_metric_mentions")
        normalized_time = normalize_llm_event_time(ev.get("event_time", {}))
        event_date_start = clean_text(normalized_time.get("start_date", ""))
        event_date_end = clean_text(normalized_time.get("end_date", ""))
        semantic_event_date = event_date_start or date
        event_key = (
            f"{article_id}|{ev_ticker}|{date}|{semantic_event_date}|{event_date_end}|"
            f"{event_type}|{idx}|{desc[:300]}"
        )
        event_id = "EVT_" + sha1_short(event_key, 18)
        explicit_causal_relations = list_field("explicit_causal_relations")
        related_event_mentions = list_field("related_event_mentions")
        participant_support_text = " ".join(
            [desc, event_trigger, *explicit_causal_relations, *related_event_mentions]
        )
        participants = filter_supported_participants(
            participants,
            participant_support_text,
            anchor_context,
            anchor_lookup,
        )

        row = {
            "event_id": event_id,
            "article_id": article_id,
            "ticker": ev_ticker,
            "date": semantic_event_date,
            "article_date": date,
            "available_date": date,
            "event_date_start": event_date_start,
            "event_date_end": event_date_end,
            "event_time_role": clean_text(normalized_time.get("role", "unknown")),
            "event_time_granularity": clean_text(normalized_time.get("granularity", "unknown")),
            "event_time_source": clean_text(normalized_time.get("source_text", "")),
            "event_time_confidence": float(normalized_time.get("confidence", 0.0) or 0.0),
            "event_time_normalizer": clean_text(normalized_time.get("normalizer", "")),
            "event_time_json": json_dumps(normalized_time),
            "event_type": event_type,
            "event_trigger": event_trigger,
            "event_description": desc,
            "main_company": clean_text(ev.get("main_company", "")),
            "participants_json": json_dumps(participants),
            "mentioned_entities_json": json_dumps(mentioned_entities),
            "financial_metric_mentions_json": json_dumps(financial_metric_mentions),
            "temporal_expression": clean_text(ev.get("temporal_expression", "")),
            "explicit_causal_relations_json": json_dumps(explicit_causal_relations),
            "related_event_mentions_json": json_dumps(related_event_mentions),
            "impacted_anchor_companies_json": json_dumps(impacted_anchors),
            "importance_score": importance,
            "confidence": confidence,
            "article_title": title,
            "url": url,
        }
        primary_anchor_tickers = {
            primary
            for primary in [
                ev_ticker if ev_ticker in anchor_context else "",
                resolve_anchor_ticker(row["main_company"], ev_ticker, anchor_context, anchor_lookup),
            ]
            if primary
        }
        classified_impacts = [
            impact
            for impact in impacted_anchors
            if normalize_ticker(impact.get("ticker", "")) not in primary_anchor_tickers
        ]
        if classified_impacts:
            row["impacted_anchor_companies_json"] = json_dumps(classified_impacts)
        else:
            row["impacted_anchor_companies_json"] = json_dumps(
                derive_anchor_impacts_for_event(pd.Series(row), anchor_context, anchor_lookup)
            )
        rows.append(row)
    return rows


def run_extraction(args: argparse.Namespace, articles: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Run evidence-first LLM extraction for all loaded articles.

    Args:
        args: Parsed CLI arguments.
        articles: Normalized article dataframe from `load_articles`.
        out_dir: Directory where chunk outputs are written.

    Returns:
        DataFrame containing flattened event rows written to `events.csv`.
    """
    if not hasattr(args, "evidence_judge_batch_size"):
        args.evidence_judge_batch_size = 12
    if not hasattr(args, "temporal_batch_size"):
        args.temporal_batch_size = 8

    raw_path = out_dir / "event_extractions.jsonl"
    events_path = out_dir / "events.csv"
    event_columns = [
        "event_id",
        "article_id",
        "ticker",
        "date",
        "article_date",
        "available_date",
        "event_date_start",
        "event_date_end",
        "event_time_role",
        "event_time_granularity",
        "event_time_source",
        "event_time_confidence",
        "event_time_normalizer",
        "event_time_json",
        "event_type",
        "event_trigger",
        "event_description",
        "main_company",
        "participants_json",
        "mentioned_entities_json",
        "financial_metric_mentions_json",
        "temporal_expression",
        "explicit_causal_relations_json",
        "related_event_mentions_json",
        "impacted_anchor_companies_json",
        "importance_score",
        "confidence",
        "article_title",
        "url",
    ]
    completed_extractions: Dict[str, Dict[str, Any]] = {}
    if args.resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f_raw_read:
            for line in f_raw_read:
                line = line.strip()
                if not line:
                    continue
                try:
                    extraction = json.loads(line)
                except Exception:
                    continue
                article_id = clean_text(extraction.get("article_id", ""))
                if article_id:
                    completed_extractions[article_id] = extraction

    all_event_rows: List[Dict[str, Any]] = []
    if completed_extractions:
        anchor_context = load_anchor_company_context(
            getattr(args, "anchor_tickers", ""),
            getattr(args, "anchor_aliases_json", ""),
        )
        for article_id, extraction in completed_extractions.items():
            all_event_rows.extend(
                validate_and_flatten_events(
                    extraction,
                    article_id,
                    clean_text(extraction.get("ticker", "")),
                    clean_text(extraction.get("date", "")),
                    clean_text(extraction.get("title", "")),
                    clean_text(extraction.get("url", "")),
                    anchor_context,
                )
            )
        checkpoint_df = pd.DataFrame(all_event_rows, columns=event_columns)
        checkpoint_df.to_csv(events_path, index=False)
        print(
            f"Resuming extraction in {out_dir}: found {len(completed_extractions)} completed articles "
            f"and restored {len(checkpoint_df)} event rows from event_extractions.jsonl",
            file=sys.stderr,
            flush=True,
        )

    if args.llm_backend == "llama_cpp":
        extractor = LlamaCppExtractor(
            model_path=args.llama_model_path,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=args.disable_thinking,
            n_ctx=args.llama_n_ctx,
            n_gpu_layers=args.llama_n_gpu_layers,
            n_batch=args.llama_n_batch,
            n_threads=args.llama_n_threads,
            verbose=args.llama_verbose,
        )
    else:
        extractor = HFExtractor(
            model_name=args.llm_model,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=args.disable_thinking,
        )

    anchor_context = load_anchor_company_context(
        getattr(args, "anchor_tickers", ""),
        getattr(args, "anchor_aliases_json", ""),
    )
    raw_mode = "a" if args.resume and raw_path.exists() else "w"
    with raw_path.open(raw_mode, encoding="utf-8") as f_raw:
        for _, row in tqdm(articles.iterrows(), total=len(articles), desc="Extracting events"):
            article_id = row["article_id"]
            if article_id in completed_extractions:
                continue
            ticker = row["_ticker"]
            date = row["_date"]
            title = row["_title"]
            url = row["_url"]

            extraction = run_evidence_first_extraction_for_article(extractor, row, args)

            extraction["article_id"] = article_id
            extraction["ticker"] = ticker
            extraction["date"] = date
            extraction["title"] = title
            extraction["url"] = url
            f_raw.write(json_dumps(extraction) + "\n")
            f_raw.flush()
            os.fsync(f_raw.fileno())

            all_event_rows.extend(validate_and_flatten_events(extraction, article_id, ticker, date, title, url, anchor_context))
            pd.DataFrame(all_event_rows, columns=event_columns).to_csv(events_path, index=False)

    event_df = pd.DataFrame(all_event_rows, columns=event_columns)
    if len(event_df) == 0:
        event_df = pd.DataFrame(columns=event_columns)
    event_df.to_csv(events_path, index=False)
    return event_df
