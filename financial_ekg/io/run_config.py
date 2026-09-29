from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from financial_ekg.config import EVENT_TYPES

def save_config(args: argparse.Namespace, out_dir: Path, articles: pd.DataFrame, event_df: pd.DataFrame) -> None:
    """Save run configuration and summary metadata.

    Args:
        args: Parsed CLI arguments.
        out_dir: Output directory.
        articles: Article dataframe for this run.
        event_df: Event dataframe for this run.

    Returns:
        None.
    """
    cfg = vars(args).copy()
    cfg.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "num_articles_processed": int(len(articles)),
            "num_events_extracted": int(len(event_df)),
            "event_types": EVENT_TYPES,
        }
    )
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
