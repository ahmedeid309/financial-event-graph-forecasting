from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from financial_ekg.utils.dates import parse_date
from financial_ekg.utils.text import clean_text, normalize_ticker, sha1_short

def make_article_id(row: pd.Series, index: int) -> str:
    """Create a stable article ID from source metadata.

    Args:
        row: Raw CSV row.
        index: Physical source row index.

    Returns:
        Article ID string with `ART_` prefix.
    """
    for col in ["article_id", "Article_ID", "id", "ID", "Unnamed: 0"]:
        if col in row and clean_text(row[col]):
            return "ART_" + sha1_short(clean_text(row[col]) + "|" + str(index), 14)
    base = "|".join([clean_text(row.get(c, "")) for c in ["Date", "Stock_symbol", "Article_title", "Url"]])
    return "ART_" + sha1_short(base + "|" + str(index), 14)


def load_articles(input_csv: str, limit: Optional[int] = None, start_row: int = 0) -> pd.DataFrame:
    """
    Load only the columns needed for event extraction.

    Args:
        input_csv: Path to the news CSV.
        limit: Optional number of rows to read.
        start_row: Physical CSV data row to start from, excluding the header.

    Returns:
        Normalized article dataframe with internal `_date`, `_ticker`,
        `_title`, `_article`, `_url`, and `article_id` columns.

    start_row and limit are based on physical CSV data rows after the header.
    This is important for large files because --limit now maps to pandas nrows,
    so a 20GB CSV is not fully loaded during small tests.
    """
    usecols = ["Date", "Article_title", "Stock_symbol", "Article", "Url"]
    nrows = limit if limit is not None and limit > 0 else None

    skiprows = None
    if start_row and start_row > 0:
        # Keep header row 0, skip data rows 1..start_row inclusive.
        skiprows = range(1, start_row + 1)

    df = pd.read_csv(
        input_csv,
        usecols=lambda c: c in usecols,
        dtype=str,
        low_memory=False,
        keep_default_na=False,
        nrows=nrows,
        skiprows=skiprows,
    )
    required = ["Date", "Article_title", "Stock_symbol", "Article"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df["_source_row"] = np.arange(start_row, start_row + len(df), dtype=np.int64)
    df["_date"] = df["Date"].apply(parse_date)
    df["_ticker"] = df["Stock_symbol"].apply(normalize_ticker)
    df["_title"] = df["Article_title"].apply(clean_text)
    df["_article"] = df["Article"].apply(clean_text)
    df["_url"] = df["Url"].apply(clean_text) if "Url" in df.columns else ""
    df = df[(df["_date"] != "") & (df["_ticker"] != "") & (df["_article"] != "")].copy()
    df = df.drop_duplicates(subset=["_date", "_ticker", "_title", "_article"]).reset_index(drop=True)
    df["article_id"] = [make_article_id(row, int(row.get("_source_row", i))) for i, row in df.iterrows()]
    return df


def save_articles_snapshot(articles: pd.DataFrame, out_dir: Path) -> None:
    """Write article metadata used by a chunk or global build.

    Args:
        articles: Normalized article dataframe.
        out_dir: Output directory.

    Returns:
        None.
    """
    """Save the exact article metadata needed for the later global graph build phase."""
    cols = ["article_id", "_source_row", "_date", "_ticker", "_title", "_article", "_url"]
    existing = [c for c in cols if c in articles.columns]
    articles[existing].to_csv(out_dir / "articles.csv", index=False)
