#!/usr/bin/env python3
"""Join exported GNN embeddings into TimeXer stock CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from build_temporal_dataset import normalize_ticker, parse_date, parse_tickers


def load_price_rows(stock_dir: str | Path, ticker: str) -> List[Dict[str, object]]:
    """Read one ticker's OHLCV history, sorted by date.

    Lives here rather than in build_temporal_dataset because the GNN is news-only
    and never touches prices; this script is the boundary where price data
    re-enters, on its way to the downstream forecaster.
    """

    stock_path = Path(stock_dir)
    target = normalize_ticker(ticker)
    path = stock_path / f"{target}.csv"
    if not path.exists():
        matches = [p for p in stock_path.glob("*.csv") if p.stem.upper() == target]
        if not matches:
            raise FileNotFoundError(f"Missing stock CSV for {target} in {stock_path}")
        path = matches[0]

    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        # "adj close" carries a space in these files; normalise once.
        field_map = {
            field: field.strip().lower().replace(" ", "_") for field in reader.fieldnames or []
        }
        for raw in reader:
            row = {field_map[k]: v for k, v in raw.items() if k is not None}
            row_date = parse_date(row.get("date", ""))
            if row_date is None:
                continue
            try:
                close = float(row["close"])
                rows.append({
                    "date": row_date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": close,
                    "adj_close": float(row.get("adj_close", close) or close),
                    "volume": float(row["volume"]),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda item: item["date"])
    if not rows:
        raise ValueError(f"No usable stock rows in {path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create graph-enhanced TimeXer stock CSVs.")
    parser.add_argument("--stock_dir", default="dataset/full_history")
    parser.add_argument("--embeddings_csv", default="gnn_outputs/daily_ticker_embeddings.csv")
    parser.add_argument("--output_dir", default="TimeXer/dataset/stock")
    parser.add_argument(
        "--tickers",
        default="",
        help="Optional comma-separated subset. Empty uses every ticker in the embedding CSV.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=0,
        help=(
            "Reduce the embedding to this many dimensions before joining, fitting the "
            "projection on training rows only. 0 keeps every dimension. Use this to "
            "separate 'the embedding is uninformative' from 'too many columns for the "
            "downstream sample size'."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_graph",
        help="Output filename suffix, so different training objectives do not overwrite each other.",
    )
    parser.add_argument(
        "--graph_target_lead",
        type=int,
        default=0,
        help=(
            "Attach the embedding from this many later price rows. A value of 1 "
            "puts target-day news beside the final historical price row used to "
            "forecast that target day's close."
        ),
    )
    return parser.parse_args()


def load_embeddings(path: str | Path) -> Tuple[Dict[str, Dict[object, Dict[str, str]]], List[str], object, object]:
    embeddings_by_ticker: Dict[str, Dict[object, Dict[str, str]]] = defaultdict(dict)
    dates = []
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        gnn_cols = sorted(
            [name for name in fieldnames if name.startswith("gnn_")],
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not gnn_cols:
            raise ValueError(f"{path} does not contain gnn_* embedding columns")
        for row in reader:
            ticker = normalize_ticker(row.get("ticker", ""))
            row_date = parse_date(row.get("date", ""))
            if not ticker or row_date is None:
                continue
            embeddings_by_ticker[ticker][row_date] = row
            dates.append(row_date)
    if not dates:
        raise ValueError(f"{path} contains no usable embedding rows")
    return embeddings_by_ticker, gnn_cols, min(dates), max(dates)


def reduce_embeddings(
    embeddings_by_ticker: Dict[str, Dict[object, Dict[str, str]]],
    gnn_cols: List[str],
    components: int,
) -> Tuple[Dict[str, Dict[object, Dict[str, str]]], List[str]]:
    """Project the embedding onto its leading training-set principal components."""

    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("scikit-learn and NumPy are required for --pca_components.") from exc

    keys = [(ticker, day) for ticker, days in embeddings_by_ticker.items() for day in days]
    if not keys:
        return embeddings_by_ticker, gnn_cols
    matrix = np.asarray(
        [[float(embeddings_by_ticker[t][d][c]) for c in gnn_cols] for t, d in keys],
        dtype=np.float64,
    )
    train_mask = np.asarray(
        [embeddings_by_ticker[t][d].get("split", "") == "train" for t, d in keys], dtype=bool
    )
    if not train_mask.any():
        raise ValueError("Cannot fit the PCA projection without training rows.")
    components = min(int(components), matrix.shape[1], int(train_mask.sum()))
    projection = PCA(n_components=components, random_state=0).fit(matrix[train_mask])
    reduced = projection.transform(matrix)
    new_cols = [f"gnn_{index}" for index in range(components)]
    for position, (ticker, day) in enumerate(keys):
        row = embeddings_by_ticker[ticker][day]
        for index, name in enumerate(new_cols):
            row[name] = str(float(reduced[position, index]))
    explained = float(projection.explained_variance_ratio_.sum())
    print(f"PCA: {len(gnn_cols)} -> {components} dims, explained variance {explained:.3f}")
    return embeddings_by_ticker, new_cols


def main() -> None:
    args = parse_args()
    if int(args.graph_target_lead) < 0:
        raise ValueError("--graph_target_lead must be non-negative.")
    embeddings_by_ticker, gnn_cols, min_date, max_date = load_embeddings(args.embeddings_csv)
    if int(args.pca_components) > 0:
        embeddings_by_ticker, gnn_cols = reduce_embeddings(
            embeddings_by_ticker, gnn_cols, int(args.pca_components)
        )
    tickers = parse_tickers(args.tickers) or sorted(embeddings_by_ticker)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_values = {col: 0.0 for col in gnn_cols}

    for ticker in tickers:
        price_rows = load_price_rows(args.stock_dir, ticker)
        price_rows = [
            row for row in price_rows if min_date <= row["date"] <= max_date
        ]
        rows = []
        for row_index, stock_row in enumerate(price_rows):
            row_date = stock_row["date"]
            embedding_index = row_index + int(args.graph_target_lead)
            embedding_date = (
                price_rows[embedding_index]["date"]
                if embedding_index < len(price_rows)
                else None
            )
            embedding_row = embeddings_by_ticker.get(ticker, {}).get(embedding_date)
            gnn_values = {
                col: float(embedding_row[col]) if embedding_row and embedding_row.get(col, "") != "" else zero_values[col]
                for col in gnn_cols
            }
            out_row = {
                "date": row_date.isoformat(),
                "open": stock_row["open"],
                "high": stock_row["high"],
                "low": stock_row["low"],
                "adj_close": stock_row["adj_close"],
                "volume": stock_row["volume"],
                **gnn_values,
                "close": stock_row["close"],
            }
            rows.append(out_row)

        output_path = output_dir / f"{ticker}{args.suffix}.csv"
        fieldnames = ["date", "open", "high", "low", "adj_close", "volume", *gnn_cols, "close"]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"Wrote {len(rows)} rows to {output_path} "
            f"(graph_target_lead={int(args.graph_target_lead)})"
        )


if __name__ == "__main__":
    main()
