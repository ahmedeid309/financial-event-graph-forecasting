#!/usr/bin/env python3
"""Export a daily text-only news embedding per (ticker, date) for TimeXer.

This is the NO-GRAPH baseline. It answers one question: how much of the
downstream benefit came from extraction + graph + GNN, and how much would a
sentence encoder have delivered on its own?

The article text is encoded directly with the same BGE model the graph uses for
its node features, pooled over the same calendar window the GNN sees, with the
same exponential age decay. No events are extracted, no graph is built, no
price labels are used. Everything downstream of this file -- the join, the
column layout, the TimeXer fusion pathway -- is byte-for-byte the graph arm's.

The output schema is deliberately identical to gnn/export_embeddings.py,
including the `gnn_*` column prefix, because join_embeddings_to_stock.py and
TimeXer's data_loader.py both select the embedding block by that prefix. Keeping
the name means the baseline needs no downstream code change at all; the arm is
identified by the file it came from and by the TimeXer --model_id, not by the
column names.

Run it from the project root so `gnn/` lands on sys.path:

    python gnn/export_text_embeddings.py --output_csv gnn_outputs_text/...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Run this with the project Python environment/venv.") from exc

from build_temporal_dataset import (
    clean_text,
    load_forecast_tasks,
    normalize_ticker,
    parse_date,
    parse_tickers,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from financial_ekg.graph.embeddings import get_text_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export daily text-only (no-graph) ticker embeddings for the TimeXer baseline."
    )
    parser.add_argument("--forecast_tasks", default="gnn_data/forecast_task.csv")
    parser.add_argument("--articles_csv", default="ekg_final/articles.csv")
    parser.add_argument("--output_csv", default="gnn_outputs_text/daily_ticker_text_embeddings.csv")
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated subset. Empty keeps every ticker in the task file.",
    )
    parser.add_argument("--horizon_trading_days", type=int, default=0)
    parser.add_argument("--min_abs_label_return", type=float, default=0.0)

    # These three must match stage 3, or the arms see different news.
    parser.add_argument(
        "--window_days",
        type=int,
        default=1,
        help="Pool articles published in [cutoff - window_days, cutoff], inclusive, "
             "exactly as build_temporal_dataset._active_nodes admits event nodes.",
    )
    parser.add_argument(
        "--decay_lambda",
        type=float,
        default=0.10,
        help="Article weight is exp(-decay_lambda * age_days), matching the GNN's edge decay.",
    )

    parser.add_argument(
        "--embedding_model",
        default="models/bge-large-en-v1.5",
        help="Same local BGE snapshot the graph build used, so the encoder is not a confound.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--expect_dim",
        type=int,
        default=1024,
        help="Refuse to write if the encoder returns another width. get_text_embeddings falls "
             "back to TF-IDF on failure, and a silent fallback would make this a graph-vs-TFIDF "
             "comparison that still looks completely normal. 0 disables the check.",
    )

    parser.add_argument(
        "--max_chars_per_chunk",
        type=int,
        default=1800,
        help="BGE stops at 512 tokens (~2000 chars). Longer articles are split on whitespace "
             "and their chunk vectors averaged, so the baseline reads the whole article instead "
             "of only its opening.",
    )
    parser.add_argument(
        "--max_chunks_per_article",
        type=int,
        default=12,
        help="Cap on chunks per article. 12 x 1800 chars covers well past the 90th percentile "
             "article length in this corpus.",
    )
    parser.add_argument(
        "--include_title",
        action="store_true",
        default=True,
        help="Prepend the article title to its body before chunking (default on).",
    )
    parser.add_argument("--no_include_title", dest="include_title", action="store_false")

    parser.add_argument("--normalization", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--embedding_prefix", default="gnn")
    parser.add_argument(
        "--cache_path",
        default="gnn_data/text_embedding_cache.npz",
        help="Article vectors keyed by a hash of the encoded TEXT and encoder settings, never "
             "by article id or file path, so changing the corpus or the chunking can never "
             "serve a stale vector.",
    )
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------
# articles


def load_articles(
    path: str | Path, include_title: bool
) -> Tuple[Dict[str, Dict[date, List[int]]], List[str]]:
    """Index article text by ticker and publication date.

    Returns (articles_by_ticker_date, texts). Routing is by the ticker the
    article was FILED under, which is all a no-pipeline baseline can know:
    'this article about Intel also moves AMD' is itself an output of the
    extraction stage being held out here.
    """

    article_path = Path(path)
    if not article_path.exists():
        raise FileNotFoundError(f"Article file not found: {article_path}")

    by_ticker: Dict[str, Dict[date, List[int]]] = {}
    texts: List[str] = []
    skipped = 0

    csv.field_size_limit(10 ** 9)  # article bodies are large
    with article_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            ticker = normalize_ticker(row.get("_ticker", ""))
            published = parse_date(row.get("_date", ""))
            body = clean_text(row.get("_article", ""))
            if not ticker or published is None or not body:
                skipped += 1
                continue
            title = clean_text(row.get("_title", "")) if include_title else ""
            index = len(texts)
            texts.append(f"{title}\n\n{body}" if title else body)
            by_ticker.setdefault(ticker, {}).setdefault(published, []).append(index)

    if not texts:
        raise ValueError(f"No usable articles in {article_path}")
    print(f"Articles: {len(texts)} usable, {skipped} skipped, {len(by_ticker)} tickers")
    return by_ticker, texts


# --------------------------------------------------------------------------
# encoding


def chunk_text(text: str, max_chars: int, max_chunks: int) -> List[str]:
    """Split on whitespace into <= max_chars pieces without breaking words."""

    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for word in text.split():
        addition = len(word) + (1 if current else 0)
        if length + addition > max_chars and current:
            chunks.append(" ".join(current))
            if len(chunks) >= max_chunks:
                return chunks
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition
    if current and len(chunks) < max_chunks:
        chunks.append(" ".join(current))
    return chunks


def cache_key(text: str, model_name: str, max_chars: int, max_chunks: int) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(f"|{max_chars}|{max_chunks}|".encode("utf-8"))
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def encode_articles(texts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    """One unit-norm vector per article: mean of its chunk vectors, renormalized."""

    keys = [
        cache_key(text, args.embedding_model, args.max_chars_per_chunk, args.max_chunks_per_article)
        for text in texts
    ]

    cached: Dict[str, np.ndarray] = {}
    cache_path = Path(args.cache_path)
    if not args.no_cache and cache_path.exists():
        with np.load(cache_path) as store:
            cached = {name: store[name] for name in store.files}
        print(f"Cache: {len(cached)} article vectors loaded from {cache_path}")

    todo = [index for index, key in enumerate(keys) if key not in cached]
    if todo:
        chunk_texts: List[str] = []
        chunk_owner: List[int] = []
        for index in todo:
            pieces = chunk_text(texts[index], args.max_chars_per_chunk, args.max_chunks_per_article)
            chunk_texts.extend(pieces)
            chunk_owner.extend([index] * len(pieces))
        print(
            f"Encoding {len(todo)} articles as {len(chunk_texts)} chunks "
            f"({len(chunk_texts) / max(1, len(todo)):.2f} chunks/article)"
        )

        vectors = get_text_embeddings(
            chunk_texts, args.embedding_model, args.device, int(args.batch_size),
            fallback_dim=int(args.expect_dim) or 64,
        )
        if int(args.expect_dim) and int(vectors.shape[1]) != int(args.expect_dim):
            raise SystemExit(
                f"Encoder returned {vectors.shape[1]}-d vectors, expected {args.expect_dim}. "
                "get_text_embeddings silently falls back to TF-IDF when SentenceTransformers "
                "fails, which would turn this into a graph-vs-TFIDF comparison. Fix the model "
                f"path ({args.embedding_model}) or pass --expect_dim 0 to accept this on purpose."
            )

        owner = np.asarray(chunk_owner)
        for index in todo:
            pooled = vectors[owner == index].mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            cached[keys[index]] = (pooled / norm if norm > 1e-12 else pooled).astype(np.float32)

        if not args.no_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **cached)
            print(f"Cache: {len(cached)} article vectors written to {cache_path}")
    else:
        print("Cache: every article already encoded, nothing to do")

    return np.stack([cached[key] for key in keys], axis=0)


# --------------------------------------------------------------------------
# pooling and normalization


def normalize_embeddings(
    embeddings: np.ndarray,
    splits: Sequence[str],
    method: str,
    has_article: np.ndarray,
) -> np.ndarray:
    """Train-split statistics only, as in gnn/export_embeddings.py.

    Two deviations, both forced by the fact that the GNN produces a vector for
    every row while this arm cannot:

    * Statistics come from training rows that actually carry an article. A
      no-news row is an absence, not an observation, and letting a block of
      zeros into the mean would shift the scale of every real row.
    * No-news rows are put back to exact zero afterwards. Z-scoring would
      otherwise map them to -mean/std, turning "no news" into a specific
      non-zero vector rather than the same zero fill join_embeddings_to_stock.py
      writes for a missing day.
    """

    if method == "none" or embeddings.size == 0:
        return embeddings.astype(np.float32)
    train_mask = np.asarray([s == "train" for s in splits], dtype=bool) & has_article
    if not train_mask.any():
        raise ValueError(
            "Cannot z-score embeddings without training rows that carry an article."
        )
    train_values = embeddings[train_mask]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    scaled = ((embeddings - mean) / std).astype(np.float32)
    scaled[~has_article] = 0.0
    return scaled


def main() -> None:
    args = parse_args()
    if int(args.window_days) < 0:
        raise SystemExit("--window_days must be non-negative.")

    # The same loader stage 3 uses, so the baseline covers exactly the GNN's rows.
    tasks = load_forecast_tasks(
        args.forecast_tasks, parse_tickers(args.tickers),
        args.horizon_trading_days, args.min_abs_label_return,
    )
    tickers = sorted({task.ticker for task in tasks})
    print(f"Tasks: {len(tasks)} rows over {len(tickers)} tickers")

    by_ticker, texts = load_articles(args.articles_csv, bool(args.include_title))
    article_vectors = encode_articles(texts, args)
    dim = int(article_vectors.shape[1])
    print(f"Article vectors: {article_vectors.shape}")

    rows: List[Dict[str, object]] = []
    pooled: List[np.ndarray] = []
    empty_rows = 0
    article_counts: List[int] = []

    for task in tasks:
        by_date = by_ticker.get(task.ticker, {})
        indices: List[int] = []
        weights: List[float] = []
        for offset in range(int(args.window_days) + 1):
            day = task.cutoff_date - timedelta(days=offset)
            members = by_date.get(day)
            if not members:
                continue
            weight = float(np.exp(-float(args.decay_lambda) * offset))
            indices.extend(members)
            weights.extend([weight] * len(members))

        if indices:
            weight_array = np.asarray(weights, dtype=np.float32)[:, None]
            vector = (article_vectors[indices] * weight_array).sum(axis=0) / weight_array.sum()
        else:
            # Same convention join_embeddings_to_stock.py uses for a missing day.
            vector = np.zeros((dim,), dtype=np.float32)
            empty_rows += 1

        article_counts.append(len(indices))
        rows.append({
            "task_id": task.task_id,
            "date": task.cutoff_date.isoformat(),
            "ticker": task.ticker,
            "split": task.split,
            "label_return": task.label_return,
            "label_direction": task.label_direction,
            # No classifier produced this arm, so the column stays empty rather
            # than carrying a number nothing computed.
            "direction_probability": "",
        })
        pooled.append(vector.astype(np.float32))

    coverage = 1.0 - (empty_rows / max(1, len(rows)))
    print(
        f"Pooling: {len(rows)} rows, {empty_rows} with no article in the window "
        f"(coverage {coverage:.1%}), mean {np.mean(article_counts):.2f} articles/row"
    )
    if coverage < 0.5:
        print(
            "WARNING: more than half the rows have no article. The task file covers tickers "
            "beyond those filed in the article file; restrict --tickers to the evaluated ones.",
            file=sys.stderr,
        )

    embeddings = normalize_embeddings(
        np.stack(pooled, axis=0), [str(r["split"]) for r in rows], args.normalization,
        np.asarray([count > 0 for count in article_counts], dtype=bool),
    )

    prefix = args.embedding_prefix
    fieldnames = [
        "task_id", "date", "ticker", "split", "label_return", "label_direction",
        "direction_probability",
    ] + [f"{prefix}_{i}" for i in range(dim)]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, embedding in zip(rows, embeddings):
            out = dict(row)
            out.update({f"{prefix}_{i}": float(v) for i, v in enumerate(embedding)})
            writer.writerow(out)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": "text_only_no_graph",
        "rows": len(rows),
        "ticker_count": len(tickers),
        "tickers": tickers,
        "embedding_dim": dim,
        "normalization": args.normalization,
        "output_csv": str(output_path),
        "source": {
            "forecast_tasks": str(args.forecast_tasks),
            "articles_csv": str(args.articles_csv),
            "encoder": args.embedding_model,
        },
        "pooling": {
            "window_days": int(args.window_days),
            "decay_lambda": float(args.decay_lambda),
            "max_chars_per_chunk": int(args.max_chars_per_chunk),
            "max_chunks_per_article": int(args.max_chunks_per_article),
            "include_title": bool(args.include_title),
            "routing": "filed_ticker_only",
        },
        "coverage": {
            "rows_with_articles": len(rows) - empty_rows,
            "rows_without_articles": empty_rows,
            "fraction_covered": coverage,
            "mean_articles_per_row": float(np.mean(article_counts)),
            "max_articles_per_row": int(np.max(article_counts)) if article_counts else 0,
        },
    }
    # Same naming stage 3 uses: <stem>.metadata.json beside the CSV.
    metadata_path = output_path.parent / (output_path.stem + ".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {len(rows)} rows x {dim} dims to {output_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
