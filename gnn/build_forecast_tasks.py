#!/usr/bin/env python3
"""Build chronological next-day direction tasks for the temporal GNN.

Two supervision scopes are supported:

``source_articles``
    One task per trading day of every ticker that authored articles in the
    graph. This reproduces the original single-source-ticker table.

``event_tickers`` (default)
    One task per trading day of every ticker that graph *events* refer to and
    that has a usable price history. The extraction corpus mentions far more
    companies than it was crawled for, so this multiplies the supervision and,
    more importantly, forces the encoder to produce company-specific
    representations instead of a single daily market summary.

Labels are total-return based (``adj close``) and are filtered for the vendor
splice artifacts present in ``dataset/full_history`` (a single-day adjustment
discontinuity shared by many tickers on 2020-07-06).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
FIELDNAMES = [
    "task_id",
    "ticker",
    "cutoff_date",
    "horizon_trading_days",
    "label_return",
    "label_direction",
    "label_abs_move",
    "trailing_abs_median",
    "raw_return",
    "market_return",
    "news_event_count",
    "split",
]


def clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_ticker(value: object) -> str:
    return clean_text(value).upper()


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def parse_tickers(value: str) -> List[str]:
    return sorted({normalize_ticker(t) for t in value.split(",") if normalize_ticker(t)})


def iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        yield from csv.DictReader(handle)


def read_article_bounds(
    articles_path: Path,
    requested_tickers: Sequence[str],
) -> Dict[str, Tuple[str, str]]:
    """Article-date span per *source* ticker."""

    if not articles_path.exists():
        raise FileNotFoundError(f"Graph article table not found: {articles_path}")

    requested = set(requested_tickers)
    dates_by_ticker: Dict[str, List[str]] = defaultdict(list)
    for row in iter_csv_rows(articles_path):
        ticker = normalize_ticker(row.get("_ticker", "") or row.get("source_ticker", "") or row.get("ticker", ""))
        article_date = clean_text(row.get("_date", "") or row.get("date", ""))[:10]
        if not ticker or (requested and ticker not in requested):
            continue
        if len(article_date) == 10:
            dates_by_ticker[ticker].append(article_date)

    if requested:
        missing = sorted(requested - set(dates_by_ticker))
        if missing:
            raise ValueError(
                "Requested tickers have no source articles in the graph: " + ",".join(missing)
            )
    if not dates_by_ticker:
        raise ValueError(f"No source tickers with article dates found in {articles_path}.")
    return {t: (min(d), max(d)) for t, d in dates_by_ticker.items() if d}


def read_event_availability(events_path: Path) -> Dict[str, str]:
    """event_id -> availability date (never a semantic/occurrence date)."""

    if not events_path.exists():
        raise FileNotFoundError(f"Graph event table not found: {events_path}")
    availability: Dict[str, str] = {}
    for row in iter_csv_rows(events_path):
        event_id = clean_text(row.get("event_id", ""))
        if not event_id:
            continue
        day = clean_text(
            row.get("available_date", "") or row.get("article_date", "") or row.get("date", "")
        )[:10]
        if len(day) == 10:
            availability[event_id] = day
    return availability


def read_ticker_news_days(
    graph_dir: Path,
    cache_path: Optional[Path] = None,
) -> Dict[str, Dict[str, int]]:
    """Map ticker -> {availability_date: event_count} from Event->Ticker edges."""

    if cache_path is not None and cache_path.exists():
        cached: Dict[str, Dict[str, int]] = defaultdict(dict)
        for row in iter_csv_rows(cache_path):
            ticker = normalize_ticker(row.get("ticker", ""))
            day = clean_text(row.get("date", ""))[:10]
            if ticker and len(day) == 10:
                cached[ticker][day] = int(float(row.get("event_count", 0) or 0))
        if cached:
            print(f"Loaded ticker news-day cache from {cache_path}", flush=True)
            return dict(cached)

    availability = read_event_availability(graph_dir / "events.csv")
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for edge in iter_csv_rows(graph_dir / "edges.csv"):
        source = clean_text(edge.get("source", ""))
        target = clean_text(edge.get("target", ""))
        if not source.startswith("event:") or not target.startswith("ticker:"):
            continue
        day = availability.get(source.split(":", 1)[1])
        if day is None:
            continue
        counts[normalize_ticker(target.split(":", 1)[1])][day] += 1

    result = {ticker: dict(days) for ticker, days in counts.items()}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["ticker", "date", "event_count"])
            for ticker in sorted(result):
                for day in sorted(result[ticker]):
                    writer.writerow([ticker, day, result[ticker][day]])
        print(f"Wrote ticker news-day cache to {cache_path}", flush=True)
    return result


def load_stock_rows(path: Path, price_column: str) -> List[Tuple[str, float]]:
    """Return sorted (date, price) rows using the requested price column."""

    rows: List[Tuple[str, float]] = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        field_map = {field: normalize_column_name(field) for field in reader.fieldnames or []}
        for raw in reader:
            row = {field_map[key]: value for key, value in raw.items() if key is not None}
            day = clean_text(row.get("date", ""))[:10]
            if len(day) != 10:
                continue
            price = _to_float(row.get(price_column, ""))
            if price is None or price <= 0.0:
                price = _to_float(row.get("close", ""))
            if price is None or price <= 0.0:
                continue
            rows.append((day, price))
    rows.sort(key=lambda item: item[0])
    return rows


def _to_float(value: object) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_returns(
    stock_rows: Sequence[Tuple[str, float]],
    horizon: int,
) -> Dict[str, float]:
    """cutoff_date -> forward return over `horizon` trading days.

    ``horizon == 0`` is the same-day target: the session the cutoff date itself
    trades through, close(t-1) -> close(t). The premise is that articles dated
    day t are published in the morning and move that day's close.

    Two warnings that belong with this option. First, price features computed at
    the cutoff already contain close(t), and market feature 0 is exactly
    close(t)/close(t-1)-1 -- the label itself -- so this target is only valid
    with --disable_price; train_gnn.py refuses the combination. Second, 99.9% of
    this corpus carries a midnight placeholder instead of a publication time, so
    an article dated t may well have been published after the close and describe
    the very move being predicted. Pair this with --drop_price_echo_events and
    read the gap between the two as the size of that contamination.
    """

    returns: Dict[str, float] = {}
    if int(horizon) == 0:
        for index in range(1, len(stock_rows)):
            cutoff, price = stock_rows[index]
            previous = stock_rows[index - 1][1]
            if previous > 0.0:
                returns[cutoff] = price / previous - 1.0
        return returns
    for index in range(len(stock_rows) - horizon):
        cutoff, base = stock_rows[index]
        future = stock_rows[index + horizon][1]
        if base > 0.0:
            returns[cutoff] = future / base - 1.0
    return returns


def build_trailing_abs_medians(
    stock_rows: Sequence[Tuple[str, float]],
    horizon: int,
    lookback: int = 60,
) -> Dict[str, float]:
    """cutoff_date -> median |return| over the preceding `lookback` days.

    The comparison level for the magnitude target. It uses only prices strictly
    up to the cutoff, so the resulting label stays leakage-free.
    """

    medians: Dict[str, float] = {}
    history: List[float] = []
    # horizon 0 measures the same-day move, so its comparison level is the
    # trailing median of *one-day* moves. Using a lag of 0 would compare each
    # price with itself and make every median zero.
    lag = max(1, int(horizon))
    for index in range(len(stock_rows)):
        cutoff, price = stock_rows[index]
        if index >= lag:
            previous = stock_rows[index - lag][1]
            if previous > 0.0:
                history.append(abs(price / previous - 1.0))
        window = history[-lookback:]
        if len(window) >= 20:
            ordered = sorted(window)
            middle = len(ordered) // 2
            medians[cutoff] = (
                ordered[middle]
                if len(ordered) % 2
                else 0.5 * (ordered[middle - 1] + ordered[middle])
            )
    return medians


def assign_splits(
    cutoffs: Sequence[str],
    train_end: str,
    val_end: str,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> Dict[str, str]:
    """Bucket every cutoff date into train/val/test.

    This is the ONLY place the chronological boundary is decided. Stage 4 does
    not recompute it: TimeXer reads the resulting boundary dates back out of the
    exported embedding file, so both stages always cut the calendar in the same
    two places. See 4_train_timexer.sh.
    """

    if train_end or val_end:
        if not train_end or not val_end:
            raise ValueError("--train_end and --val_end must be provided together.")
        return {
            day: "train" if day <= train_end else "val" if day <= val_end else "test"
            for day in cutoffs
        }
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("--train_fraction and --val_fraction must lie in (0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("--train_fraction + --val_fraction must leave room for a test period.")
    count = len(cutoffs)
    train_cutoff = cutoffs[max(0, int(count * train_fraction) - 1)]
    val_cutoff = cutoffs[max(0, int(count * (train_fraction + val_fraction)) - 1)]
    return {
        day: "train" if day <= train_cutoff else "val" if day <= val_cutoff else "test"
        for day in cutoffs
    }


def resolve_universe(
    args: argparse.Namespace,
    graph_dir: Path,
    stock_dir: Path,
    requested: Sequence[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Dict[str, int]]]:
    """Decide which tickers get tasks and over which date span."""

    news_days: Dict[str, Dict[str, int]] = {}
    if args.task_source == "source_articles":
        bounds = read_article_bounds(graph_dir / "articles.csv", requested)
        if args.news_conditioned or args.min_news_days > 0:
            news_days = read_ticker_news_days(graph_dir, _cache_path(args, graph_dir))
        return bounds, news_days

    news_days = read_ticker_news_days(graph_dir, _cache_path(args, graph_dir))
    requested_set = set(requested)
    bounds: Dict[str, Tuple[str, str]] = {}
    skipped_no_price = 0
    skipped_thin = 0
    for ticker, days in news_days.items():
        if requested_set and ticker not in requested_set:
            continue
        if len(days) < int(args.min_news_days):
            skipped_thin += 1
            continue
        if not (stock_dir / f"{ticker}.csv").exists():
            skipped_no_price += 1
            continue
        bounds[ticker] = (min(days), max(days))
    print(
        f"universe: kept={len(bounds)} skipped_thin={skipped_thin} skipped_no_price={skipped_no_price}",
        flush=True,
    )
    if not bounds:
        raise ValueError("No tickers satisfied the universe filters.")
    return bounds, news_days


def _graph_fingerprint(graph_dir: Path) -> str:
    """Short identity for a built graph, from the files the cache derives from.

    Size and mtime rather than a content hash: edges.csv is over 500 MB here, and
    a spurious miss only costs a recompute while a spurious *hit* silently trains
    on another graph's supervision.
    """

    parts = [str(graph_dir.resolve())]
    for name in ("events.csv", "edges.csv"):
        path = graph_dir / name
        stat = path.stat() if path.exists() else None
        parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}" if stat else f"{name}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _cache_path(args: argparse.Namespace, graph_dir: Path) -> Optional[Path]:
    """Per-graph cache path.

    The cache used to be one fixed filename with no graph identity in it, so
    building tasks against a rebuilt or different graph silently reloaded the
    previous graph's ticker/day universe -- the subgraphs came from the new
    graph while the supervision came from the old one.
    """

    if not args.news_days_cache:
        return None
    base = Path(args.news_days_cache)
    return base.with_name(f"{base.stem}_{_graph_fingerprint(graph_dir)}{base.suffix}")


def build_tasks(
    args: argparse.Namespace,
    bounds: Dict[str, Tuple[str, str]],
    news_days: Dict[str, Dict[str, int]],
    stock_dir: Path,
    market_returns: Dict[str, float],
) -> List[Dict[str, object]]:
    horizon = int(args.horizon_trading_days)
    max_abs = float(args.max_abs_label_return)
    records: List[Dict[str, object]] = []
    dropped_extreme = 0
    dropped_no_news = 0

    for ticker in sorted(bounds):
        start, end = bounds[ticker]
        ticker_start = clean_text(args.start_date) or start
        ticker_end = clean_text(args.end_date) or end
        stock_path = stock_dir / f"{ticker}.csv"
        if not stock_path.exists():
            continue
        stock_rows = load_stock_rows(stock_path, args.label_price_column)
        if len(stock_rows) <= horizon:
            continue
        returns = build_returns(stock_rows, horizon)
        abs_medians = build_trailing_abs_medians(stock_rows, horizon)
        ticker_news = news_days.get(ticker, {})

        for cutoff, raw_return in returns.items():
            if cutoff < ticker_start or cutoff > ticker_end:
                continue
            news_count = _news_in_window(ticker_news, cutoff, int(args.news_window_days))
            if args.news_conditioned and news_count <= 0:
                dropped_no_news += 1
                continue
            market_return = market_returns.get(cutoff, 0.0)
            label_return = raw_return - market_return if args.label_mode == "excess" else raw_return
            if not math.isfinite(label_return) or abs(raw_return) > max_abs:
                dropped_extreme += 1
                continue
            abs_median = abs_medians.get(cutoff, float("nan"))
            records.append(
                {
                    "ticker": ticker,
                    "cutoff_date": cutoff,
                    "horizon_trading_days": horizon,
                    "label_return": label_return,
                    "label_direction": 1 if label_return > 0.0 else 0,
                    "label_abs_move": (
                        1 if math.isfinite(abs_median) and abs(raw_return) > abs_median else 0
                    ),
                    "trailing_abs_median": abs_median if math.isfinite(abs_median) else "",
                    "raw_return": raw_return,
                    "market_return": market_return,
                    "news_event_count": news_count,
                }
            )

    print(
        f"dropped_extreme_returns={dropped_extreme} dropped_no_news={dropped_no_news}",
        flush=True,
    )
    if not records:
        raise ValueError("No forecast tasks could be generated from the graph and stock histories.")

    cutoffs = sorted({str(record["cutoff_date"]) for record in records})
    split_by_date = assign_splits(
        cutoffs,
        clean_text(args.train_end),
        clean_text(args.val_end),
        float(args.train_fraction),
        float(args.val_fraction),
    )
    records.sort(key=lambda record: (str(record["cutoff_date"]), str(record["ticker"])))
    for record in records:
        record["split"] = split_by_date[str(record["cutoff_date"])]

    # Companies that barely appear in training cannot be fairly evaluated: their
    # ticker embedding is close to its random initialisation, so their rows are
    # near-noise and drag measured AUC toward 0.50 regardless of the graph. Drop
    # those rows from val/test while KEEPING their train rows, which still help
    # the shared encoder. They cannot be moved into train instead -- they are
    # late-dated, and the split is chronological precisely to keep the test
    # period out of training.
    minimum = int(getattr(args, "min_train_rows_for_eval", 0) or 0)
    if minimum > 0:
        train_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            if record["split"] == "train":
                train_counts[str(record["ticker"])] += 1
        kept = [
            record
            for record in records
            if record["split"] == "train" or train_counts[str(record["ticker"])] >= minimum
        ]
        thin = {
            str(record["ticker"])
            for record in records
            if record["split"] != "train" and train_counts[str(record["ticker"])] < minimum
        }
        print(
            f"min_train_rows_for_eval={minimum}: dropped {len(records) - len(kept)} val/test rows "
            f"from {len(thin)} thinly-trained tickers",
            flush=True,
        )
        records = kept
        if not any(record["split"] == "test" for record in records):
            raise ValueError(
                f"--min_train_rows_for_eval {minimum} removed every test row. Lower it."
            )

    # Split was assigned before the filter above; ids are numbered after it so
    # they stay contiguous.
    for index, record in enumerate(records):
        record["task_id"] = f"TASK_{index:08d}"
    return records


def _news_in_window(ticker_news: Dict[str, int], cutoff: str, window_days: int) -> int:
    """Event count available in [cutoff - window_days, cutoff] by calendar date."""

    if not ticker_news:
        return 0
    from datetime import date, timedelta

    end = date.fromisoformat(cutoff)
    total = 0
    for offset in range(int(window_days) + 1):
        total += ticker_news.get((end - timedelta(days=offset)).isoformat(), 0)
    return total


def load_market_returns(
    stock_dir: Path,
    ticker: str,
    horizon: int,
    price_column: str,
) -> Dict[str, float]:
    path = stock_dir / f"{normalize_ticker(ticker)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Market benchmark history not found: {path}")
    return build_returns(load_stock_rows(path, price_column), horizon)


def write_tasks(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in FIELDNAMES})
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chronological forecast supervision for the temporal GNN."
    )
    parser.add_argument("--graph_dir", default="ekg_final")
    parser.add_argument("--stock_dir", default="dataset/full_history")
    parser.add_argument("--output", default="gnn_data/forecast_task.csv")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--horizon_trading_days", type=int, default=1)
    parser.add_argument(
        "--task_source",
        choices=["event_tickers", "source_articles"],
        default="event_tickers",
        help="event_tickers supervises every company the graph events refer to.",
    )
    parser.add_argument(
        "--min_news_days",
        type=int,
        default=5,
        help="Minimum distinct availability dates a ticker must appear on to enter the universe.",
    )
    parser.add_argument(
        "--news_conditioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only emit tasks on days where the ticker has graph evidence in the window.",
    )
    parser.add_argument("--news_window_days", type=int, default=1)
    parser.add_argument(
        "--label_price_column",
        default="adj_close",
        help="Price column used for labels. adj_close gives dividend/split-consistent returns.",
    )
    parser.add_argument(
        "--label_mode",
        choices=["absolute", "excess"],
        default="absolute",
        help="excess subtracts the benchmark return, removing the common market factor.",
    )
    parser.add_argument("--market_ticker", default="SPY")
    parser.add_argument(
        "--max_abs_label_return",
        type=float,
        default=0.25,
        help="Drop tasks whose raw return exceeds this. Filters vendor adjustment splices.",
    )
    parser.add_argument(
        "--min_train_rows_for_eval",
        type=int,
        default=0,
        help=(
            "Drop val/test rows for tickers with fewer than N training rows, keeping "
            "their train rows. Without it ~48%% of test rows come from companies with "
            "under 50 training examples, whose predictions are close to noise."
        ),
    )
    parser.add_argument("--news_days_cache", default="gnn_data/ticker_news_days.csv")
    parser.add_argument("--start_date", default="")
    parser.add_argument("--end_date", default="")
    parser.add_argument("--train_end", default="")
    parser.add_argument("--val_end", default="")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.7,
        help="Fraction of unique cutoff dates used for training when no explicit "
             "--train_end/--val_end is given. Stage 4 inherits the resulting boundary.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of unique cutoff dates used for validation. The remainder is test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.horizon_trading_days) < 0:
        raise ValueError("--horizon_trading_days must be 0 (same day) or positive.")
    graph_dir = Path(args.graph_dir)
    stock_dir = Path(args.stock_dir)
    requested = parse_tickers(args.tickers)

    bounds, news_days = resolve_universe(args, graph_dir, stock_dir, requested)
    market_returns: Dict[str, float] = {}
    if args.label_mode == "excess":
        market_returns = load_market_returns(
            stock_dir, args.market_ticker, int(args.horizon_trading_days), args.label_price_column
        )

    records = build_tasks(args, bounds, news_days, stock_dir, market_returns)
    write_tasks(Path(args.output), records)

    tickers: Set[str] = {str(record["ticker"]) for record in records}
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val", "test")
    }
    positives = sum(int(record["label_direction"]) for record in records)
    print(
        {
            "output": str(Path(args.output)),
            "rows": len(records),
            "task_source": args.task_source,
            "label_mode": args.label_mode,
            "label_price_column": args.label_price_column,
            "news_conditioned": bool(args.news_conditioned),
            "ticker_count": len(tickers),
            "positive_rate": round(positives / len(records), 4),
            "splits": split_counts,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
