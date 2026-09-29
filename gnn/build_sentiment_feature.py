#!/usr/bin/env python3
"""Build the daily sentiment feature released with the dataset paper.

This is NOT a reimplementation. The scores are the authors' own ChatGPT-derived
values, taken from the per-company files distributed with their experiment code,
so the sentiment arm uses exactly the representation their paper evaluates
rather than an approximation of it.

Their column `Sentiment_gpt` holds the 1-5 score, averaged over the articles of
a day and already decayed toward the neutral value 3 on days without news, as
their Equation 1 specifies. `Scaled_sentiment` is the affine map
(Sentiment_gpt - 1) / 4 and is therefore equivalent after standardisation.

Four of the five series end on 2023-12-15 while the evaluation period runs to
2023-12-28. Those trailing days are filled with the authors' own decay rule,
which is what their method does for any day without news.

Output matches the schema of gnn/export_embeddings.py, including the `gnn_`
column prefix, so the existing join and loader need no modification.
"""
import argparse, csv, math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Daily sentiment feature from the dataset paper's released scores.")
    p.add_argument("--source_dir", default="fnspid_sentiment",
                   help="Directory holding the authors' per-company files, named fnspid_<TICKER>.csv")
    p.add_argument("--stock_dir", default="TimeXer/dataset/stock",
                   help="Directory holding <TICKER>_graph.csv, used to fix the exact date list")
    p.add_argument("--tickers", default="T,INTC,AMD,CVX,BABA")
    p.add_argument("--train_end", required=True, help="Last training date; standardisation uses these rows only")
    p.add_argument("--val_end", required=True)
    p.add_argument("--decay_lambda", type=float, default=0.03,
                   help="Their decay constant toward the neutral score")
    p.add_argument("--neutral", type=float, default=3.0)
    p.add_argument("--column", default="Sentiment_gpt")
    p.add_argument("--output_csv", default="gnn_outputs_sentiment/daily_ticker_sentiment.csv")
    return p.parse_args()


def main():
    a = parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    out_rows = []
    print(f"{'ticker':7s} {'dates':>6s} {'direct':>7s} {'decayed':>8s} {'their last':>12s}")
    for tk in tickers:
        src = Path(a.source_dir) / f"fnspid_{tk}.csv"
        if not src.exists():
            raise SystemExit(f"ERROR: missing {src}")
        theirs = {}
        for r in csv.DictReader(src.open(encoding="utf-8", errors="replace")):
            d = r["Date"][:10]
            v = r.get(a.column, "").strip()
            if d and v:
                theirs[d] = float(v)
        if not theirs:
            raise SystemExit(f"ERROR: no {a.column} values in {src}")
        last_date = max(theirs)
        last_val = theirs[last_date]

        price = Path(a.stock_dir) / f"{tk}_graph.csv"
        dates = [r["date"] for r in csv.DictReader(price.open(encoding="utf-8", errors="replace"))]

        direct = decayed = 0
        series = []
        for d in dates:
            if d in theirs:
                series.append((d, theirs[d])); direct += 1
            else:
                # their Equation 1: S(t) = neutral + (S(0) - neutral) * exp(-lambda * t)
                gap = sum(1 for x in dates if last_date < x <= d)
                v = a.neutral + (last_val - a.neutral) * math.exp(-a.decay_lambda * gap)
                series.append((d, v)); decayed += 1
        print(f"{tk:7s} {len(series):6d} {direct:7d} {decayed:8d} {last_date:>12s}")

        train = [v for d, v in series if d <= a.train_end]
        mu = sum(train) / len(train)
        var = sum((x - mu) ** 2 for x in train) / len(train)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        for d, v in series:
            split = "train" if d <= a.train_end else ("val" if d <= a.val_end else "test")
            out_rows.append(dict(task_id=f"{tk}_{d}", date=d, ticker=tk, split=split,
                                 gnn_0=f"{(v - mu) / sd:.10f}"))

    out = Path(a.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["task_id", "date", "ticker", "split", "gnn_0"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows to {out}")
    print("Standardised on training rows only, one dimension, prefixed gnn_ so the")
    print("existing join and loader treat it exactly as any other news representation.")


if __name__ == "__main__":
    main()
