#!/usr/bin/env python3
"""Input files for the FNSPID Transformer replication (stage 8).

Two sets:

data_theirs/  The dataset authors' own per-stock files for the five stocks their
              paper evaluates (KO, AMD, TSM, GOOG, WMT), copied byte for byte
              from their repository at commit eecdcdb (2024-02-19), the upload
              that accompanied the Table 3 results. Used only to check that the
              port reproduces their numbers before it is trusted on our data.

data_ours/    Their files for the thesis's five companies (T, INTC, AMD, CVX,
              BABA): their prices and their Sentiment_gpt/Scaled_sentiment
              columns, unchanged, plus the 64 graph dimensions gnn_0..gnn_63
              from the stage-3 embedding file. The graph values are joined the
              way their code treats the sentiment column: the representation of
              the news dated d sits on the row dated d. Days without an
              embedding get a zero vector, the convention every other stage
              uses. Rows before the first embedding date are dropped for every
              arm, so all three arms see the identical rows and the graph arm
              never trains on years of all-zero news.

The files for our five companies are the same bytes as fnspid_sentiment/, which
were taken from the same repository for stage 7.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

OURS = ["T", "INTC", "AMD", "CVX", "BABA"]
THEIRS = ["KO", "AMD", "TSM", "GOOG", "WMT"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_data_dir", required=True)
    parser.add_argument("--sentiment_dir", required=True)
    parser.add_argument("--embeddings_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    (out / "data_theirs").mkdir(parents=True, exist_ok=True)
    (out / "data_ours").mkdir(parents=True, exist_ok=True)

    repo = Path(args.repo_data_dir)
    for tk in THEIRS:
        source = next(p for p in repo.iterdir() if p.name.lower() == f"{tk.lower()}.csv")
        shutil.copyfile(source, out / "data_theirs" / f"{tk}.csv")
        print(f"copied  {source.name:9s} -> data_theirs/{tk}.csv")

    embedding = {}
    first_date = None
    dims = None
    with open(args.embeddings_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        dims = [c for c in reader.fieldnames if c.startswith("gnn_")]
        dims.sort(key=lambda c: int(c.split("_", 1)[1]))
        for row in reader:
            if row["ticker"] not in OURS:
                continue
            embedding[(row["ticker"], row["date"])] = [row[c] for c in dims]
            if first_date is None or row["date"] < first_date:
                first_date = row["date"]
    if len(dims) != 64:
        raise SystemExit(f"expected 64 gnn_* columns, found {len(dims)}")
    print(f"embeddings: {len(embedding)} company-days, first date {first_date}")

    zero = ["0.0"] * len(dims)
    for tk in OURS:
        source = Path(args.sentiment_dir) / f"fnspid_{tk}.csv"
        with open(source, newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames)
            rows = list(reader)
        kept, with_news = [], 0
        for row in rows:
            day = row["Date"][:10]
            if day < first_date:
                continue
            vector = embedding.get((tk, day))
            if vector is not None:
                with_news += 1
            row.update(zip(dims, vector if vector is not None else zero))
            kept.append(row)
        with open(out / "data_ours" / f"{tk}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + dims)
            writer.writeheader()
            writer.writerows(kept)
        print(f"wrote   data_ours/{tk}.csv  {len(kept)} rows "
              f"({kept[0]['Date'][:10]} -> {kept[-1]['Date'][:10]}), "
              f"{with_news} with a graph embedding, {len(rows) - len(kept)} earlier rows dropped")


if __name__ == "__main__":
    main()
