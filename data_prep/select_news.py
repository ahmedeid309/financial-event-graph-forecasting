#!/usr/bin/env python3
"""Select the thesis news corpus from the FNSPID news archive.

Keeps the articles filed under the five focal companies, published from
1 January 2018 through 31 December 2023, and with non-empty article text
(3,872 archive rows in that range have none), with the five columns the
pipeline reads.
Rows are grouped by company in the order T, INTC, AMD, CVX, BABA and keep their
archive order within each company. The result is dataset/news_2018_2023.csv
(36,992 articles).

Usage (from the repository root):
    python data_prep/select_news.py \
        --archive_csv <FNSPID>/nasdaq_exteral_data.csv \
        --output_csv dataset/news_2018_2023.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

TICKERS = ["T", "INTC", "AMD", "CVX", "BABA"]
COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Article"]
FIRST_DAY = "2018-01-01"
LAST_DAY = "2023-12-31"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive_csv", required=True, help="FNSPID nasdaq_exteral_data.csv")
    parser.add_argument("--output_csv", default="dataset/news_2018_2023.csv")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    selected = {ticker: [] for ticker in TICKERS}
    with open(args.archive_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker = row["Stock_symbol"]
            if ticker in selected and FIRST_DAY <= row["Date"][:10] <= LAST_DAY and row["Article"].strip():
                selected[ticker].append([row[column] for column in COLUMNS])

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for ticker in TICKERS:
            writer.writerows(selected[ticker])
    print({ticker: len(rows) for ticker, rows in selected.items()}, "->", output)


if __name__ == "__main__":
    main()
