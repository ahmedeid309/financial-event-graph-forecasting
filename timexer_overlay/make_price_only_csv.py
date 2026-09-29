#!/usr/bin/env python3
"""Create a price-only TimeXer CSV from a graph-enhanced stock CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop graph embedding columns while preserving the same dates.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    columns = ["date", "open", "high", "low", "adj_close", "volume", "close"]

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8", errors="replace") as source:
        reader = csv.DictReader(source)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {missing}")

        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for row in reader:
                writer.writerow({column: row[column] for column in columns})

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
