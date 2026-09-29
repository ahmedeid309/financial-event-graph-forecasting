#!/usr/bin/env python3
"""Aggregate training_metrics_*.json into a seed-averaged comparison table."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Run this with the project Python environment/venv.") from exc


SEED_SUFFIX = re.compile(r"_seed\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GNN runs across seeds.")
    parser.add_argument("--metrics_dir", default="gnn_outputs_shared")
    parser.add_argument("--output_json", default="gnn_outputs_shared/run_summary.json")
    parser.add_argument(
        "--sort_by",
        choices=["val_auc", "test_auc"],
        default="val_auc",
        help="Ranking metric. Defaults to validation so the test set stays unused for selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for path in sorted(metrics_dir.glob("training_metrics_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        run_name = str(payload.get("summary", {}).get("run_name", path.stem))
        groups[SEED_SUFFIX.sub("", run_name)].append(payload)

    rows: List[Dict[str, object]] = []
    for group, runs in sorted(groups.items()):
        def collect(section: str, key: str) -> List[float]:
            values = []
            for run in runs:
                value = run.get(section, {}).get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))
            return values

        summary = runs[0].get("summary", {})
        row: Dict[str, object] = {
            "run_group": group,
            "seeds": len(runs),
            "model": summary.get("model", ""),
            "event_ticker_scope": summary.get("context", {}).get("event_ticker_scope", ""),
            "graph_control": summary.get("context", {}).get("graph_control", "none"),
            "forecast_tasks": summary.get("forecast_tasks", ""),
            "ticker_count": summary.get("ticker_count", 0),
            "train_count": summary.get("splits", {}).get("train", {}).get("count", 0),
            "test_count": summary.get("splits", {}).get("test", {}).get("count", 0),
        }
        for label, (section, key) in {
            "val_auc": ("val_metrics", "auc"),
            "test_auc": ("test_metrics", "auc"),
            "val_acc_0_5": ("val_metrics", "directional_accuracy_at_0_5"),
            "test_acc_0_5": ("test_metrics", "directional_accuracy_at_0_5"),
            "test_bal_acc_0_5": ("test_metrics", "direction_balanced_accuracy_at_0_5"),
        }.items():
            values = collect(section, key)
            row[f"{label}_mean"] = round(float(np.mean(values)), 4) if values else None
            row[f"{label}_std"] = round(float(np.std(values)), 4) if values else None
        rows.append(row)

    # Selection is a validation decision. Ranking by test AUC would make the
    # test set part of model selection.
    rows.sort(key=lambda item: -(item.get(f"{args.sort_by}_mean") or 0.0))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    header = (
        f"{'run_group':38s} {'control':16s} {'n':>3s} "
        f"{'val_auc':>14s} {'test_auc':>14s} {'test_acc':>14s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        def cell(label: str) -> str:
            mean = row.get(f"{label}_mean")
            std = row.get(f"{label}_std")
            return f"{mean:.3f}+-{std:.3f}" if mean is not None else "n/a"

        print(
            f"{str(row['run_group'])[:38]:38s} {str(row['graph_control']):16s} "
            f"{row['seeds']:>3d} {cell('val_auc'):>14s} {cell('test_auc'):>14s} "
            f"{cell('test_acc_0_5'):>14s}",
            flush=True,
        )


if __name__ == "__main__":
    main()
