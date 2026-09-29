#!/usr/bin/env python3
"""Export the daily 64-d news embedding per (ticker, date) for TimeXer.

Runs the trained encoder over every forecast date and writes its graph embedding.
The embedding is z-scored using *training-split* statistics only, so no test-period
information reaches the scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Run this with the project Python environment/venv.") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Run this with the project Python environment/venv.") from exc

from build_temporal_dataset import (
    add_common_args,
    load_forecast_tasks,
    load_graph_artifacts,
    parse_tickers,
)
from models import TemporalEventGNN
from train_gnn import build_batch, iterate_batches, load_or_build_examples, resolve_device, run_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily ticker embeddings.")
    add_common_args(parser)
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument("--checkpoint")
    checkpoint_source.add_argument(
        "--best_validation_metrics_dir",
        help="Select the seed checkpoint with the highest validation AUC in this directory.",
    )
    parser.add_argument(
        "--run_group",
        default="gnn",
        help="Run-name prefix used with --best_validation_metrics_dir (default: gnn).",
    )
    parser.add_argument("--output_csv", default="gnn_outputs/daily_ticker_embeddings.csv")
    parser.add_argument("--embedding_prefix", default="gnn")
    parser.add_argument("--normalization", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--example_cache_dir", default="gnn_data/example_cache")
    # Kept so the export shares build_batch with training.
    parser.add_argument("--weight_scale", type=float, default=0.015)
    parser.add_argument("--label_weighting", choices=["none", "magnitude"], default="none")
    return parser.parse_args()


def select_best_validation_checkpoint(
    metrics_dir: Path, run_group: str
) -> Tuple[Path, Dict[str, object]]:
    """Choose a seed using validation AUC only; the test set is never inspected."""
    metric_paths = sorted(metrics_dir.glob(f"training_metrics_{run_group}_seed*.json"))
    if not metric_paths:
        raise FileNotFoundError(
            f"No training_metrics_{run_group}_seed*.json files found in {metrics_dir}."
        )

    candidates: List[Tuple[float, str, Path]] = []
    for metric_path in metric_paths:
        with metric_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        run_name = str(payload.get("summary", {}).get("run_name", ""))
        val_auc = payload.get("val_metrics", {}).get("auc")
        if not run_name or not isinstance(val_auc, (int, float)) or not np.isfinite(val_auc):
            raise ValueError(f"Missing finite validation AUC or run name in {metric_path}.")
        candidates.append((float(val_auc), run_name, metric_path))

    # Highest validation AUC wins. Run name is an ascending deterministic
    # tiebreaker; test metrics are deliberately not read.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    val_auc, run_name, metric_path = candidates[0]
    checkpoint_path = metrics_dir / "checkpoints" / f"event_gnn_{run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best validation run is {run_name}, but its checkpoint is missing: {checkpoint_path}"
        )
    return checkpoint_path, {
        "method": "highest_validation_auc",
        "run_name": run_name,
        "validation_auc": val_auc,
        "metrics_file": str(metric_path),
    }


def normalize_embeddings(embeddings: np.ndarray, splits: List[str], method: str) -> np.ndarray:
    if method == "none" or embeddings.size == 0:
        return embeddings.astype(np.float32)
    train_mask = np.asarray([s == "train" for s in splits], dtype=bool)
    if not train_mask.any():
        raise ValueError("Cannot z-score embeddings without training rows.")
    train_values = embeddings[train_mask]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((embeddings - mean) / std).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.best_validation_metrics_dir:
        checkpoint_path, checkpoint_selection = select_best_validation_checkpoint(
            Path(args.best_validation_metrics_dir), str(args.run_group)
        )
    else:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_selection = {"method": "explicit_checkpoint"}
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run gnn/train_gnn.py first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_graph: Dict[str, object] = checkpoint.get("graph_config", {})

    tasks = load_forecast_tasks(
        args.forecast_tasks, parse_tickers(args.tickers),
        args.horizon_trading_days, args.min_abs_label_return,
    )
    tickers = sorted({task.ticker for task in tasks})
    graph = load_graph_artifacts(args.graph_dir, tickers)
    # A silent vocabulary mismatch produces embeddings that look fine and mean
    # nothing, so refuse rather than warn.
    if checkpoint_graph.get("node_type_to_id") != graph.node_type_to_id:
        raise ValueError("Graph node types do not match the checkpoint. Retrain the GNN.")
    if checkpoint_graph.get("relation_to_id") != graph.relation_to_id:
        raise ValueError("Graph relations do not match the checkpoint. Retrain the GNN.")

    examples = load_or_build_examples(args, graph, tasks, tickers)
    device = resolve_device(args.device)
    global_node_features = torch.as_tensor(
        np.array(graph.node_features, copy=True), dtype=torch.float32, device=device
    )
    model = TemporalEventGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ticker_to_id = dict(checkpoint_graph.get("ticker_to_id") or {t: i + 1 for i, t in enumerate(tickers)})
    missing = [t for t in tickers if t not in ticker_to_id]
    if missing:
        raise ValueError(f"Checkpoint has no ticker embedding for {missing}. Retrain the GNN.")

    rows: List[Dict[str, object]] = []
    raw: List[np.ndarray] = []
    with torch.no_grad():
        for group in iterate_batches(examples, int(args.eval_batch_size), False):
            batch = build_batch(
                group, global_node_features, ticker_to_id, device,
                args.weight_scale, args.label_weighting,
            )
            logits, embeddings, _ = run_model(model, batch)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            values = embeddings.float().cpu().numpy()
            for position, example in enumerate(group):
                rows.append({
                    "task_id": example.task_id,
                    "date": example.current_date.isoformat(),
                    "ticker": example.ticker,
                    "split": example.split,
                    "label_return": example.label_return,
                    "label_direction": example.label,
                    "direction_probability": float(probabilities[position]),
                })
                raw.append(values[position])

    embeddings = normalize_embeddings(
        np.stack(raw, axis=0), [str(r["split"]) for r in rows], args.normalization
    )
    prefix = args.embedding_prefix
    dim = int(embeddings.shape[1])
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
        "rows": len(rows),
        "ticker_count": len(tickers),
        "embedding_dim": dim,
        "normalization": args.normalization,
        "output_csv": str(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": checkpoint_selection,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps({**metadata, "metadata_json": str(metadata_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
