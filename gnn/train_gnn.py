#!/usr/bin/env python3
"""Train the news-only event GNN on next/same-day price direction.

One configuration. Subgraphs are packed block-diagonally into real minibatches,
labels are weighted by move magnitude rather than discarded, and model selection
runs on a smoothed validation AUC so a single lucky epoch cannot win.

Optional structural ablations remain available for diagnostic work, but the
canonical workflow trains only the real graph configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NumPy is required. Run this with the project Python environment/venv.") from exc

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Run this with the project Python environment/venv.") from exc

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required in the project Python environment.") from exc

from build_temporal_dataset import (
    TEMPORAL_FEATURE_DIM,
    TemporalExample,
    add_common_args,
    build_temporal_examples,
    load_forecast_tasks,
    load_graph_artifacts,
    parse_tickers,
    split_examples,
    summarize_examples,
)
from models import TemporalEventGNN


def resolve_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(value)


@dataclass
class Batch:
    text_features: torch.Tensor
    temporal_features: torch.Tensor
    node_type_ids: torch.Tensor
    event_type_ids: torch.Tensor
    time_role_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_relation_ids: torch.Tensor
    edge_weights: torch.Tensor
    node_batch: torch.Tensor
    event_node_mask: torch.Tensor
    target_node_indices: torch.Tensor
    ticker_ids: torch.Tensor
    labels: torch.Tensor
    label_returns: torch.Tensor
    weights: torch.Tensor
    size: int


def build_batch(
    examples: Sequence[TemporalExample],
    global_node_features: torch.Tensor,
    ticker_to_id: Dict[str, int],
    device: torch.device,
    weight_scale: float,
    weighting: str,
    graph_control: str = "none",
    control_rng: Optional[random.Random] = None,
) -> Batch:
    """Pack subgraphs block-diagonally into one minibatch.

    ``graph_control`` damages one specific part of the graph signal and leaves
    everything else intact. If a control scores like the real graph, that part
    was contributing nothing.
    """

    text_blocks: List[torch.Tensor] = []
    temporal_blocks: List[np.ndarray] = []
    node_type_blocks: List[np.ndarray] = []
    event_type_blocks: List[np.ndarray] = []
    time_role_blocks: List[np.ndarray] = []
    event_mask_blocks: List[np.ndarray] = []
    edge_index_blocks: List[np.ndarray] = []
    relation_blocks: List[np.ndarray] = []
    edge_weight_blocks: List[np.ndarray] = []
    node_batch_blocks: List[np.ndarray] = []
    target_indices: List[int] = []
    ticker_ids: List[int] = []
    labels: List[float] = []
    label_returns: List[float] = []

    offset = 0
    for position, example in enumerate(examples):
        graph = example
        node_count = int(graph.node_indices.shape[0])
        indices = torch.as_tensor(graph.node_indices, dtype=torch.long, device=device)
        base = global_node_features.index_select(0, indices)
        if bool(graph.fusion_node_mask.any()):
            # Fusion-node state is populated dynamically by member messages.
            base = base.clone()
            base[torch.as_tensor(graph.fusion_node_mask, dtype=torch.bool, device=device)] = 0.0
        if graph_control == "zero_text":
            base = torch.zeros_like(base)
        text_blocks.append(base)
        temporal_blocks.append(graph.temporal_features)
        node_type_blocks.append(graph.node_type_ids)
        event_type_blocks.append(graph.node_event_type_ids)
        time_role_blocks.append(graph.node_time_role_ids)
        event_mask_blocks.append(graph.event_node_mask)
        if graph.edge_index.size and graph_control != "no_edges":
            edge_index_blocks.append(graph.edge_index + offset)
            relation_blocks.append(graph.edge_relation_ids)
            edge_weight_blocks.append(graph.edge_weights)
        node_batch_blocks.append(np.full((node_count,), position, dtype=np.int64))
        target_indices.append(int(graph.target_node_idx) + offset)
        ticker_ids.append(ticker_to_id[example.ticker])
        labels.append(float(example.label))
        label_returns.append(float(example.label_return))
        offset += node_count

    returns = np.asarray(label_returns, dtype=np.float32)
    if weighting == "magnitude":
        weights = np.clip(np.abs(returns) / max(1e-6, weight_scale), 0.2, 3.0).astype(np.float32)
    else:
        weights = np.ones_like(returns)

    def stack(blocks: List[np.ndarray], dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.concatenate(blocks, axis=0), dtype=dtype, device=device)

    if graph_control == "shuffle_relations" and relation_blocks:
        # Keep topology and edge weights, destroy only which relation each edge
        # carries. Permuting across the batch preserves the marginal distribution.
        merged = np.concatenate(relation_blocks)
        permutation = np.random.default_rng(
            None if control_rng is None else control_rng.getrandbits(32)
        ).permutation(merged.shape[0])
        merged = merged[permutation]
        rebuilt, cursor = [], 0
        for block in relation_blocks:
            rebuilt.append(merged[cursor : cursor + block.shape[0]])
            cursor += block.shape[0]
        relation_blocks = rebuilt

    empty_edges = not edge_index_blocks
    return Batch(
        text_features=torch.cat(text_blocks, dim=0),
        temporal_features=stack(temporal_blocks, torch.float32),
        node_type_ids=stack(node_type_blocks, torch.long),
        event_type_ids=stack(event_type_blocks, torch.long),
        time_role_ids=stack(time_role_blocks, torch.long),
        edge_index=(
            torch.zeros((2, 0), dtype=torch.long, device=device)
            if empty_edges
            else torch.as_tensor(
                np.concatenate(edge_index_blocks, axis=1), dtype=torch.long, device=device
            )
        ),
        edge_relation_ids=(
            torch.zeros((0,), dtype=torch.long, device=device)
            if empty_edges
            else stack(relation_blocks, torch.long)
        ),
        edge_weights=(
            torch.zeros((0,), dtype=torch.float32, device=device)
            if empty_edges
            else stack(edge_weight_blocks, torch.float32)
        ),
        node_batch=stack(node_batch_blocks, torch.long),
        event_node_mask=stack(event_mask_blocks, torch.bool),
        target_node_indices=torch.as_tensor(target_indices, dtype=torch.long, device=device),
        ticker_ids=torch.as_tensor(ticker_ids, dtype=torch.long, device=device),
        labels=torch.as_tensor(labels, dtype=torch.float32, device=device),
        label_returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        weights=torch.as_tensor(weights, dtype=torch.float32, device=device),
        size=len(examples),
    )


def run_model(model: nn.Module, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(
        batch.text_features,
        batch.temporal_features,
        batch.node_type_ids,
        batch.event_type_ids,
        batch.time_role_ids,
        batch.edge_index,
        batch.edge_relation_ids,
        batch.edge_weights,
        batch.node_batch,
        batch.event_node_mask,
        batch.target_node_indices,
        batch.ticker_ids,
        num_graphs=batch.size,
    )


def compute_loss(
    logits: torch.Tensor,
    predicted_return: torch.Tensor,
    batch: Batch,
    return_loss_weight: float,
    return_scale: float,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, batch.labels, weight=batch.weights, reduction="mean"
    )
    if return_loss_weight > 0.0:
        # Predicting the size of the move alongside its sign gives the encoder a
        # richer gradient than a single bit. tanh stops one -20% day dominating.
        target = torch.tanh(batch.label_returns / max(1e-6, return_scale))
        loss = loss + return_loss_weight * nn.functional.huber_loss(
            torch.tanh(predicted_return), target, delta=1.0
        )
    return loss


def iterate_batches(
    examples: Sequence[TemporalExample],
    batch_size: int,
    shuffle: bool,
    rng: Optional[random.Random] = None,
) -> List[List[TemporalExample]]:
    order = list(range(len(examples)))
    if shuffle and rng is not None:
        rng.shuffle(order)
    return [
        [examples[index] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def train_epoch(
    model: nn.Module,
    examples: Sequence[TemporalExample],
    global_node_features: torch.Tensor,
    ticker_to_id: Dict[str, int],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    args: argparse.Namespace,
    rng: random.Random,
) -> float:
    model.train()
    losses: List[float] = []
    for group in iterate_batches(examples, int(args.batch_size), True, rng):
        batch = build_batch(
            group, global_node_features, ticker_to_id, device,
            args.weight_scale, args.label_weighting, args.graph_control, rng,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _, predicted_return = run_model(model, batch)
        loss = compute_loss(
            logits, predicted_return, batch, float(args.return_loss_weight), float(args.return_scale)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def direction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {k: float("nan") for k in
                ("auc", "directional_accuracy_at_0_5", "direction_balanced_accuracy_at_0_5")}
    predictions = (probabilities >= 0.5).astype(np.int64)
    positive, negative = labels == 1, labels == 0
    positive_recall = float(np.mean(predictions[positive] == 1)) if positive.any() else float("nan")
    negative_recall = float(np.mean(predictions[negative] == 0)) if negative.any() else float("nan")
    recalls = [v for v in (positive_recall, negative_recall) if not math.isnan(v)]
    return {
        "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan"),
        "directional_accuracy_at_0_5": float(np.mean(predictions == labels)),
        "direction_balanced_accuracy_at_0_5": float(np.mean(recalls)) if recalls else float("nan"),
        "direction_positive_recall_at_0_5": positive_recall,
        "direction_negative_recall_at_0_5": negative_recall,
        "direction_pred_positive_rate_at_0_5": float(np.mean(predictions == 1)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    examples: Sequence[TemporalExample],
    global_node_features: torch.Tensor,
    ticker_to_id: Dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    if not examples:
        return {"loss": float("nan"), "count": 0, **direction_metrics(np.asarray([]), np.asarray([]))}, []
    model.eval()
    losses: List[float] = []
    labels: List[int] = []
    probabilities: List[float] = []
    rows: List[Dict[str, object]] = []
    for group in iterate_batches(examples, int(args.eval_batch_size), False):
        batch = build_batch(
            group, global_node_features, ticker_to_id, device,
            args.weight_scale, args.label_weighting, args.graph_control, random.Random(12345),
        )
        logits, _, predicted_return = run_model(model, batch)
        losses.append(float(compute_loss(
            logits, predicted_return, batch, float(args.return_loss_weight), float(args.return_scale)
        ).item()))
        probs = torch.sigmoid(logits).cpu().numpy()
        logit_values = logits.cpu().numpy()
        for position, example in enumerate(group):
            labels.append(example.label)
            probabilities.append(float(probs[position]))
            rows.append({
                "task_id": example.task_id,
                "date": example.current_date.isoformat(),
                "ticker": example.ticker,
                "split": example.split,
                "label_return": example.label_return,
                "direction_label": example.label,
                "direction_logit": float(logit_values[position]),
                "direction_probability": float(probs[position]),
                "direction_prediction_0_5": int(float(probs[position]) >= 0.5),
            })
    labels_arr = np.asarray(labels, dtype=np.int64)
    probabilities_arr = np.asarray(probabilities, dtype=np.float64)
    return {
        "loss": float(np.mean(losses)),
        "count": len(examples),
        **direction_metrics(labels_arr, probabilities_arr),
        "direction_probability_min": float(probabilities_arr.min()),
        "direction_probability_max": float(probabilities_arr.max()),
        "direction_probability_mean": float(probabilities_arr.mean()),
        "direction_probability_std": float(probabilities_arr.std()),
    }, rows


def write_predictions(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id", "date", "ticker", "split", "label_return",
        "direction_label", "direction_logit", "direction_probability",
        "direction_prediction_0_5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def example_cache_key(args: argparse.Namespace, tickers: Sequence[str]) -> str:
    # Hash the supervision content, not just its path: regenerating a table in
    # place (new columns, new labels) must invalidate cached examples.
    task_digest = hashlib.sha256(Path(args.forecast_tasks).read_bytes()).hexdigest()[:16]
    payload = json.dumps(
        {
            "graph_dir": str(Path(args.graph_dir).resolve()),
            "forecast_tasks_digest": task_digest,
            "window_days": int(args.window_days),
            "horizon": int(args.horizon_trading_days),
            "decay_lambda": float(args.decay_lambda),
            "num_hops": int(args.num_hops),
            "semantic_tau_days": float(args.semantic_tau_days),
            "event_ticker_scope": str(args.event_ticker_scope),
            "min_abs_label_return": float(args.min_abs_label_return),
            "tickers": list(tickers),
            "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def load_or_build_examples(args, graph, tasks, tickers: Sequence[str]) -> List[TemporalExample]:
    cache_dir = Path(args.example_cache_dir) if args.example_cache_dir else None
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_file = cache_dir / f"examples_{example_cache_key(args, tickers)}.pkl"
        if cache_file.exists():
            print(f"Loading cached temporal examples from {cache_file}", flush=True)
            with cache_file.open("rb") as handle:
                return pickle.load(handle)

    examples = build_temporal_examples(
        graph, tasks, args.window_days, args.decay_lambda, args.num_hops,
        args.semantic_tau_days, args.event_ticker_scope,
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached temporal examples to {cache_file}", flush=True)
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the news-only event GNN.")
    add_common_args(parser)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--text_projection_dim", type=int, default=96)
    parser.add_argument("--aggregation", choices=["mean", "sqrt", "sum"], default="sqrt")
    parser.add_argument("--graph_gate_init", type=float, default=2.0)
    parser.add_argument(
        "--readout_key_source",
        choices=["post", "pre", "both"],
        default="post",
        help=(
            "Which event state the attention key sees. 'post' is after message "
            "passing (contextualised, but events become ~0.93 pairwise cosine); "
            "'pre' is the event's own content only; 'both' concatenates them."
        ),
    )
    parser.add_argument(
        "--disable_graph",
        action="store_true",
        help="Remove the graph pathway, leaving ticker identity only. The floor any result must beat.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--label_weighting", choices=["none", "magnitude"], default="magnitude")
    parser.add_argument("--weight_scale", type=float, default=0.015)
    parser.add_argument("--return_loss_weight", type=float, default=0.3)
    parser.add_argument("--return_scale", type=float, default=0.02)
    parser.add_argument("--selection_smoothing", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--example_cache_dir", default="gnn_data/example_cache")
    parser.add_argument(
        "--graph_control",
        choices=["none", "no_edges", "zero_text", "shuffle_relations"],
        default="none",
        help=(
            "Negative control. Damages one part of the graph signal and leaves the "
            "rest intact; if a control scores like the real graph, that part was "
            "contributing nothing. Quote a control with every reported result."
        ),
    )
    parser.add_argument("--run_name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = load_forecast_tasks(
        args.forecast_tasks, parse_tickers(args.tickers),
        args.horizon_trading_days, args.min_abs_label_return,
    )
    tickers = sorted({task.ticker for task in tasks})
    graph = load_graph_artifacts(args.graph_dir, tickers)
    examples = load_or_build_examples(args, graph, tasks, tickers)

    ticker_to_id = {ticker: index + 1 for index, ticker in enumerate(tickers)}
    train_examples = split_examples(examples, "train")
    val_examples = split_examples(examples, "val")
    test_examples = split_examples(examples, "test")
    if not train_examples:
        raise ValueError("No training examples were built.")

    device = resolve_device(args.device)
    global_node_features = torch.as_tensor(
        np.array(graph.node_features, copy=True), dtype=torch.float32, device=device
    )
    model_config = {
        "text_dim": int(graph.node_features.shape[1]),
        "temporal_dim": TEMPORAL_FEATURE_DIM,
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "num_node_types": len(graph.node_type_to_id),
        "num_relations": len(graph.relation_to_id),
        "num_event_types": len(graph.event_type_to_id) + 1,
        "num_time_roles": len(graph.time_role_to_id) + 1,
        "num_tickers": len(tickers) + 1,
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "text_projection_dim": int(args.text_projection_dim),
        "aggregation": args.aggregation,
        "graph_gate_init": float(args.graph_gate_init),
        "use_graph": not bool(args.disable_graph),
        "readout_key_source": str(args.readout_key_source),
    }
    model = TemporalEventGNN(**model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    steps_per_epoch = max(1, math.ceil(len(train_examples) / int(args.batch_size)))
    total_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = max(1, int(total_steps * float(args.warmup_fraction)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or "event_gnn"
    checkpoint_path = checkpoint_dir / f"event_gnn_{run_name}.pt"

    summary = {
        "run_name": run_name,
        "graph_dir": str(Path(args.graph_dir)),
        "forecast_tasks": str(Path(args.forecast_tasks)),
        "ticker_count": len(tickers),
        "splits": {
            split: summarize_examples(split, split_examples(examples, split))
            for split in ("train", "val", "test")
        },
        "context": {
            "window_days": int(args.window_days),
            "horizon_trading_days": int(args.horizon_trading_days),
            "decay_lambda": float(args.decay_lambda),
            "num_hops": int(args.num_hops),
            "semantic_tau_days": float(args.semantic_tau_days),
            "event_ticker_scope": str(args.event_ticker_scope),
            "min_abs_label_return": float(args.min_abs_label_return),
            "disable_graph": bool(args.disable_graph),
            "graph_control": str(args.graph_control),
            "readout_key_source": str(args.readout_key_source),
            "label_weighting": args.label_weighting,
            "return_loss_weight": float(args.return_loss_weight),
        },
        "optimization": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "seed": int(args.seed),
        },
        "model_config": model_config,
    }
    print(json.dumps(summary, indent=2), flush=True)

    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    recent_scores: List[float] = []
    history: List[Dict[str, object]] = []
    epochs_without_improvement = 0

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_epoch(
            model, train_examples, global_node_features, ticker_to_id,
            optimizer, scheduler, device, args, random.Random(int(args.seed) + epoch),
        )
        val_metrics, _ = evaluate(
            model, val_examples, global_node_features, ticker_to_id, device, args
        )
        val_auc = float(val_metrics["auc"])
        recent_scores.append(val_auc if math.isfinite(val_auc) else 0.5)
        # A single epoch's validation AUC swings by several points at this sample
        # size; smoothing stops a lucky epoch from winning selection.
        smoothed = float(np.mean(recent_scores[-max(1, int(args.selection_smoothing)) :]))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_auc": val_auc,
            "val_auc_smoothed": smoothed,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={float(val_metrics['loss']):.6f} "
            f"val_auc={val_auc:.4f} val_auc_smooth={smoothed:.4f} "
            f"val_acc_0.5={val_metrics['directional_accuracy_at_0_5']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3g} "
            f"gate={float(torch.sigmoid(model.graph_gate).item()):.4f}",
            flush=True,
        )

        if smoothed > best_score:
            best_score, best_epoch, epochs_without_improvement = smoothed, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.early_stopping_patience):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, val_rows = evaluate(
        model, val_examples, global_node_features, ticker_to_id, device, args
    )
    test_metrics, test_rows = evaluate(
        model, test_examples, global_node_features, ticker_to_id, device, args
    )
    write_predictions(output_dir / f"val_predictions_{run_name}.csv", val_rows)
    write_predictions(output_dir / f"test_predictions_{run_name}.csv", test_rows)

    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": model_config,
            "graph_config": {
                "graph_dir": str(Path(args.graph_dir)),
                "forecast_tasks": str(Path(args.forecast_tasks)),
                "tickers": tickers,
                "ticker_to_id": ticker_to_id,
                "node_type_to_id": graph.node_type_to_id,
                "relation_to_id": graph.relation_to_id,
            },
            "history": history,
            "best_epoch": best_epoch,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "summary": summary,
        },
        checkpoint_path,
    )
    with (output_dir / f"training_metrics_{run_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary, "history": history, "best_epoch": best_epoch,
                "best_score": best_score, "val_metrics": val_metrics, "test_metrics": test_metrics,
            },
            handle, indent=2,
        )
    print(
        json.dumps(
            {
                "run_name": run_name,
                "best_epoch": best_epoch,
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],
                "test_acc_0_5": test_metrics["directional_accuracy_at_0_5"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved checkpoint to {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
