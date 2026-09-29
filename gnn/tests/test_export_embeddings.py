#!/usr/bin/env python3
"""Tests for validation-only checkpoint selection during embedding export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export_embeddings import select_best_validation_checkpoint  # noqa: E402


def write_run(root: Path, seed: int, val_auc: float, test_auc: float) -> None:
    run_name = f"gnn_seed{seed}"
    payload = {
        "summary": {"run_name": run_name},
        "val_metrics": {"auc": val_auc},
        "test_metrics": {"auc": test_auc},
    }
    (root / f"training_metrics_{run_name}.json").write_text(json.dumps(payload))
    checkpoint = root / "checkpoints" / f"event_gnn_{run_name}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.touch()


def test_selects_highest_validation_auc_not_highest_test_auc(tmp_path: Path) -> None:
    write_run(tmp_path, 2021, val_auc=0.61, test_auc=0.90)
    write_run(tmp_path, 2022, val_auc=0.65, test_auc=0.50)
    write_run(tmp_path, 2023, val_auc=0.63, test_auc=0.80)

    checkpoint, selection = select_best_validation_checkpoint(tmp_path, "gnn")

    assert checkpoint.name == "event_gnn_gnn_seed2022.pt"
    assert selection["run_name"] == "gnn_seed2022"
    assert selection["validation_auc"] == 0.65
