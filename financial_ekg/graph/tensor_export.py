from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from financial_ekg.utils.anchors import reverse_edge_type
from financial_ekg.utils.text import clean_text, safe_str

def safe_tensor_key(*parts: Any) -> str:
    """Build a stable key for NPZ tensor exports.

    Args:
        *parts: Text fragments that identify a node or edge tensor.

    Returns:
        Sanitized key suitable for use inside `.npz` and metadata files.
    """
    return "__".join(re.sub(r"[^A-Za-z0-9_]+", "_", safe_str(part)).strip("_") or "UNKNOWN" for part in parts)


def export_heterogeneous_graph_tensors(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_features: np.ndarray,
    out_dir: Path,
    add_reverse_edges: bool = True,
    export_torch: bool = False,
) -> None:
    """Export a framework-neutral heterogeneous graph tensor package.

    Always writes:
      - `hetero_graph_tensors.npz`
      - `hetero_graph_metadata.json`

    When `export_torch=True` and PyTorch is available, also writes
    `hetero_graph_tensors.pt`. When PyTorch Geometric is available, also writes
    `hetero_graph.pt`.

    Args:
        nodes_df: Node dataframe with feature-row metadata.
        edges_df: Edge dataframe containing source, target, type, and weight.
        node_features: Feature matrix aligned to `nodes_df`.
        out_dir: Output directory for tensor artifacts.
        add_reverse_edges: Whether to add reverse relations to the exported
            heterogeneous edge dictionaries.
        export_torch: Whether to also export PyTorch/PyG `.pt` artifacts.

    Returns:
        None.
    """
    if nodes_df.empty:
        return

    node_type_to_ids: Dict[str, List[str]] = {}
    node_type_to_global: Dict[str, List[int]] = {}
    node_mapping: Dict[str, Dict[str, int]] = {}
    for global_idx, row in nodes_df.reset_index(drop=True).iterrows():
        node_type = clean_text(row.get("node_type", "")) or "Unknown"
        node_id = clean_text(row.get("node_id", ""))
        node_type_to_ids.setdefault(node_type, []).append(node_id)
        node_type_to_global.setdefault(node_type, []).append(int(global_idx))
        node_mapping.setdefault(node_type, {})[node_id] = len(node_type_to_ids[node_type]) - 1

    node_id_to_type = {
        clean_text(row.get("node_id", "")): clean_text(row.get("node_type", "")) or "Unknown"
        for _, row in nodes_df.iterrows()
    }

    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {
        "feature_dim": int(node_features.shape[1]) if node_features.ndim == 2 else 0,
        "node_types": {},
        "edge_types": {},
        "add_reverse_edges": bool(add_reverse_edges),
        "export_torch": bool(export_torch),
    }

    for node_type, ids in node_type_to_ids.items():
        global_indices = np.asarray(node_type_to_global[node_type], dtype=np.int64)
        key = safe_tensor_key("x", node_type)
        arrays[key] = node_features[global_indices].astype(np.float32)
        arrays[safe_tensor_key("global_node_index", node_type)] = global_indices
        metadata["node_types"][node_type] = {
            "x_key": key,
            "node_ids": ids,
            "global_node_index_key": safe_tensor_key("global_node_index", node_type),
        }

    edge_groups: Dict[Tuple[str, str, str], List[Tuple[int, int, float]]] = {}
    for _, edge in edges_df.iterrows():
        source = clean_text(edge.get("source", ""))
        target = clean_text(edge.get("target", ""))
        edge_type = clean_text(edge.get("edge_type", "")) or "RELATED_TO"
        source_type = node_id_to_type.get(source)
        target_type = node_id_to_type.get(target)
        if not source_type or not target_type:
            continue
        src_idx = node_mapping[source_type].get(source)
        dst_idx = node_mapping[target_type].get(target)
        if src_idx is None or dst_idx is None:
            continue
        try:
            weight = float(edge.get("weight", 1.0))
        except Exception:
            weight = 1.0
        edge_groups.setdefault((source_type, edge_type, target_type), []).append((src_idx, dst_idx, weight))
        if add_reverse_edges:
            edge_groups.setdefault((target_type, reverse_edge_type(edge_type), source_type), []).append((dst_idx, src_idx, weight))

    for edge_key, triples in edge_groups.items():
        source_type, edge_type, target_type = edge_key
        edge_index = np.asarray([[src, dst] for src, dst, _ in triples], dtype=np.int64).T
        edge_weight = np.asarray([weight for _, _, weight in triples], dtype=np.float32)
        tensor_key = safe_tensor_key("edge_index", source_type, edge_type, target_type)
        weight_key = safe_tensor_key("edge_weight", source_type, edge_type, target_type)
        arrays[tensor_key] = edge_index
        arrays[weight_key] = edge_weight
        metadata["edge_types"]["|".join(edge_key)] = {
            "source_type": source_type,
            "edge_type": edge_type,
            "target_type": target_type,
            "edge_index_key": tensor_key,
            "edge_weight_key": weight_key,
            "edge_count": int(edge_index.shape[1]),
        }

    np.savez_compressed(out_dir / "hetero_graph_tensors.npz", **arrays)
    with (out_dir / "hetero_graph_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if not export_torch:
        return

    try:
        import torch

        x_dict = {
            node_type: torch.tensor(arrays[info["x_key"]], dtype=torch.float32)
            for node_type, info in metadata["node_types"].items()
        }
        edge_index_dict = {}
        edge_weight_dict = {}
        for key, info in metadata["edge_types"].items():
            edge_tuple = (info["source_type"], info["edge_type"], info["target_type"])
            edge_index_dict[edge_tuple] = torch.tensor(arrays[info["edge_index_key"]], dtype=torch.long)
            edge_weight_dict[edge_tuple] = torch.tensor(arrays[info["edge_weight_key"]], dtype=torch.float32)
        torch.save(
            {
                "x_dict": x_dict,
                "edge_index_dict": edge_index_dict,
                "edge_weight_dict": edge_weight_dict,
                "node_ids": {node_type: info["node_ids"] for node_type, info in metadata["node_types"].items()},
                "metadata": metadata,
            },
            out_dir / "hetero_graph_tensors.pt",
        )

        try:
            from torch_geometric.data import HeteroData

            data = HeteroData()
            for node_type, x in x_dict.items():
                data[node_type].x = x
                data[node_type].node_id = metadata["node_types"][node_type]["node_ids"]
            for edge_tuple, edge_index in edge_index_dict.items():
                data[edge_tuple].edge_index = edge_index
                data[edge_tuple].edge_weight = edge_weight_dict[edge_tuple]
            torch.save(data, out_dir / "hetero_graph.pt")
        except Exception as exc:
            print(f"WARNING: PyTorch Geometric HeteroData export skipped: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: PyTorch tensor export skipped: {exc}", file=sys.stderr)
