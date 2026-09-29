from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from financial_ekg.utils.text import clean_text, normalize_ticker, safe_str

def get_text_embeddings(
    texts: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
    fallback_dim: int = 64,
) -> np.ndarray:
    """Encode text with SentenceTransformers or a TF-IDF fallback.

    Args:
        texts: Text strings to embed.
        model_name: SentenceTransformer model name or local path.
        device: Device string for embedding inference.
        batch_size: Embedding batch size.
        fallback_dim: Embedding dimension used by the fallback encoder.

    Returns:
        Two-dimensional float32 embedding array.
    """
    if len(texts) == 0:
        return np.zeros((0, fallback_dim), dtype=np.float32)
    embedding_model_key = safe_str(model_name).strip().lower()
    if embedding_model_key in {"", "none", "skip", "zero"}:
        return np.zeros((len(texts), fallback_dim), dtype=np.float32)
    try:
        if embedding_model_key in {"tfidf", "tf-idf", "sklearn", "fallback"}:
            raise RuntimeError("forced TF-IDF/SVD embedding fallback")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device if device.startswith("cuda") else "cpu")
        emb = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)
    except Exception as e:
        print(f"WARNING: SentenceTransformer failed ({e}). Falling back to TF-IDF + SVD embeddings.", file=sys.stderr)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        max_features = max(500, fallback_dim * 10)
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(list(texts))
        n_components = min(fallback_dim, max(1, min(X.shape) - 1))
        if n_components < 2:
            arr = X.toarray().astype(np.float32)
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            arr = svd.fit_transform(X).astype(np.float32)
        arr = normalize(arr)
        if arr.shape[1] < fallback_dim:
            arr = np.pad(arr, ((0, 0), (0, fallback_dim - arr.shape[1])), mode="constant")
        return arr.astype(np.float32)


def add_event_embeddings(event_df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create and save embeddings for extracted events.

    Args:
        event_df: Event dataframe.
        args: Parsed CLI arguments.
        out_dir: Output directory.

    Returns:
        Tuple of unchanged event dataframe and event embedding matrix.
    """
    if len(event_df) == 0:
        return event_df, np.zeros((0, args.feature_embedding_dim), dtype=np.float32)
    texts = (
        event_df["ticker"].astype(str)
        + " | "
        + event_df["event_type"].astype(str)
        + " | "
        + event_df["event_description"].astype(str)
    ).tolist()
    emb = get_text_embeddings(
        texts,
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        fallback_dim=max(args.feature_embedding_dim, 64),
    )
    # Save full event embeddings separately.
    np.save(out_dir / "event_embeddings.npy", emb)
    return event_df, emb


def fit_feature_dim(features: np.ndarray, feature_dim: int) -> np.ndarray:
    """Pad or truncate a feature matrix to a target dimension.

    Args:
        features: Two-dimensional feature matrix to normalize.
        feature_dim: Required output column count.

    Returns:
        Float32 feature matrix with exactly `feature_dim` columns.
    """
    if features.ndim != 2:
        features = np.asarray(features).reshape((features.shape[0], -1))
    if features.shape[1] == feature_dim:
        return features.astype(np.float32)
    if features.shape[1] > feature_dim:
        return features[:, :feature_dim].astype(np.float32)
    return np.pad(features, ((0, 0), (0, feature_dim - features.shape[1])), mode="constant").astype(np.float32)


def node_feature_text(
    row: pd.Series,
    articles_by_id: Dict[str, Dict[str, Any]],
    events_by_id: Dict[str, Dict[str, Any]],
) -> str:
    """Build text used for non-event node features.

    Args:
        row: Node row from `nodes.csv`.
        articles_by_id: Article metadata indexed by article ID.
        events_by_id: Event metadata indexed by event ID.

    Returns:
        Compact text representation used as embedding input for the node.
    """
    node_id = clean_text(row.get("node_id", ""))
    node_type = clean_text(row.get("node_type", ""))
    name = clean_text(row.get("name", ""))
    ticker = normalize_ticker(row.get("ticker", ""))
    date = clean_text(row.get("date", ""))

    if node_type == "Article" and node_id.startswith("article:"):
        article_id = node_id.split(":", 1)[1]
        meta = articles_by_id.get(article_id, {})
        return " | ".join(
            part
            for part in [
                "Article",
                clean_text(meta.get("_ticker", ticker)),
                clean_text(meta.get("_date", date)),
                clean_text(meta.get("_title", name)),
            ]
            if part
        )

    if node_type == "Event" and node_id.startswith("event:"):
        event_id = node_id.split(":", 1)[1]
        ev = events_by_id.get(event_id, {})
        return " | ".join(
            part
            for part in [
                "Event",
                normalize_ticker(ev.get("ticker", ticker)),
                clean_text(ev.get("event_type", "")),
                clean_text(ev.get("main_company", "")),
                clean_text(ev.get("event_description", name)),
            ]
            if part
        )

    return " | ".join(part for part in [node_type, ticker, date, name] if part)


def build_and_save_node_features(
    nodes_df: pd.DataFrame,
    articles: pd.DataFrame,
    event_df: pd.DataFrame,
    event_embeddings: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create an aligned feature matrix for every graph node.

    `node_features.npy[i]` corresponds exactly to `nodes.csv.iloc[i]`.
    Event rows reuse `event_embeddings.npy` whenever dimensions are compatible;
    all other nodes receive text embeddings from their node labels/metadata.

    Args:
        nodes_df: Node table produced by graph construction.
        articles: Article dataframe used for article-node metadata.
        event_df: Event dataframe used for event-node metadata.
        event_embeddings: Event embedding matrix aligned to `event_df`.
        args: Parsed CLI arguments controlling embedding options.
        out_dir: Output directory where feature artifacts are written.

    Returns:
        Tuple containing the node dataframe with feature metadata columns and
        the aligned node feature matrix.
    """
    if nodes_df.empty:
        feature_dim = int(getattr(args, "feature_embedding_dim", 64) or 64)
        features = np.zeros((0, feature_dim), dtype=np.float32)
        np.save(out_dir / "node_features.npy", features)
        return nodes_df, features

    articles_by_id = articles.set_index("article_id", drop=False).to_dict(orient="index") if "article_id" in articles.columns else {}
    events_by_id = event_df.set_index("event_id", drop=False).to_dict(orient="index") if "event_id" in event_df.columns else {}

    fallback_dim = int(getattr(args, "feature_embedding_dim", 64) or 64)
    if isinstance(event_embeddings, np.ndarray) and event_embeddings.ndim == 2 and event_embeddings.shape[1] > 0:
        feature_dim = int(event_embeddings.shape[1])
    else:
        feature_dim = max(fallback_dim, 64)

    feature_texts = [node_feature_text(row, articles_by_id, events_by_id) for _, row in nodes_df.iterrows()]
    node_features = get_text_embeddings(
        feature_texts,
        model_name=getattr(args, "embedding_model", ""),
        device=getattr(args, "device", "cpu"),
        batch_size=int(getattr(args, "embedding_batch_size", 64) or 64),
        fallback_dim=feature_dim,
    )
    node_features = fit_feature_dim(node_features, feature_dim)
    feature_sources = ["node_text_embedding"] * len(nodes_df)

    if isinstance(event_embeddings, np.ndarray) and event_embeddings.ndim == 2 and event_embeddings.shape[0] == len(event_df):
        event_embeddings_fit = fit_feature_dim(event_embeddings, feature_dim)
        event_id_to_embedding_idx = {
            clean_text(event_id): idx
            for idx, event_id in enumerate(event_df["event_id"].astype(str).tolist())
        } if "event_id" in event_df.columns else {}
        for node_idx, node_id in enumerate(nodes_df["node_id"].astype(str).tolist()):
            if not node_id.startswith("event:"):
                continue
            event_id = node_id.split(":", 1)[1]
            emb_idx = event_id_to_embedding_idx.get(event_id)
            if emb_idx is None:
                continue
            node_features[node_idx] = event_embeddings_fit[emb_idx]
            feature_sources[node_idx] = "event_embeddings.npy"

    nodes_df = nodes_df.copy()
    nodes_df["feature_row"] = np.arange(len(nodes_df), dtype=np.int64)
    nodes_df["feature_source"] = feature_sources
    nodes_df["feature_dim"] = feature_dim

    np.save(out_dir / "node_features.npy", node_features.astype(np.float32))
    pd.DataFrame(
        {
            "feature_row": np.arange(len(nodes_df), dtype=np.int64),
            "node_id": nodes_df["node_id"].astype(str),
            "node_type": nodes_df["node_type"].astype(str),
            "feature_source": feature_sources,
            "feature_text": feature_texts,
        }
    ).to_csv(out_dir / "node_features_manifest.csv", index=False)
    return nodes_df, node_features.astype(np.float32)
