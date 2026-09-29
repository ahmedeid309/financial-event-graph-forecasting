from __future__ import annotations

import argparse
import time

from financial_ekg.utils.text import ensure_dir

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed CLI namespace.
    """
    p = argparse.ArgumentParser(description="Build a Financial Event Knowledge Graph from financial news CSV.")

    p.add_argument(
        "--mode",
        choices=["extract", "build"],
        default="extract",
        help=(
            "extract: only run article-level event extraction for a CSV chunk; "
            "build: merge extracted chunks and build one global graph."
        ),
    )
    p.add_argument("--input_csv", default="", help="Path to input news CSV. Required for --mode extract.")
    p.add_argument("--output_dir", required=True, help="Directory where outputs will be written.")
    p.add_argument("--limit", type=int, default=0, help="Process N CSV rows. 0 means all rows from start_row onward.")
    p.add_argument("--start_row", type=int, default=0, help="Start reading at this CSV data row, after the header. Used for chunked extraction.")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume extraction from existing event_extractions.jsonl in output_dir. Use --no-resume to overwrite.",
    )
    p.add_argument(
        "--extraction_dirs",
        nargs="*",
        default=[],
        help="Explicit chunk output directories produced by --mode extract. Optional alternative to --chunks_parent_dir.",
    )
    p.add_argument(
        "--chunks_parent_dir",
        default="",
        help="Parent directory containing chunk_* extraction folders. Used by --mode build.",
    )

    p.add_argument("--extractor", choices=["llm"], default="llm", help="Compatibility option; extraction is LLM-only.")
    p.add_argument(
        "--llm_backend",
        choices=["transformers", "llama_cpp"],
        default="transformers",
        help="Inference backend. transformers uses Hugging Face weights; llama_cpp uses a local GGUF file.",
    )
    p.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face instruct model for extraction.")
    p.add_argument("--load_in_4bit", action="store_true", help="Use bitsandbytes 4-bit quantization for the LLM.")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu. Used for LLM and embeddings.")
    p.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum generated tokens for extraction.")
    p.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Ask Qwen-style thinking chat templates to skip the reasoning block and return the final JSON directly.",
    )
    p.add_argument("--llama_model_path", default="", help="Local .gguf model path used when --llm_backend llama_cpp.")
    p.add_argument("--llama_n_ctx", type=int, default=8192, help="llama.cpp context window.")
    p.add_argument("--llama_n_gpu_layers", type=int, default=-1, help="llama.cpp GPU-offloaded layers. -1 offloads all layers.")
    p.add_argument("--llama_n_batch", type=int, default=512, help="llama.cpp prompt-processing batch size.")
    p.add_argument("--llama_n_threads", type=int, default=0, help="llama.cpp CPU thread count. 0 lets llama.cpp choose.")
    p.add_argument("--llama_verbose", action="store_true", help="Show verbose llama.cpp runtime logs.")
    p.add_argument("--adaptive_block_chars", type=int, default=2500, help="Approximate character budget for each adaptive evidence block.")
    p.add_argument("--adaptive_max_blocks", type=int, default=8, help="Maximum adaptive evidence-block extraction calls per article.")
    p.add_argument("--evidence_judge_batch_size", type=int, default=12, help="Maximum candidate events judged per evidence-admissibility LLM call.")
    p.add_argument("--temporal_batch_size", type=int, default=8, help="Maximum finalized events processed per evidence-first temporal LLM batch.")
    p.add_argument("--stage6_batch_size", type=int, default=40, help="Maximum finalized events classified per Stage 6 anchor-impact call.")
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model for event embeddings and semantic graph links.")
    p.add_argument("--embedding_batch_size", type=int, default=64)
    p.add_argument("--feature_embedding_dim", type=int, default=64, help="Fallback event embedding dimension when no events are available.")

    p.add_argument("--same_event_threshold", type=float, default=0.92, help="Cosine similarity threshold for SAME_AS event edges.")
    p.add_argument("--related_threshold", type=float, default=0.78, help="Cosine similarity threshold for RELATED_TO event edges.")
    p.add_argument("--related_neighbors", type=int, default=5, help="Nearest semantic neighbors considered within each ticker/event_type group.")
    p.add_argument("--semantic_link_max_days", type=int, default=30, help="Maximum date distance for semantic event links.")
    p.add_argument(
        "--anchor_tickers",
        default="T,INTC,AMD,CVX,BABA",
        help="Comma-separated forecasting target tickers used as canonical anchor companies.",
    )
    p.add_argument(
        "--anchor_aliases_json",
        default="",
        help=(
            "Optional JSON file overriding/adding anchor aliases. Accepts either "
            "{\"INTC\": {\"company\": \"Intel\", \"aliases\": [...]}} or a list of objects."
        ),
    )
    p.add_argument(
        "--add_generic_company_edges",
        action="store_true",
        help="Also write legacy generic Event->Company INVOLVES_COMPANY edges in addition to specific relation edges.",
    )
    p.add_argument(
        "--semantic_entity_resolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve otherwise-unmatched Entity nodes to Company nodes using high-threshold text-embedding similarity.",
    )
    p.add_argument(
        "--entity_resolution_threshold",
        type=float,
        default=0.96,
        help="Cosine threshold for --semantic_entity_resolution.",
    )
    p.add_argument(
        "--export_hetero_tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export node_features.npy plus heterogeneous graph tensor files for PyG/DGL-style loading.",
    )
    p.add_argument(
        "--hetero_add_reverse_edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include reverse relations in hetero tensor exports for directional message passing.",
    )
    p.add_argument(
        "--export_torch_hetero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export PyTorch/PyG .pt hetero graph files. Disabled by default because torch import can be slow on some clusters.",
    )

    p.add_argument("--calendar", choices=["business", "calendar", "news-only"], default="business", help=argparse.SUPPRESS)
    p.add_argument("--add_rolling", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rolling_windows", type=int, nargs="+", default=[3, 7], help=argparse.SUPPRESS)
    p.add_argument("--add_decay", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--decay_lambda", type=float, default=0.30, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    """Run extract and/or build mode from the command line.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    if args.mode == "extract" and args.llm_backend == "llama_cpp" and not args.llama_model_path:
        raise ValueError("--llama_model_path is required when --llm_backend llama_cpp")
    out_dir = ensure_dir(args.output_dir)
    limit = args.limit if args.limit and args.limit > 0 else None

    t0 = time.time()

    if args.mode == "extract":
        from financial_ekg.io.articles import load_articles, save_articles_snapshot
        from financial_ekg.io.run_config import save_config
        from financial_ekg.pipeline.extraction import run_extraction

        if not args.input_csv:
            raise ValueError("--input_csv is required for --mode extract")
        articles = load_articles(args.input_csv, limit=limit, start_row=max(0, args.start_row))
        print(f"Loaded {len(articles)} usable articles from {args.input_csv} starting at row {args.start_row}")
        save_articles_snapshot(articles, out_dir)

        event_df = run_extraction(args, articles, out_dir)
        print(f"Extracted {len(event_df)} events")

        save_config(args, out_dir, articles, event_df)
        print(f"Extraction-only mode done in {(time.time() - t0):.1f}s. Outputs written to: {out_dir}")
        return

    if args.mode == "build":
        from financial_ekg.graph.builder import run_global_build
        from financial_ekg.io.chunks import discover_extraction_dirs, load_extracted_chunks

        extraction_dirs = args.extraction_dirs
        if args.chunks_parent_dir:
            extraction_dirs = discover_extraction_dirs(args.chunks_parent_dir)
            print(f"Discovered {len(extraction_dirs)} extraction chunk directories in {args.chunks_parent_dir}")
        if not extraction_dirs:
            raise ValueError("--chunks_parent_dir or --extraction_dirs is required for --mode build")
        articles, event_df = load_extracted_chunks(extraction_dirs)
        run_global_build(args, articles, event_df, out_dir)
        print(f"Global build mode done in {(time.time() - t0):.1f}s. Outputs written to: {out_dir}")
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
