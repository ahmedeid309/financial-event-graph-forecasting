#!/bin/bash
#SBATCH --job-name=2_build_graph
#SBATCH --output=logs/2_build_graph_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 2 - Graph construction
#
# Merges every partition in ekg_chunks/ into one event knowledge graph. Events
# and entities are embedded with BGE-large-en-v1.5, clustered, and fused, so that
# mentions of one company or one event in different articles become one node.
# Submit only after all extraction jobs have finished; the partitions are read
# directly.
#
# Input   ekg_chunks/chunk_*/, models/bge-large-en-v1.5
# Output  ekg_final/: articles.csv, events.csv, nodes.csv, edges.csv, event and
#         entity clusters with their members, event_embeddings.npy,
#         node_features.npy, graph.graphml, heterogeneous graph tensors,
#         fusion_report.json, run_config.json
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/2_build_graph.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "chunks being merged:"
ls -d ekg_chunks/chunk_* | sed 's#.*chunk_##' | sort -n | tr '\n' ' '
echo

python -m financial_ekg.cli \
  --mode build \
  --chunks_parent_dir ekg_chunks \
  --output_dir ekg_final \
  --device cuda \
  --embedding_model models/bge-large-en-v1.5

echo
echo "=== graph built ==="
wc -l ekg_final/events.csv ekg_final/articles.csv ekg_final/edges.csv
