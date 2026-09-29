#!/bin/bash
#SBATCH --job-name=1_extract_events
#SBATCH --output=logs/1_extract_events_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=100G
#SBATCH --gpus=2

# =============================================================================
# Step 1 - Event extraction
#
# Runs the extraction pipeline (financial_ekg) over two adjacent partitions of
# dataset/news_2018_2023.csv, one per GPU, with Qwen3.5-35B-A3B (Q4_K_M GGUF)
# served in-process by llama-cpp-python at temperature zero. Each partition
# writes its settings, a snapshot of its articles, its event records, and a
# quality record per article to ekg_chunks/chunk_<start row>/.
#
# Set START_ROW and CHUNK before each submission; the job processes the
# partitions starting at START_ROW and START_ROW + CHUNK. The graph of the
# thesis merges these 29 disjoint partitions, which cover rows 0-36991:
#   1000 rows   start rows 0, 1000, ..., 9000          (10 partitions)
#   2000 rows   start rows 10000, 12000, ..., 30000    (11 partitions)
#   1000 rows   start rows 32000, 33000                (2 partitions)
#    500 rows   start rows 34000, 34500, ..., 36500    (6 partitions)
# A partition that reaches the end of the corpus stops there.
#
# Input   dataset/news_2018_2023.csv (data_prep/select_news.py)
#         models/Qwen3.5-35B-A3B-Q4_K_M.gguf
# Output  ekg_chunks/chunk_<row>/; one log per partition in logs/
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/1_extract_events.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

# First partition of this job and partition size (see the header).
START_ROW=0
CHUNK=1000

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROW_A=$START_ROW
ROW_B=$((START_ROW + CHUNK))

run_extract () {
  local gpu="$1" row="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -m financial_ekg.cli \
    --mode extract \
    --input_csv dataset/news_2018_2023.csv \
    --no-resume \
    --llm_backend llama_cpp \
    --llama_model_path models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
    --llama_n_ctx 32768 \
    --max_new_tokens 16000 \
    --llama_n_gpu_layers -1 \
    --llama_n_batch 512 \
    --llama_n_threads 4 \
    --disable_thinking \
    --adaptive_max_blocks 8 \
    --evidence_judge_batch_size 12 \
    --temporal_batch_size 8 \
    --stage6_batch_size 40 \
    --anchor_tickers T,INTC,AMD,CVX,BABA \
    --start_row "$row" \
    --limit "$CHUNK" \
    --output_dir "ekg_chunks/chunk_${row}" \
    > "logs/extract_chunk_${row}_${SLURM_JOB_ID}.out" 2>&1
}

run_extract 0 "$ROW_A" &
PID_A=$!
run_extract 1 "$ROW_B" &
PID_B=$!

echo "job ${SLURM_JOB_ID}: GPU0 rows ${ROW_A}-${ROW_B} -> ekg_chunks/chunk_${ROW_A}"
echo "job ${SLURM_JOB_ID}: GPU1 rows ${ROW_B}-$((ROW_B + CHUNK)) -> ekg_chunks/chunk_${ROW_B}"

wait $PID_A
STATUS_A=$?
wait $PID_B
STATUS_B=$?

echo "GPU0 exited with status $STATUS_A"
echo "GPU1 exited with status $STATUS_B"

if [ $STATUS_A -ne 0 ] || [ $STATUS_B -ne 0 ]; then
  exit 1
fi
