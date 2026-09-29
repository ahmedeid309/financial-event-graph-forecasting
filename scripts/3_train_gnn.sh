#!/bin/bash
#SBATCH --job-name=3_train_gnn
#SBATCH --output=logs/3_train_gnn_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 3 - Graph encoder and daily company representations
#
# Builds the supervision table (company-days with news, labelled with the
# same-day direction of the adjusted close), trains the temporal graph encoder
# with seeds 2021, 2022 and 2023, and exports one 64-dimensional representation
# per company and day from the seed with the highest validation AUC. The encoder
# sees news only; prices enter only through the labels.
#
# This step also fixes the chronological split (70/10/20 of the cutoff dates).
# Every later step reads the two boundary dates back from the exported file, so
# the encoder and all forecasters share one split.
#
# Design choices
#   --event_ticker_scope target_or_anchor   a company's own events plus the events
#                                           the extractor marked as affecting it
#   --readout_key_source both               the readout attention sees each event
#                                           before and after message passing
#   --selection_smoothing 1                 the epoch with the highest single
#                                           validation AUC is kept
#   three seeds                             GPU scatter-adds sum in arbitrary order,
#                                           so single runs differ; never report one
#
# Input   ekg_final/, dataset/full_history/ (labels)
# Output  gnn_data/forecast_task.csv; gnn_outputs_shared/ with the checkpoints and
#         metrics of the three seeds, run_summary.json, and
#         daily_ticker_embeddings.csv with its .metadata.json
#
# Remove gnn_outputs_shared/ before a rerun (run_chain.sh does this):
# summarize_runs.py averages every metrics file it finds.
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/3_train_gnn.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

# ------------------------------------------------------------- supervision
# Which (ticker, day) pairs are worth predicting, and what actually happened.
python gnn/build_forecast_tasks.py \
  --graph_dir ekg_final \
  --stock_dir dataset/full_history \
  --output gnn_data/forecast_task.csv \
  --task_source event_tickers \
  --news_conditioned \
  --min_news_days 5 \
  --news_window_days 1 \
  --label_price_column adj_close \
  --label_mode absolute \
  --min_train_rows_for_eval 50 \
  --horizon_trading_days 0 \
  --train_fraction 0.7 --val_fraction 0.1

# ------------------------------------------------------------ the real model
python gnn/train_gnn.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --output_dir gnn_outputs_shared \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 --hidden_dim 128 --num_layers 2 --num_heads 4 \
  --dropout 0.10 --graph_gate_init 2.0 \
  --readout_key_source both \
  --batch_size 32 --learning_rate 1e-3 --epochs 40 \
  --early_stopping_patience 20 --selection_smoothing 1 --device cuda \
  --run_name gnn_seed2021 --seed 2021

python gnn/train_gnn.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --output_dir gnn_outputs_shared \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 --hidden_dim 128 --num_layers 2 --num_heads 4 \
  --dropout 0.10 --graph_gate_init 2.0 \
  --readout_key_source both \
  --batch_size 32 --learning_rate 1e-3 --epochs 40 \
  --early_stopping_patience 20 --selection_smoothing 1 --device cuda \
  --run_name gnn_seed2022 --seed 2022

python gnn/train_gnn.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --output_dir gnn_outputs_shared \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 --hidden_dim 128 --num_layers 2 --num_heads 4 \
  --dropout 0.10 --graph_gate_init 2.0 \
  --readout_key_source both \
  --batch_size 32 --learning_rate 1e-3 --epochs 40 \
  --early_stopping_patience 20 --selection_smoothing 1 --device cuda \
  --run_name gnn_seed2023 --seed 2023

# ------------------------------------------- seed-averaged results table
python gnn/summarize_runs.py \
  --metrics_dir gnn_outputs_shared \
  --output_json gnn_outputs_shared/run_summary.json

# --------------------------------------- one 64-number summary per day
# All three seed checkpoints are saved. The exporter automatically selects the
# seed with the highest validation AUC. Test AUC is never used for selection.
python gnn/export_embeddings.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 \
  --best_validation_metrics_dir gnn_outputs_shared \
  --run_group gnn \
  --output_csv gnn_outputs_shared/daily_ticker_embeddings.csv \
  --embedding_prefix gnn \
  --normalization zscore \
  --device cuda

echo
echo "=== STAGE 3 OUTPUT: STAGE 4 READS THESE EXACT FILES ==="
ls -lh gnn_outputs_shared/daily_ticker_embeddings.csv
ls -lh gnn_outputs_shared/daily_ticker_embeddings.metadata.json
cat gnn_outputs_shared/daily_ticker_embeddings.metadata.json
