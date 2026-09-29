#!/bin/bash
#SBATCH --job-name=4b_text_baseline
#SBATCH --output=logs/4b_text_baseline_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 4b - Direct-text controls: TEXT64 and TEXT1024
#
# Encodes the same articles directly with BGE-large-en-v1.5 (1024 dimensions),
# pools them per company and day with the graph's window and decay (one day,
# weight exp(-0.10 * age in days)), standardises them on training rows, and feeds
# them through the fusion pathway of GRAPH:
#   TEXT64    projected to 64 dimensions by a PCA fitted on training rows
#   TEXT1024  all 1024 dimensions
# Both use the adapter width 64 of GRAPH and load the PRICE checkpoints of step 4,
# so PRICE, TEXT and GRAPH are paired within each seed. A parity check stops the
# job if a text file's rows differ from those of its graph twin.
#
# Unlike the graph, which routes events to every company they concern, articles
# are assigned to the company they are filed under, and the text representation
# is not trained on prices.
#
# Input   ekg_final/articles.csv, models/bge-large-en-v1.5, outputs of steps 3 and 4
# Output  gnn_outputs_text/daily_ticker_text_embeddings.csv,
#         TimeXer/dataset/stock/<TK>_text64.csv and <TK>_text1024.csv,
#         TimeXer/result_text_<TK>.txt
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/4b_train_text_baseline.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

echo "=== STAGE 4b INPUTS ==="
test -s ekg_final/articles.csv || {
  echo "ERROR: ekg_final/articles.csv is missing or empty. Run stage 2."
  exit 1
}
test -s gnn_data/forecast_task.csv || {
  echo "ERROR: gnn_data/forecast_task.csv is missing. Run stage 3."
  exit 1
}
test -s gnn_outputs_shared/daily_ticker_embeddings.csv || {
  echo "ERROR: stage 3 embedding CSV is missing. The shared split lives there."
  exit 1
}

# --------------------------------------------------------------- ONE SPLIT
# Read the same two boundary dates stage 4 reads, from the same file. The text
# arm must never derive its own split: if it did, the two arms would be scored
# over different test periods and the difference between them would mean
# nothing, with no error anywhere to say so.
read -r TRAIN_END VAL_END < <(python - <<'EOF'
import csv
last = {}
with open("gnn_outputs_shared/daily_ticker_embeddings.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        split = row["split"]
        date = row["date"]
        if date > last.get(split, ""):
            last[split] = date
missing = [name for name in ("train", "val", "test") if name not in last]
if missing:
    raise SystemExit(f"ERROR: embedding CSV has no rows for split(s): {', '.join(missing)}")
if not last["train"] < last["val"] < last["test"]:
    raise SystemExit("ERROR: embedding splits are not chronologically ordered.")
print(last["train"], last["val"])
EOF
)

echo
echo "=== ONE SPLIT, INHERITED FROM STAGE 3 ==="
echo "train ends: ${TRAIN_END}   val ends: ${VAL_END}   test: everything after"
echo "Identical to stage 4. Both arms are scored over the same days."

# ------------------------------------------------- TEXT EMBEDDING EXPORT
# All five tickers together, not one at a time. join_embeddings_to_stock.py
# clips the price rows to the embedding file's own min and max date, so a
# single-ticker export can end days early (each ticker's last news day differs)
# and the text arm would silently be scored on fewer samples than the graph
# arm. The parity check below enforces this rather than trusting it.
echo
echo "=== ENCODING ARTICLES (no events, no graph, no GNN) ==="
python gnn/export_text_embeddings.py \
  --forecast_tasks gnn_data/forecast_task.csv \
  --articles_csv ekg_final/articles.csv \
  --tickers T,INTC,AMD,CVX,BABA \
  --horizon_trading_days 0 \
  --window_days 1 \
  --decay_lambda 0.10 \
  --embedding_model models/bge-large-en-v1.5 \
  --expect_dim 1024 \
  --device cuda \
  --batch_size 64 \
  --max_chars_per_chunk 1800 \
  --max_chunks_per_article 12 \
  --normalization zscore \
  --cache_path gnn_data/text_embedding_cache.npz \
  --output_csv gnn_outputs_text/daily_ticker_text_embeddings.csv

echo
cat gnn_outputs_text/daily_ticker_text_embeddings.metadata.json

# ------------------------------------------------------------------ JOINS
# Same script, same graph_target_lead, same zero-fill convention as stage 4.
# The 1024 -> 64 projection is fitted on training rows only, inside the join.
echo
echo "=== JOIN: 64-d arm (width-matched to the graph block) ==="
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_text/daily_ticker_text_embeddings.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --pca_components 64 \
  --suffix _text64

echo
echo "=== JOIN: 1024-d arm (full encoder output) ==="
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_text/daily_ticker_text_embeddings.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --suffix _text1024

# ------------------------------------------------------------ PARITY GATE
# A text file with different rows than its graph twin produces a perfectly
# plausible MSE that cannot be compared to anything. Fail here instead.
echo
echo "=== PARITY: every text file must cover the graph file's exact dates ==="
python - <<'EOF'
import csv
import sys

def dates(path):
    with open(path, newline="") as handle:
        return [row["date"] for row in csv.DictReader(handle)]

problems = []
for ticker in ("T", "INTC", "AMD", "CVX", "BABA"):
    graph = dates(f"TimeXer/dataset/stock/{ticker}_graph.csv")
    for suffix in ("text64", "text1024"):
        text = dates(f"TimeXer/dataset/stock/{ticker}_{suffix}.csv")
        if text == graph:
            print(f"  OK   {ticker}_{suffix}.csv  {len(text)} rows, identical dates to {ticker}_graph.csv")
        else:
            problems.append(
                f"  FAIL {ticker}_{suffix}.csv has {len(text)} rows vs "
                f"{len(graph)} in {ticker}_graph.csv"
            )
if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(
        "ERROR: text and graph arms do not cover the same samples, so their MSEs "
        "are not comparable. Check the embedding file's date range."
    )
EOF

echo
echo "=== row counts ==="
wc -l TimeXer/dataset/stock/T_graph.csv    TimeXer/dataset/stock/T_text64.csv    TimeXer/dataset/stock/T_text1024.csv
wc -l TimeXer/dataset/stock/INTC_graph.csv TimeXer/dataset/stock/INTC_text64.csv TimeXer/dataset/stock/INTC_text1024.csv
wc -l TimeXer/dataset/stock/AMD_graph.csv TimeXer/dataset/stock/AMD_text64.csv TimeXer/dataset/stock/AMD_text1024.csv
wc -l TimeXer/dataset/stock/CVX_graph.csv TimeXer/dataset/stock/CVX_text64.csv TimeXer/dataset/stock/CVX_text1024.csv
wc -l TimeXer/dataset/stock/BABA_graph.csv TimeXer/dataset/stock/BABA_text64.csv TimeXer/dataset/stock/BABA_text1024.csv

cd "$REPO_ROOT/TimeXer"

echo
echo "=== PRICE-ONLY PARENTS FROM STAGE 4 (loaded, never retrained) ==="
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for T seed 11. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for T seed 22. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for T seed 33. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for T seed 44. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for T seed 55. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for INTC seed 11. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for INTC seed 22. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for INTC seed 33. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for INTC seed 44. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for INTC seed 55. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for AMD seed 11. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for AMD seed 22. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for AMD seed 33. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for AMD seed 44. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for AMD seed 55. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for CVX seed 11. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for CVX seed 22. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for CVX seed 33. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for CVX seed 44. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for CVX seed 55. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for BABA seed 11. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for BABA seed 22. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for BABA seed 33. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for BABA seed 44. Run stage 4 first."
  exit 1
}
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth || {
  echo "ERROR: missing price-only parent for BABA seed 55. Run stage 4 first."
  exit 1
}

# ############################################################################
# TICKER T
# ############################################################################

# ==================================================== T, TEXT64, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text64.csv \
  --model_id TEXT64_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT64, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text64.csv \
  --model_id TEXT64_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT64, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text64.csv \
  --model_id TEXT64_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT64, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text64.csv \
  --model_id TEXT64_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT64, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text64.csv \
  --model_id TEXT64_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT1024, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text1024.csv \
  --model_id TEXT1024_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT1024, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text1024.csv \
  --model_id TEXT1024_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT1024, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text1024.csv \
  --model_id TEXT1024_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT1024, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text1024.csv \
  --model_id TEXT1024_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ==================================================== T, TEXT1024, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_text1024.csv \
  --model_id TEXT1024_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_text_T.txt --no-plot_data_splits

# ############################################################################
# TICKER INTC
# ############################################################################

# ==================================================== INTC, TEXT64, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text64.csv \
  --model_id TEXT64_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT64, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text64.csv \
  --model_id TEXT64_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT64, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text64.csv \
  --model_id TEXT64_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT64, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text64.csv \
  --model_id TEXT64_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT64, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text64.csv \
  --model_id TEXT64_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT1024, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text1024.csv \
  --model_id TEXT1024_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT1024, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text1024.csv \
  --model_id TEXT1024_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT1024, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text1024.csv \
  --model_id TEXT1024_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT1024, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text1024.csv \
  --model_id TEXT1024_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ==================================================== INTC, TEXT1024, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_text1024.csv \
  --model_id TEXT1024_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_text_INTC.txt --no-plot_data_splits

# ############################################################################
# TICKER AMD
# ############################################################################

# ==================================================== AMD, TEXT64, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text64.csv \
  --model_id TEXT64_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT64, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text64.csv \
  --model_id TEXT64_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT64, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text64.csv \
  --model_id TEXT64_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT64, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text64.csv \
  --model_id TEXT64_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT64, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text64.csv \
  --model_id TEXT64_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT1024, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text1024.csv \
  --model_id TEXT1024_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT1024, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text1024.csv \
  --model_id TEXT1024_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT1024, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text1024.csv \
  --model_id TEXT1024_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT1024, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text1024.csv \
  --model_id TEXT1024_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ==================================================== AMD, TEXT1024, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_text1024.csv \
  --model_id TEXT1024_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_text_AMD.txt --no-plot_data_splits

# ############################################################################
# TICKER CVX
# ############################################################################

# ==================================================== CVX, TEXT64, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text64.csv \
  --model_id TEXT64_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT64, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text64.csv \
  --model_id TEXT64_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT64, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text64.csv \
  --model_id TEXT64_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT64, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text64.csv \
  --model_id TEXT64_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT64, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text64.csv \
  --model_id TEXT64_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT1024, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text1024.csv \
  --model_id TEXT1024_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT1024, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text1024.csv \
  --model_id TEXT1024_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT1024, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text1024.csv \
  --model_id TEXT1024_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT1024, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text1024.csv \
  --model_id TEXT1024_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ==================================================== CVX, TEXT1024, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_text1024.csv \
  --model_id TEXT1024_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_text_CVX.txt --no-plot_data_splits

# ############################################################################
# TICKER BABA
# ############################################################################

# ==================================================== BABA, TEXT64, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text64.csv \
  --model_id TEXT64_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT64, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text64.csv \
  --model_id TEXT64_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT64, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text64.csv \
  --model_id TEXT64_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT64, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text64.csv \
  --model_id TEXT64_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT64, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text64.csv \
  --model_id TEXT64_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des TEXT64_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT1024, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text1024.csv \
  --model_id TEXT1024_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT1024, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text1024.csv \
  --model_id TEXT1024_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT1024, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text1024.csv \
  --model_id TEXT1024_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT1024, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text1024.csv \
  --model_id TEXT1024_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

# ==================================================== BABA, TEXT1024, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_text1024.csv \
  --model_id TEXT1024_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 1030 --dec_in 1030 --c_out 1 \
  --des TEXT1024_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1024 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_text_BABA.txt --no-plot_data_splits

cd "$REPO_ROOT"

echo
echo "=== DONE ==="
echo "Text-baseline results:  TimeXer/result_text_{T,INTC,AMD,CVX,BABA}.txt"
echo "Graph results to compare against:  TimeXer/result_{T,INTC,AMD,CVX,BABA}.txt"
echo
echo "Compare WITHIN ticker and WITHIN seed. Never pool MSE across tickers -- T and"
echo "INTC differ in price scale. For each seed the three arms share one price-only"
echo "parent, so PRICE / TEXT64 / TEXT1024 / GRAPH are four numbers on one baseline."
echo "Report wins and a paired statistic over the five seeds, not a mean alone."
