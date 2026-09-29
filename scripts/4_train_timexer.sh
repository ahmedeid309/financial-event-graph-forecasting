#!/bin/bash
#SBATCH --job-name=4_train_timexer
#SBATCH --output=logs/4_train_timexer_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 4 - Main experiment: PRICE and GRAPH on the one-step task
#
# For each company (T, INTC, AMD, CVX, BABA) and seed (11, 22, 33, 44, 55):
#   PRICE  TimeXer on ten days of prices, forecasting the next trading day's
#          adjusted close.
#   GRAPH  the same model, initialised from the PRICE checkpoint of the same seed
#          with the price pathway frozen, plus the 64-dimensional graph
#          representation through its own cross-attention, a 64-wide adapter, and
#          a tanh gate that starts at zero. The initial price-only state is
#          validated first and stays selected unless validation MSE improves.
# The representation of the target day sits on the last observed row
# (--graph_target_lead 1), so forecasts are conditional on target-day news.
#
# Prices come from dataset/full_history_fixed/ (gnn/fix_price_splice.py), which
# corrects the adjustment break of 6 July 2020. Rows from that date on are
# unchanged, so the validation and test periods are untouched.
#
# Input   gnn_outputs_shared/daily_ticker_embeddings.csv, dataset/full_history_fixed/
# Output  TimeXer/dataset/stock/<TK>_graph.csv and <TK>_price_only.csv,
#         TimeXer/checkpoints_<TK>/, TimeXer/results/, TimeXer/result_<TK>.txt
#
# Results are appended to result_<TK>.txt; remove old result files before a rerun
# (run_chain.sh does this).
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/4_train_timexer.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

echo "=== STAGE 4 INPUT: PRODUCED BY STAGE 3 ==="
test -s gnn_outputs_shared/daily_ticker_embeddings.csv || {
  echo "ERROR: stage 3 embedding CSV is missing or empty"
  exit 1
}
test -s gnn_outputs_shared/daily_ticker_embeddings.metadata.json || {
  echo "ERROR: stage 3 embedding metadata is missing or empty"
  exit 1
}
ls -lh gnn_outputs_shared/daily_ticker_embeddings.csv
cat gnn_outputs_shared/daily_ticker_embeddings.metadata.json

# --------------------------------------------------------------- ONE SPLIT
# Stage 3 already decided which days are train/val/test and stamped that on
# every row of the embedding CSV. Read the two boundary dates back out instead
# of letting TimeXer re-derive its own split from row fractions.
#
# WHY THIS MATTERS: the GNN and TimeXer run over the identical set of days. If
# each picks its own boundary, the GNN ends up validated on days that TimeXer
# is tested on, so checkpoint selection sees the downstream test period. Nothing
# errors and the MSE looks fine -- the contamination is invisible. Sharing one
# split makes that impossible rather than merely unlikely.
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
echo "Both the GNN and TimeXer cut the calendar on these two dates."

# One joined file per ticker. join_embeddings_to_stock.py loops over --tickers
# and writes <TICKER>_graph.csv for each.
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --suffix _graph

# Each price-only twin is derived from that ticker's OWN graph file, so the two
# variants for a ticker cover exactly the same dates. Without this the price-only
# arm would silently be a different sample: tasks are news-conditioned, so the
# graph file has no row on days without news.
python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/T_graph.csv \
  --output_csv TimeXer/dataset/stock/T_price_only.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/INTC_graph.csv \
  --output_csv TimeXer/dataset/stock/INTC_price_only.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/AMD_graph.csv \
  --output_csv TimeXer/dataset/stock/AMD_price_only.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/CVX_graph.csv \
  --output_csv TimeXer/dataset/stock/CVX_price_only.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/BABA_graph.csv \
  --output_csv TimeXer/dataset/stock/BABA_price_only.csv

echo
echo "=== row counts: each ticker's two files must match each other ==="
wc -l TimeXer/dataset/stock/T_graph.csv    TimeXer/dataset/stock/T_price_only.csv
wc -l TimeXer/dataset/stock/INTC_graph.csv TimeXer/dataset/stock/INTC_price_only.csv
wc -l TimeXer/dataset/stock/AMD_graph.csv TimeXer/dataset/stock/AMD_price_only.csv
wc -l TimeXer/dataset/stock/CVX_graph.csv TimeXer/dataset/stock/CVX_price_only.csv
wc -l TimeXer/dataset/stock/BABA_graph.csv TimeXer/dataset/stock/BABA_price_only.csv

cd "$REPO_ROOT/TimeXer"

# ############################################################################
# TICKER T
# ############################################################################

# ==================================================== T, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only.csv \
  --model_id PRICE_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph.csv \
  --model_id GRAPH_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

# ==================================================== T, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only.csv \
  --model_id PRICE_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph.csv \
  --model_id GRAPH_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

# ==================================================== T, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only.csv \
  --model_id PRICE_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph.csv \
  --model_id GRAPH_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

# ==================================================== T, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only.csv \
  --model_id PRICE_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph.csv \
  --model_id GRAPH_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

# ==================================================== T, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only.csv \
  --model_id PRICE_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph.csv \
  --model_id GRAPH_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_T/ \
  --result_file result_T.txt --no-plot_data_splits

# ############################################################################
# TICKER INTC
# ############################################################################

# ==================================================== INTC, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only.csv \
  --model_id PRICE_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph.csv \
  --model_id GRAPH_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

# ==================================================== INTC, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only.csv \
  --model_id PRICE_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph.csv \
  --model_id GRAPH_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

# ==================================================== INTC, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only.csv \
  --model_id PRICE_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph.csv \
  --model_id GRAPH_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

# ==================================================== INTC, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only.csv \
  --model_id PRICE_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph.csv \
  --model_id GRAPH_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

# ==================================================== INTC, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only.csv \
  --model_id PRICE_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph.csv \
  --model_id GRAPH_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_INTC/ \
  --result_file result_INTC.txt --no-plot_data_splits

# ############################################################################
# TICKER AMD
# ############################################################################

# ==================================================== AMD, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only.csv \
  --model_id PRICE_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph.csv \
  --model_id GRAPH_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

# ==================================================== AMD, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only.csv \
  --model_id PRICE_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph.csv \
  --model_id GRAPH_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

# ==================================================== AMD, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only.csv \
  --model_id PRICE_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph.csv \
  --model_id GRAPH_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

# ==================================================== AMD, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only.csv \
  --model_id PRICE_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph.csv \
  --model_id GRAPH_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

# ==================================================== AMD, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only.csv \
  --model_id PRICE_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph.csv \
  --model_id GRAPH_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_AMD/ \
  --result_file result_AMD.txt --no-plot_data_splits

# ############################################################################
# TICKER CVX
# ############################################################################

# ==================================================== CVX, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only.csv \
  --model_id PRICE_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph.csv \
  --model_id GRAPH_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

# ==================================================== CVX, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only.csv \
  --model_id PRICE_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph.csv \
  --model_id GRAPH_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

# ==================================================== CVX, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only.csv \
  --model_id PRICE_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph.csv \
  --model_id GRAPH_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

# ==================================================== CVX, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only.csv \
  --model_id PRICE_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph.csv \
  --model_id GRAPH_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

# ==================================================== CVX, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only.csv \
  --model_id PRICE_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph.csv \
  --model_id GRAPH_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_CVX/ \
  --result_file result_CVX.txt --no-plot_data_splits

# ############################################################################
# TICKER BABA
# ############################################################################

# ==================================================== BABA, paired seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only.csv \
  --model_id PRICE_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph.csv \
  --model_id GRAPH_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

# ==================================================== BABA, paired seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only.csv \
  --model_id PRICE_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph.csv \
  --model_id GRAPH_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

# ==================================================== BABA, paired seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only.csv \
  --model_id PRICE_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph.csv \
  --model_id GRAPH_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

# ==================================================== BABA, paired seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only.csv \
  --model_id PRICE_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph.csv \
  --model_id GRAPH_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

# ==================================================== BABA, paired seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only.csv \
  --model_id PRICE_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph.csv \
  --model_id GRAPH_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_BABA/ \
  --result_file result_BABA.txt --no-plot_data_splits

echo
echo "=== stage 4 done ==="
echo "T    metrics: TimeXer/result_T.txt"
echo "INTC metrics: TimeXer/result_INTC.txt"
echo "AMD  metrics: TimeXer/result_AMD.txt"
echo "CVX  metrics: TimeXer/result_CVX.txt"
echo "BABA metrics: TimeXer/result_BABA.txt"
echo "Pair PRICE_<TK>_sN against GRAPH_<TK>_sN within a ticker. Never average"
echo "across seeds, and never pool tickers -- they are different series."
