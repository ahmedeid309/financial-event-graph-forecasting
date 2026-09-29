#!/bin/bash
#SBATCH --job-name=4c_encoder_variance
#SBATCH --output=logs/4c_encoder_variance_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 4c - Sensitivity to the trained graph encoder
#
# Step 3 trains three encoders and exports one. This step exports the daily
# representations of all three (seeds 2021, 2022, 2023), joins each to the
# prices, and trains the GRAPH arm for five companies and five seeds per encoder
# (75 runs) from the PRICE checkpoints of step 4. No encoder and no price-only
# model is retrained.
#
# Built-in check: encoder 2022 is the one step 3 exported, so its 25 runs must
# reproduce the GRAPH results of step 4 exactly. analysis/analyze_final.py
# verifies this before it reports anything.
#
# Output  gnn_outputs_shared/daily_ticker_embeddings_e<seed>.csv,
#         TimeXer/result_encvar_<TK>.txt
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/4c_encoder_variance.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

read -r TRAIN_END VAL_END < <(python - <<'EOF'
import csv
last = {}
with open("gnn_outputs_shared/daily_ticker_embeddings.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["date"] > last.get(row["split"], ""):
            last[row["split"]] = row["date"]
missing = [n for n in ("train", "val", "test") if n not in last]
if missing:
    raise SystemExit(f"ERROR: embedding CSV has no rows for split(s): {', '.join(missing)}")
if not last["train"] < last["val"] < last["test"]:
    raise SystemExit("ERROR: embedding splits are not chronologically ordered.")
print(last["train"], last["val"])
EOF
)

echo
echo "=== ONE SPLIT, INHERITED FROM STAGE 3 ==="
echo "train ends: ${TRAIN_END}   val ends: ${VAL_END}"

# ############################################################################
# ENCODER SEED 2021
# ############################################################################

echo
echo "=== ENCODER 2021: exporting its daily embeddings ==="
python gnn/export_embeddings.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 \
  --checkpoint gnn_outputs_shared/checkpoints/event_gnn_gnn_seed2021.pt \
  --output_csv gnn_outputs_shared/daily_ticker_embeddings_e2021.csv \
  --embedding_prefix gnn \
  --normalization zscore \
  --device cuda

python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings_e2021.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --suffix _graph_e2021

echo
echo "=== ENCODER 2021 PARITY: dates must equal the production graph files ==="
python - <<'EOF'
import csv, sys
def dates(p):
    with open(p, newline="") as h:
        return [r["date"] for r in csv.DictReader(h)]
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    a = dates(f"TimeXer/dataset/stock/{tk}_graph.csv")
    b = dates(f"TimeXer/dataset/stock/{tk}_graph_e2021.csv")
    if a == b:
        print(f"  OK   {tk}_graph_e2021.csv  {len(b)} rows, identical dates to {tk}_graph.csv")
    else:
        bad.append(f"  FAIL {tk}: {len(b)} rows vs {len(a)}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit("ERROR: encoder 2021 covers different samples; its MSEs are not comparable.")
EOF

cd "$REPO_ROOT/TimeXer"

# ==================================================== encoder 2021, T
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2021.csv \
  --model_id GRAPH_E2021_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2021.csv \
  --model_id GRAPH_E2021_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2021.csv \
  --model_id GRAPH_E2021_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2021.csv \
  --model_id GRAPH_E2021_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2021.csv \
  --model_id GRAPH_E2021_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

# ==================================================== encoder 2021, INTC
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2021.csv \
  --model_id GRAPH_E2021_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2021.csv \
  --model_id GRAPH_E2021_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2021.csv \
  --model_id GRAPH_E2021_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2021.csv \
  --model_id GRAPH_E2021_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2021.csv \
  --model_id GRAPH_E2021_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

# ==================================================== encoder 2021, AMD
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2021.csv \
  --model_id GRAPH_E2021_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2021.csv \
  --model_id GRAPH_E2021_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2021.csv \
  --model_id GRAPH_E2021_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2021.csv \
  --model_id GRAPH_E2021_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2021.csv \
  --model_id GRAPH_E2021_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

# ==================================================== encoder 2021, CVX
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2021.csv \
  --model_id GRAPH_E2021_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2021.csv \
  --model_id GRAPH_E2021_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2021.csv \
  --model_id GRAPH_E2021_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2021.csv \
  --model_id GRAPH_E2021_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2021.csv \
  --model_id GRAPH_E2021_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

# ==================================================== encoder 2021, BABA
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2021.csv \
  --model_id GRAPH_E2021_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2021.csv \
  --model_id GRAPH_E2021_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2021.csv \
  --model_id GRAPH_E2021_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2021.csv \
  --model_id GRAPH_E2021_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2021.csv \
  --model_id GRAPH_E2021_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2021_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

cd "$REPO_ROOT"

# ############################################################################
# ENCODER SEED 2022
# ############################################################################

echo
echo "=== ENCODER 2022: exporting its daily embeddings ==="
python gnn/export_embeddings.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 \
  --checkpoint gnn_outputs_shared/checkpoints/event_gnn_gnn_seed2022.pt \
  --output_csv gnn_outputs_shared/daily_ticker_embeddings_e2022.csv \
  --embedding_prefix gnn \
  --normalization zscore \
  --device cuda

python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings_e2022.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --suffix _graph_e2022

echo
echo "=== ENCODER 2022 PARITY: dates must equal the production graph files ==="
python - <<'EOF'
import csv, sys
def dates(p):
    with open(p, newline="") as h:
        return [r["date"] for r in csv.DictReader(h)]
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    a = dates(f"TimeXer/dataset/stock/{tk}_graph.csv")
    b = dates(f"TimeXer/dataset/stock/{tk}_graph_e2022.csv")
    if a == b:
        print(f"  OK   {tk}_graph_e2022.csv  {len(b)} rows, identical dates to {tk}_graph.csv")
    else:
        bad.append(f"  FAIL {tk}: {len(b)} rows vs {len(a)}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit("ERROR: encoder 2022 covers different samples; its MSEs are not comparable.")
EOF

cd "$REPO_ROOT/TimeXer"

# ==================================================== encoder 2022, T
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2022.csv \
  --model_id GRAPH_E2022_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2022.csv \
  --model_id GRAPH_E2022_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2022.csv \
  --model_id GRAPH_E2022_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2022.csv \
  --model_id GRAPH_E2022_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2022.csv \
  --model_id GRAPH_E2022_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

# ==================================================== encoder 2022, INTC
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2022.csv \
  --model_id GRAPH_E2022_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2022.csv \
  --model_id GRAPH_E2022_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2022.csv \
  --model_id GRAPH_E2022_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2022.csv \
  --model_id GRAPH_E2022_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2022.csv \
  --model_id GRAPH_E2022_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

# ==================================================== encoder 2022, AMD
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2022.csv \
  --model_id GRAPH_E2022_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2022.csv \
  --model_id GRAPH_E2022_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2022.csv \
  --model_id GRAPH_E2022_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2022.csv \
  --model_id GRAPH_E2022_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2022.csv \
  --model_id GRAPH_E2022_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

# ==================================================== encoder 2022, CVX
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2022.csv \
  --model_id GRAPH_E2022_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2022.csv \
  --model_id GRAPH_E2022_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2022.csv \
  --model_id GRAPH_E2022_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2022.csv \
  --model_id GRAPH_E2022_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2022.csv \
  --model_id GRAPH_E2022_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

# ==================================================== encoder 2022, BABA
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2022.csv \
  --model_id GRAPH_E2022_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2022.csv \
  --model_id GRAPH_E2022_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2022.csv \
  --model_id GRAPH_E2022_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2022.csv \
  --model_id GRAPH_E2022_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2022.csv \
  --model_id GRAPH_E2022_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2022_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

cd "$REPO_ROOT"

# ############################################################################
# ENCODER SEED 2023
# ############################################################################

echo
echo "=== ENCODER 2023: exporting its daily embeddings ==="
python gnn/export_embeddings.py \
  --graph_dir ekg_final \
  --forecast_tasks gnn_data/forecast_task.csv \
  --horizon_trading_days 0 \
  --event_ticker_scope target_or_anchor \
  --window_days 1 --decay_lambda 0.10 --num_hops 3 \
  --embedding_dim 64 \
  --checkpoint gnn_outputs_shared/checkpoints/event_gnn_gnn_seed2023.pt \
  --output_csv gnn_outputs_shared/daily_ticker_embeddings_e2023.csv \
  --embedding_prefix gnn \
  --normalization zscore \
  --device cuda

python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings_e2023.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --suffix _graph_e2023

echo
echo "=== ENCODER 2023 PARITY: dates must equal the production graph files ==="
python - <<'EOF'
import csv, sys
def dates(p):
    with open(p, newline="") as h:
        return [r["date"] for r in csv.DictReader(h)]
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    a = dates(f"TimeXer/dataset/stock/{tk}_graph.csv")
    b = dates(f"TimeXer/dataset/stock/{tk}_graph_e2023.csv")
    if a == b:
        print(f"  OK   {tk}_graph_e2023.csv  {len(b)} rows, identical dates to {tk}_graph.csv")
    else:
        bad.append(f"  FAIL {tk}: {len(b)} rows vs {len(a)}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit("ERROR: encoder 2023 covers different samples; its MSEs are not comparable.")
EOF

cd "$REPO_ROOT/TimeXer"

# ==================================================== encoder 2023, T
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2023.csv \
  --model_id GRAPH_E2023_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2023.csv \
  --model_id GRAPH_E2023_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2023.csv \
  --model_id GRAPH_E2023_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2023.csv \
  --model_id GRAPH_E2023_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_e2023.csv \
  --model_id GRAPH_E2023_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_T/ \
  --result_file result_encvar_T.txt --no-plot_data_splits

# ==================================================== encoder 2023, INTC
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2023.csv \
  --model_id GRAPH_E2023_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2023.csv \
  --model_id GRAPH_E2023_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2023.csv \
  --model_id GRAPH_E2023_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2023.csv \
  --model_id GRAPH_E2023_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_e2023.csv \
  --model_id GRAPH_E2023_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_INTC/ \
  --result_file result_encvar_INTC.txt --no-plot_data_splits

# ==================================================== encoder 2023, AMD
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2023.csv \
  --model_id GRAPH_E2023_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2023.csv \
  --model_id GRAPH_E2023_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2023.csv \
  --model_id GRAPH_E2023_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2023.csv \
  --model_id GRAPH_E2023_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_e2023.csv \
  --model_id GRAPH_E2023_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_AMD/ \
  --result_file result_encvar_AMD.txt --no-plot_data_splits

# ==================================================== encoder 2023, CVX
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2023.csv \
  --model_id GRAPH_E2023_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2023.csv \
  --model_id GRAPH_E2023_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2023.csv \
  --model_id GRAPH_E2023_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2023.csv \
  --model_id GRAPH_E2023_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_e2023.csv \
  --model_id GRAPH_E2023_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_CVX/ \
  --result_file result_encvar_CVX.txt --no-plot_data_splits

# ==================================================== encoder 2023, BABA
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2023.csv \
  --model_id GRAPH_E2023_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2023.csv \
  --model_id GRAPH_E2023_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2023.csv \
  --model_id GRAPH_E2023_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2023.csv \
  --model_id GRAPH_E2023_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_e2023.csv \
  --model_id GRAPH_E2023_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_E2023_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_encvar_BABA/ \
  --result_file result_encvar_BABA.txt --no-plot_data_splits

cd "$REPO_ROOT"

echo
echo "=== DONE: 75 graph models, three encoder draws ==="
echo "Encoder 2022 must match stage 4's GRAPH numbers exactly. Check that first."
for t in T INTC AMD CVX BABA; do
  echo "  $t: $(grep -c '^mse:' TimeXer/result_encvar_$t.txt || true) entries (expect 15)"
done
