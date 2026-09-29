#!/bin/bash
#SBATCH --job-name=6_matched_fnspid
#SBATCH --output=logs/6_matched_fnspid_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 6 - PRICE and GRAPH on the three-day task of the FNSPID study
#
# Repeats the PRICE and GRAPH arms of step 4 under the FNSPID study's task
# definition: a 50-day input window and the raw close of the next three trading
# days, with the representation of the last predicted day on the final observed
# row (join lead 3).
#   seq_len 10 -> 50, label_len 5 -> 25, pred_len 1 -> 3, target adj_close -> close
# Everything else is as in step 4.
#
# Output  TimeXer/dataset/stock/<TK>_graph_h3.csv and <TK>_price_only_h3.csv,
#         TimeXer/checkpoints_matched_<TK>/, TimeXer/result_matched_<TK>.txt
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/6_matched_fnspid.sh
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
if not last["train"] < last["val"] < last["test"]:
    raise SystemExit("ERROR: embedding splits are not chronologically ordered.")
print(last["train"], last["val"])
EOF
)
echo "=== SPLIT INHERITED FROM STAGE 3: train ends ${TRAIN_END}, val ends ${VAL_END} ==="

echo
echo "=== JOINING WITH A THREE-DAY NEWS LEAD ==="
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 3 \
  --suffix _graph_h3

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/T_graph_h3.csv \
  --output_csv TimeXer/dataset/stock/T_price_only_h3.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/INTC_graph_h3.csv \
  --output_csv TimeXer/dataset/stock/INTC_price_only_h3.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/AMD_graph_h3.csv \
  --output_csv TimeXer/dataset/stock/AMD_price_only_h3.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/CVX_graph_h3.csv \
  --output_csv TimeXer/dataset/stock/CVX_price_only_h3.csv

python TimeXer/make_price_only_csv.py \
  --input_csv TimeXer/dataset/stock/BABA_graph_h3.csv \
  --output_csv TimeXer/dataset/stock/BABA_price_only_h3.csv

echo
echo "=== PARITY: the matched files must cover the same dates as stage 4's ==="
python - <<'EOF'
import csv, sys
def dates(p):
    with open(p, newline="") as h:
        return [r["date"] for r in csv.DictReader(h)]
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    a = dates(f"TimeXer/dataset/stock/{tk}_graph.csv")
    for suf in ("_graph_h3", "_price_only_h3"):
        b = dates(f"TimeXer/dataset/stock/{tk}{suf}.csv")
        if a == b:
            print(f"  OK   {tk}{suf}.csv  {len(b)} rows, identical dates to {tk}_graph.csv")
        else:
            bad.append(f"  FAIL {tk}{suf}: {len(b)} vs {len(a)} rows")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit("ERROR: matched files cover different samples.")
EOF

cd "$REPO_ROOT/TimeXer"

# ############################################################################
# T  -  matched task: 50 days in, 3 days ahead, raw close
# ############################################################################

# ------------------------------------------- T, paired seed 11, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only_h3.csv \
  --model_id PRICE_M_T_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_h3.csv \
  --model_id GRAPH_M_T_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

# ------------------------------------------- T, paired seed 22, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only_h3.csv \
  --model_id PRICE_M_T_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_h3.csv \
  --model_id GRAPH_M_T_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

# ------------------------------------------- T, paired seed 33, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only_h3.csv \
  --model_id PRICE_M_T_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_h3.csv \
  --model_id GRAPH_M_T_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

# ------------------------------------------- T, paired seed 44, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only_h3.csv \
  --model_id PRICE_M_T_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_h3.csv \
  --model_id GRAPH_M_T_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

# ------------------------------------------- T, paired seed 55, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_price_only_h3.csv \
  --model_id PRICE_M_T_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_h3.csv \
  --model_id GRAPH_M_T_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_T/ \
  --result_file result_matched_T.txt --no-plot_data_splits

# ############################################################################
# INTC  -  matched task: 50 days in, 3 days ahead, raw close
# ############################################################################

# ------------------------------------------- INTC, paired seed 11, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only_h3.csv \
  --model_id PRICE_M_INTC_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_h3.csv \
  --model_id GRAPH_M_INTC_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

# ------------------------------------------- INTC, paired seed 22, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only_h3.csv \
  --model_id PRICE_M_INTC_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_h3.csv \
  --model_id GRAPH_M_INTC_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

# ------------------------------------------- INTC, paired seed 33, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only_h3.csv \
  --model_id PRICE_M_INTC_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_h3.csv \
  --model_id GRAPH_M_INTC_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

# ------------------------------------------- INTC, paired seed 44, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only_h3.csv \
  --model_id PRICE_M_INTC_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_h3.csv \
  --model_id GRAPH_M_INTC_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

# ------------------------------------------- INTC, paired seed 55, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_price_only_h3.csv \
  --model_id PRICE_M_INTC_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_h3.csv \
  --model_id GRAPH_M_INTC_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_INTC/ \
  --result_file result_matched_INTC.txt --no-plot_data_splits

# ############################################################################
# AMD  -  matched task: 50 days in, 3 days ahead, raw close
# ############################################################################

# ------------------------------------------- AMD, paired seed 11, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only_h3.csv \
  --model_id PRICE_M_AMD_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_h3.csv \
  --model_id GRAPH_M_AMD_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

# ------------------------------------------- AMD, paired seed 22, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only_h3.csv \
  --model_id PRICE_M_AMD_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_h3.csv \
  --model_id GRAPH_M_AMD_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

# ------------------------------------------- AMD, paired seed 33, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only_h3.csv \
  --model_id PRICE_M_AMD_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_h3.csv \
  --model_id GRAPH_M_AMD_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

# ------------------------------------------- AMD, paired seed 44, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only_h3.csv \
  --model_id PRICE_M_AMD_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_h3.csv \
  --model_id GRAPH_M_AMD_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

# ------------------------------------------- AMD, paired seed 55, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_price_only_h3.csv \
  --model_id PRICE_M_AMD_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_h3.csv \
  --model_id GRAPH_M_AMD_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_AMD/ \
  --result_file result_matched_AMD.txt --no-plot_data_splits

# ############################################################################
# CVX  -  matched task: 50 days in, 3 days ahead, raw close
# ############################################################################

# ------------------------------------------- CVX, paired seed 11, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only_h3.csv \
  --model_id PRICE_M_CVX_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_h3.csv \
  --model_id GRAPH_M_CVX_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

# ------------------------------------------- CVX, paired seed 22, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only_h3.csv \
  --model_id PRICE_M_CVX_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_h3.csv \
  --model_id GRAPH_M_CVX_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

# ------------------------------------------- CVX, paired seed 33, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only_h3.csv \
  --model_id PRICE_M_CVX_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_h3.csv \
  --model_id GRAPH_M_CVX_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

# ------------------------------------------- CVX, paired seed 44, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only_h3.csv \
  --model_id PRICE_M_CVX_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_h3.csv \
  --model_id GRAPH_M_CVX_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

# ------------------------------------------- CVX, paired seed 55, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_price_only_h3.csv \
  --model_id PRICE_M_CVX_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_h3.csv \
  --model_id GRAPH_M_CVX_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_CVX/ \
  --result_file result_matched_CVX.txt --no-plot_data_splits

# ############################################################################
# BABA  -  matched task: 50 days in, 3 days ahead, raw close
# ############################################################################

# ------------------------------------------- BABA, paired seed 11, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only_h3.csv \
  --model_id PRICE_M_BABA_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_h3.csv \
  --model_id GRAPH_M_BABA_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

# ------------------------------------------- BABA, paired seed 22, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only_h3.csv \
  --model_id PRICE_M_BABA_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_h3.csv \
  --model_id GRAPH_M_BABA_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

# ------------------------------------------- BABA, paired seed 33, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only_h3.csv \
  --model_id PRICE_M_BABA_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_h3.csv \
  --model_id GRAPH_M_BABA_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

# ------------------------------------------- BABA, paired seed 44, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only_h3.csv \
  --model_id PRICE_M_BABA_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_h3.csv \
  --model_id GRAPH_M_BABA_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

# ------------------------------------------- BABA, paired seed 55, price-only
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_price_only_h3.csv \
  --model_id PRICE_M_BABA_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --des PRICE_M_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-4 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_h3.csv \
  --model_id GRAPH_M_BABA_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_M_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_BABA/ \
  --result_file result_matched_BABA.txt --no-plot_data_splits

cd "$REPO_ROOT"
echo
echo "=== DONE: 50 models under the matched task ==="
for t in T INTC AMD CVX BABA; do
  echo "  $t: $(grep -c '^mse:' TimeXer/result_matched_$t.txt || true) entries (expect 10)"
done
