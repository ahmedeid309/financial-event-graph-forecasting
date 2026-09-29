#!/bin/bash
#SBATCH --job-name=7_sentiment_arm
#SBATCH --output=logs/7_sentiment_arm_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1

# =============================================================================
# Step 7 - SENTIMENT: the FNSPID sentiment score as the news input
#
# Uses the dataset authors' released ChatGPT sentiment scores (Sentiment_gpt in
# their per-company files, fetched by scripts/0_setup_external.sh); their scoring
# code is not public. gnn/build_sentiment_feature.py turns them into one value per
# company and day (decay 0.03, neutral value 3, standardised on training rows).
# The value enters through the fusion pathway of GRAPH (adapter width 64, frozen
# price pathway, zero-initialised gate, target-day lead) from the PRICE
# checkpoints of step 4.
#
# Input   fnspid_sentiment/fnspid_<TK>.csv, outputs of steps 3 and 4
# Output  gnn_outputs_sentiment/daily_ticker_sentiment.csv,
#         TimeXer/dataset/stock/<TK>_sent.csv, TimeXer/checkpoints_sent_<TK>/,
#         TimeXer/result_sent_<TK>.txt
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/7_train_sentiment_arm.sh
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
echo "=== BUILDING THE DAILY SENTIMENT FEATURE FROM THEIR RELEASED SCORES ==="
python gnn/build_sentiment_feature.py \
  --source_dir fnspid_sentiment \
  --stock_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --decay_lambda 0.03 --neutral 3.0 \
  --column Sentiment_gpt \
  --output_csv gnn_outputs_sentiment/daily_ticker_sentiment.csv

echo
echo "=== JOINING, SAME ONE-ROW TARGET-DAY LEAD AS STAGE 4 ==="
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_sentiment/daily_ticker_sentiment.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 1 \
  --suffix _sent

echo
echo "=== PARITY: every sentiment file must cover the graph file's exact dates ==="
python - <<'EOF'
import csv, sys
def dates(p):
    with open(p, newline="") as h:
        return [r["date"] for r in csv.DictReader(h)]
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    a = dates(f"TimeXer/dataset/stock/{tk}_graph.csv")
    b = dates(f"TimeXer/dataset/stock/{tk}_sent.csv")
    if a == b:
        print(f"  OK   {tk}_sent.csv  {len(b)} rows, identical dates to {tk}_graph.csv")
    else:
        bad.append(f"  FAIL {tk}_sent: {len(b)} vs {len(a)} rows")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit("ERROR: the sentiment arm covers different samples; its MSE is not comparable.")
EOF

echo
echo "=== PRICE-ONLY PARENTS FROM STAGE 4 (loaded, never retrained) ==="
cd "$REPO_ROOT/TimeXer"

test -s ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth || { echo "ERROR: missing price parent T seed 11. Run stage 4 first."; exit 1; }
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth || { echo "ERROR: missing price parent T seed 22. Run stage 4 first."; exit 1; }
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth || { echo "ERROR: missing price parent T seed 33. Run stage 4 first."; exit 1; }
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth || { echo "ERROR: missing price parent T seed 44. Run stage 4 first."; exit 1; }
test -s ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth || { echo "ERROR: missing price parent T seed 55. Run stage 4 first."; exit 1; }
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth || { echo "ERROR: missing price parent INTC seed 11. Run stage 4 first."; exit 1; }
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth || { echo "ERROR: missing price parent INTC seed 22. Run stage 4 first."; exit 1; }
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth || { echo "ERROR: missing price parent INTC seed 33. Run stage 4 first."; exit 1; }
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth || { echo "ERROR: missing price parent INTC seed 44. Run stage 4 first."; exit 1; }
test -s ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth || { echo "ERROR: missing price parent INTC seed 55. Run stage 4 first."; exit 1; }
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth || { echo "ERROR: missing price parent AMD seed 11. Run stage 4 first."; exit 1; }
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth || { echo "ERROR: missing price parent AMD seed 22. Run stage 4 first."; exit 1; }
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth || { echo "ERROR: missing price parent AMD seed 33. Run stage 4 first."; exit 1; }
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth || { echo "ERROR: missing price parent AMD seed 44. Run stage 4 first."; exit 1; }
test -s ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth || { echo "ERROR: missing price parent AMD seed 55. Run stage 4 first."; exit 1; }
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth || { echo "ERROR: missing price parent CVX seed 11. Run stage 4 first."; exit 1; }
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth || { echo "ERROR: missing price parent CVX seed 22. Run stage 4 first."; exit 1; }
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth || { echo "ERROR: missing price parent CVX seed 33. Run stage 4 first."; exit 1; }
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth || { echo "ERROR: missing price parent CVX seed 44. Run stage 4 first."; exit 1; }
test -s ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth || { echo "ERROR: missing price parent CVX seed 55. Run stage 4 first."; exit 1; }
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth || { echo "ERROR: missing price parent BABA seed 11. Run stage 4 first."; exit 1; }
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth || { echo "ERROR: missing price parent BABA seed 22. Run stage 4 first."; exit 1; }
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth || { echo "ERROR: missing price parent BABA seed 33. Run stage 4 first."; exit 1; }
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth || { echo "ERROR: missing price parent BABA seed 44. Run stage 4 first."; exit 1; }
test -s ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth || { echo "ERROR: missing price parent BABA seed 55. Run stage 4 first."; exit 1; }
echo "  all 25 parents present"

# ############################################################################
# T  -  sentiment arm, one dimension, same parent as the graph arm
# ############################################################################

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent.csv \
  --model_id SENT_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_T/ \
  --result_file result_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent.csv \
  --model_id SENT_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_T/ \
  --result_file result_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent.csv \
  --model_id SENT_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_T/ \
  --result_file result_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent.csv \
  --model_id SENT_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_T/ \
  --result_file result_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent.csv \
  --model_id SENT_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_T/ \
  --result_file result_sent_T.txt --no-plot_data_splits

# ############################################################################
# INTC  -  sentiment arm, one dimension, same parent as the graph arm
# ############################################################################

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent.csv \
  --model_id SENT_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_INTC/ \
  --result_file result_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent.csv \
  --model_id SENT_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_INTC/ \
  --result_file result_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent.csv \
  --model_id SENT_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_INTC/ \
  --result_file result_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent.csv \
  --model_id SENT_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_INTC/ \
  --result_file result_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent.csv \
  --model_id SENT_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_INTC/ \
  --result_file result_sent_INTC.txt --no-plot_data_splits

# ############################################################################
# AMD  -  sentiment arm, one dimension, same parent as the graph arm
# ############################################################################

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent.csv \
  --model_id SENT_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_AMD/ \
  --result_file result_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent.csv \
  --model_id SENT_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_AMD/ \
  --result_file result_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent.csv \
  --model_id SENT_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_AMD/ \
  --result_file result_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent.csv \
  --model_id SENT_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_AMD/ \
  --result_file result_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent.csv \
  --model_id SENT_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_AMD/ \
  --result_file result_sent_AMD.txt --no-plot_data_splits

# ############################################################################
# CVX  -  sentiment arm, one dimension, same parent as the graph arm
# ############################################################################

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent.csv \
  --model_id SENT_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_CVX/ \
  --result_file result_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent.csv \
  --model_id SENT_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_CVX/ \
  --result_file result_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent.csv \
  --model_id SENT_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_CVX/ \
  --result_file result_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent.csv \
  --model_id SENT_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_CVX/ \
  --result_file result_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent.csv \
  --model_id SENT_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_CVX/ \
  --result_file result_sent_CVX.txt --no-plot_data_splits

# ############################################################################
# BABA  -  sentiment arm, one dimension, same parent as the graph arm
# ############################################################################

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent.csv \
  --model_id SENT_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_BABA/ \
  --result_file result_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent.csv \
  --model_id SENT_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_BABA/ \
  --result_file result_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent.csv \
  --model_id SENT_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_BABA/ \
  --result_file result_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent.csv \
  --model_id SENT_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_BABA/ \
  --result_file result_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent.csv \
  --model_id SENT_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_sent_BABA/ \
  --result_file result_sent_BABA.txt --no-plot_data_splits

cd "$REPO_ROOT"
echo
echo "=== DONE: 25 sentiment-arm models ==="
for t in T INTC AMD CVX BABA; do
  echo "  $t: $(grep -c '^mse:' TimeXer/result_sent_$t.txt || true) entries (expect 5)"
done
