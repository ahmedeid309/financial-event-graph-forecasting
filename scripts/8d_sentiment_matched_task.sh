#!/bin/bash
#SBATCH --job-name=8d_sentiment_matched
#SBATCH --output=logs/8d_sentiment_matched_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# =============================================================================
# Step 8d - The FNSPID sentiment score on the three-day task, in both models
#
# Completes the 3 x 2 grid of the three-day task:
#                         prices only   + sentiment   + graph
#   FNSPID Transformer    8c            8d            8c
#   TimeXer               6             8d            6
# The sentiment feature is the one of step 7; its value for the last predicted
# day sits on the final observed row (join lead 3), as the graph does in step 6.
# The TimeXer runs load the price-only checkpoints of step 6; the Transformer runs
# are identical to step 8c except for the input file. Both run side by side on one
# GPU with their own logs (logs/8d_timexer_<job>.out, logs/8d_transformer_<job>.out).
#
# Output  TimeXer/checkpoints_matched_sent_<TK>/, TimeXer/result_matched_sent_<TK>.txt,
#         fnspid_transformer/results_forecast/sentiment_s<seed>/<TK>_predictions.csv
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/8d_sentiment_matched_task.sh
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
echo "=== NOTHING FROM A PREVIOUS RUN MAY EXIST (results are appended) ==="
test ! -e TimeXer/result_matched_sent_T.txt || { echo "ERROR: TimeXer/result_matched_sent_T.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_matched_sent_INTC.txt || { echo "ERROR: TimeXer/result_matched_sent_INTC.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_matched_sent_AMD.txt || { echo "ERROR: TimeXer/result_matched_sent_AMD.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_matched_sent_CVX.txt || { echo "ERROR: TimeXer/result_matched_sent_CVX.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_matched_sent_BABA.txt || { echo "ERROR: TimeXer/result_matched_sent_BABA.txt exists; archive it first."; exit 1; }
test ! -e fnspid_transformer/results_forecast/sentiment_s11 || { echo "ERROR: fnspid_transformer/results_forecast/sentiment_s11 exists; archive it first."; exit 1; }
test ! -e fnspid_transformer/results_forecast/sentiment_s22 || { echo "ERROR: fnspid_transformer/results_forecast/sentiment_s22 exists; archive it first."; exit 1; }
test ! -e fnspid_transformer/results_forecast/sentiment_s33 || { echo "ERROR: fnspid_transformer/results_forecast/sentiment_s33 exists; archive it first."; exit 1; }
test ! -e fnspid_transformer/results_forecast/sentiment_s44 || { echo "ERROR: fnspid_transformer/results_forecast/sentiment_s44 exists; archive it first."; exit 1; }
test ! -e fnspid_transformer/results_forecast/sentiment_s55 || { echo "ERROR: fnspid_transformer/results_forecast/sentiment_s55 exists; archive it first."; exit 1; }

echo
echo "=== JOINING THE SENTIMENT SCORE WITH THE THREE-ROW LEAD OF STAGE 6 ==="
test -s gnn_outputs_sentiment/daily_ticker_sentiment.csv || { echo "ERROR: stage 7's sentiment feature is missing."; exit 1; }
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_sentiment/daily_ticker_sentiment.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 3 \
  --suffix _sent_h3

echo
echo "=== PARITY: every _sent_h3 file must cover the stage-6 file's exact dates ==="
python - <<'EOF'
import csv
def dates(p):
    with open(p, newline="") as h:
        return [r["date"] for r in csv.DictReader(h)]
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    a = dates(f"TimeXer/dataset/stock/{tk}_graph_h3.csv")
    b = dates(f"TimeXer/dataset/stock/{tk}_sent_h3.csv")
    if a == b:
        print(f"  OK   {tk}_sent_h3.csv  {len(b)} rows, identical dates to {tk}_graph_h3.csv")
    else:
        bad.append(f"  FAIL {tk}_sent_h3: {len(b)} vs {len(a)} rows")
if bad:
    print("\n".join(bad))
    raise SystemExit("ERROR: the sentiment files cover different samples; their errors would not be comparable.")
EOF

echo
echo "=== STAGE-6 PRICE-ONLY PARENTS (loaded, never retrained) ==="
test -s TimeXer/checkpoints_matched_T/long_term_forecast_PRICE_M_T_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s11_0/checkpoint.pth || { echo "ERROR: missing parent T seed 11. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_T/long_term_forecast_PRICE_M_T_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s22_0/checkpoint.pth || { echo "ERROR: missing parent T seed 22. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_T/long_term_forecast_PRICE_M_T_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s33_0/checkpoint.pth || { echo "ERROR: missing parent T seed 33. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_T/long_term_forecast_PRICE_M_T_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s44_0/checkpoint.pth || { echo "ERROR: missing parent T seed 44. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_T/long_term_forecast_PRICE_M_T_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s55_0/checkpoint.pth || { echo "ERROR: missing parent T seed 55. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s11_0/checkpoint.pth || { echo "ERROR: missing parent INTC seed 11. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s22_0/checkpoint.pth || { echo "ERROR: missing parent INTC seed 22. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s33_0/checkpoint.pth || { echo "ERROR: missing parent INTC seed 33. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s44_0/checkpoint.pth || { echo "ERROR: missing parent INTC seed 44. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s55_0/checkpoint.pth || { echo "ERROR: missing parent INTC seed 55. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s11_0/checkpoint.pth || { echo "ERROR: missing parent AMD seed 11. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s22_0/checkpoint.pth || { echo "ERROR: missing parent AMD seed 22. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s33_0/checkpoint.pth || { echo "ERROR: missing parent AMD seed 33. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s44_0/checkpoint.pth || { echo "ERROR: missing parent AMD seed 44. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s55_0/checkpoint.pth || { echo "ERROR: missing parent AMD seed 55. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s11_0/checkpoint.pth || { echo "ERROR: missing parent CVX seed 11. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s22_0/checkpoint.pth || { echo "ERROR: missing parent CVX seed 22. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s33_0/checkpoint.pth || { echo "ERROR: missing parent CVX seed 33. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s44_0/checkpoint.pth || { echo "ERROR: missing parent CVX seed 44. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s55_0/checkpoint.pth || { echo "ERROR: missing parent CVX seed 55. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s11_0/checkpoint.pth || { echo "ERROR: missing parent BABA seed 11. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s22_0/checkpoint.pth || { echo "ERROR: missing parent BABA seed 22. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s33_0/checkpoint.pth || { echo "ERROR: missing parent BABA seed 33. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s44_0/checkpoint.pth || { echo "ERROR: missing parent BABA seed 44. Run stage 6 first."; exit 1; }
test -s TimeXer/checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s55_0/checkpoint.pth || { echo "ERROR: missing parent BABA seed 55. Run stage 6 first."; exit 1; }
echo "  all 25 parents present"

# ############################################################################
# TIMEXER + SENTIMENT (25 models), in the background
# ############################################################################
(
cd "$REPO_ROOT/TimeXer"

# ==================================================== TimeXer + sentiment, T
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_h3.csv \
  --model_id SENT_M_T_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_T/ \
  --result_file result_matched_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_h3.csv \
  --model_id SENT_M_T_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_T/ \
  --result_file result_matched_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_h3.csv \
  --model_id SENT_M_T_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_T/ \
  --result_file result_matched_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_h3.csv \
  --model_id SENT_M_T_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_T/ \
  --result_file result_matched_sent_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_h3.csv \
  --model_id SENT_M_T_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_T/long_term_forecast_PRICE_M_T_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_T/ \
  --result_file result_matched_sent_T.txt --no-plot_data_splits

# ==================================================== TimeXer + sentiment, INTC
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_h3.csv \
  --model_id SENT_M_INTC_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_INTC/ \
  --result_file result_matched_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_h3.csv \
  --model_id SENT_M_INTC_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_INTC/ \
  --result_file result_matched_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_h3.csv \
  --model_id SENT_M_INTC_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_INTC/ \
  --result_file result_matched_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_h3.csv \
  --model_id SENT_M_INTC_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_INTC/ \
  --result_file result_matched_sent_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_h3.csv \
  --model_id SENT_M_INTC_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_INTC/long_term_forecast_PRICE_M_INTC_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_INTC/ \
  --result_file result_matched_sent_INTC.txt --no-plot_data_splits

# ==================================================== TimeXer + sentiment, AMD
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_h3.csv \
  --model_id SENT_M_AMD_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_AMD/ \
  --result_file result_matched_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_h3.csv \
  --model_id SENT_M_AMD_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_AMD/ \
  --result_file result_matched_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_h3.csv \
  --model_id SENT_M_AMD_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_AMD/ \
  --result_file result_matched_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_h3.csv \
  --model_id SENT_M_AMD_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_AMD/ \
  --result_file result_matched_sent_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_h3.csv \
  --model_id SENT_M_AMD_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_AMD/long_term_forecast_PRICE_M_AMD_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_AMD/ \
  --result_file result_matched_sent_AMD.txt --no-plot_data_splits

# ==================================================== TimeXer + sentiment, CVX
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_h3.csv \
  --model_id SENT_M_CVX_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_CVX/ \
  --result_file result_matched_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_h3.csv \
  --model_id SENT_M_CVX_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_CVX/ \
  --result_file result_matched_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_h3.csv \
  --model_id SENT_M_CVX_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_CVX/ \
  --result_file result_matched_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_h3.csv \
  --model_id SENT_M_CVX_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_CVX/ \
  --result_file result_matched_sent_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_h3.csv \
  --model_id SENT_M_CVX_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_CVX/long_term_forecast_PRICE_M_CVX_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_CVX/ \
  --result_file result_matched_sent_CVX.txt --no-plot_data_splits

# ==================================================== TimeXer + sentiment, BABA
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_h3.csv \
  --model_id SENT_M_BABA_s11 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s11_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_BABA/ \
  --result_file result_matched_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_h3.csv \
  --model_id SENT_M_BABA_s22 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s22_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_BABA/ \
  --result_file result_matched_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_h3.csv \
  --model_id SENT_M_BABA_s33 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s33_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_BABA/ \
  --result_file result_matched_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_h3.csv \
  --model_id SENT_M_BABA_s44 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s44_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_BABA/ \
  --result_file result_matched_sent_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_h3.csv \
  --model_id SENT_M_BABA_s55 --model TimeXer --data custom \
  --features MS --target close --freq d \
  --seq_len 50 --label_len 25 --pred_len 3 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_M_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_matched_BABA/long_term_forecast_PRICE_M_BABA_s55_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_M_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_matched_sent_BABA/ \
  --result_file result_matched_sent_BABA.txt --no-plot_data_splits

) > "logs/8d_timexer_${SLURM_JOB_ID}.out" 2>&1 &
PID_TIMEXER=$!

# ############################################################################
# FNSPID TRANSFORMER + SENTIMENT (5 runs x 5 companies), in the background
# ############################################################################
(
cd "$REPO_ROOT/fnspid_transformer"
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 1 --val_end "$VAL_END" --mode sentiment --seed 11 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 1 --val_end "$VAL_END" --mode sentiment --seed 22 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 1 --val_end "$VAL_END" --mode sentiment --seed 33 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 1 --val_end "$VAL_END" --mode sentiment --seed 44 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 1 --val_end "$VAL_END" --mode sentiment --seed 55 --epochs 100 --out_dir results_forecast
) > "logs/8d_transformer_${SLURM_JOB_ID}.out" 2>&1 &
PID_TRANSFORMER=$!

echo
echo "=== RUNNING: TimeXer pid ${PID_TIMEXER}, Transformer pid ${PID_TRANSFORMER} ==="
set +e
wait $PID_TIMEXER
STATUS_TIMEXER=$?
wait $PID_TRANSFORMER
STATUS_TRANSFORMER=$?
set -e
echo "TimeXer block exited with status ${STATUS_TIMEXER}"
echo "Transformer block exited with status ${STATUS_TRANSFORMER}"
if [ $STATUS_TIMEXER -ne 0 ] || [ $STATUS_TRANSFORMER -ne 0 ]; then
  exit 1
fi
echo "=== 8d done ==="
