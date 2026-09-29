#!/bin/bash
#SBATCH --job-name=9_previous_day_news
#SBATCH --output=logs/9_previous_day_news_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# =============================================================================
# Step 9 - One-step task with previous-day news only
#
# Steps 4, 7 and 8e put the representation of the target day on the last observed
# row (join lead 1). This step puts the representation of the last observed day
# there (join lead 0). Each daily representation covers its own day and the
# calendar day before, so the forecast for day t uses news from days t-2 and t-1
# and none from day t. Encoder, sentiment feature, window, target, split, and test
# days are unchanged.
#   TimeXer      GRAPH_L0 and SENT_L0, 25 runs each, from the PRICE checkpoints
#                of step 4 with the settings of steps 4 and 7
#   Transformer  sentiment and graph, five seeds each, as in step 8e
# The price-only runs use no news and are reused from steps 4 and 8e. Before
# training, a parity check confirms that row i of each lead-0 file carries the
# news of row i-1 of its lead-1 twin.
#
# Output  TimeXer/dataset/stock/<TK>_graph_lead0.csv and <TK>_sent_lead0.csv,
#         TimeXer/checkpoints_lead0_<TK>/, TimeXer/result_lead0_<TK>.txt,
#         fnspid_transformer/results_forecast_prevday/<input>_s<seed>/<TK>_predictions.csv
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/9_previous_day_news.sh
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

test ! -e fnspid_transformer/results_forecast_prevday || { echo "ERROR: results_forecast_prevday exists; archive it first."; exit 1; }
test ! -e TimeXer/result_lead0_T.txt || { echo "ERROR: TimeXer/result_lead0_T.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_lead0_INTC.txt || { echo "ERROR: TimeXer/result_lead0_INTC.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_lead0_AMD.txt || { echo "ERROR: TimeXer/result_lead0_AMD.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_lead0_CVX.txt || { echo "ERROR: TimeXer/result_lead0_CVX.txt exists; archive it first."; exit 1; }
test ! -e TimeXer/result_lead0_BABA.txt || { echo "ERROR: TimeXer/result_lead0_BABA.txt exists; archive it first."; exit 1; }

echo
echo "=== JOINING WITH LEAD 0: the last observed day's own news representation ==="
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 0 \
  --suffix _graph_lead0
python gnn/join_embeddings_to_stock.py \
  --stock_dir dataset/full_history_fixed \
  --embeddings_csv gnn_outputs_sentiment/daily_ticker_sentiment.csv \
  --output_dir TimeXer/dataset/stock \
  --tickers T,INTC,AMD,CVX,BABA \
  --graph_target_lead 0 \
  --suffix _sent_lead0

echo
echo "=== PARITY: same dates as the lead-1 files, and each row's news is its own date's ==="
python - <<'EOF'
import csv
def rows(p):
    with open(p, newline="") as h:
        return list(csv.DictReader(h))
bad = []
for tk in ("T", "INTC", "AMD", "CVX", "BABA"):
    for new, old in ((f"{tk}_graph_lead0", f"{tk}_graph"), (f"{tk}_sent_lead0", f"{tk}_sent")):
        a = rows(f"TimeXer/dataset/stock/{new}.csv"); b = rows(f"TimeXer/dataset/stock/{old}.csv")
        if [r["date"] for r in a] != [r["date"] for r in b]:
            bad.append(f"{new}: dates differ"); continue
        cols = [c for c in a[0] if c.startswith("gnn_")]
        # lead 0 on row i must equal lead 1 on row i-1 (the same date's representation)
        shifted = all(all(a[i][c] == b[i - 1][c] for c in cols) for i in range(1, len(a)))
        print(f"  {'OK  ' if shifted else 'FAIL'} {new}.csv  {len(a)} rows; row i holds the representation lead 1 put on row i-1")
        if not shifted:
            bad.append(f"{new}: not a one-row shift")
if bad:
    raise SystemExit("ERROR: " + "; ".join(bad))
EOF

test -s TimeXer/checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth || { echo "ERROR: missing parent T s11"; exit 1; }
test -s TimeXer/checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth || { echo "ERROR: missing parent T s22"; exit 1; }
test -s TimeXer/checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth || { echo "ERROR: missing parent T s33"; exit 1; }
test -s TimeXer/checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth || { echo "ERROR: missing parent T s44"; exit 1; }
test -s TimeXer/checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth || { echo "ERROR: missing parent T s55"; exit 1; }
test -s TimeXer/checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth || { echo "ERROR: missing parent INTC s11"; exit 1; }
test -s TimeXer/checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth || { echo "ERROR: missing parent INTC s22"; exit 1; }
test -s TimeXer/checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth || { echo "ERROR: missing parent INTC s33"; exit 1; }
test -s TimeXer/checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth || { echo "ERROR: missing parent INTC s44"; exit 1; }
test -s TimeXer/checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth || { echo "ERROR: missing parent INTC s55"; exit 1; }
test -s TimeXer/checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth || { echo "ERROR: missing parent AMD s11"; exit 1; }
test -s TimeXer/checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth || { echo "ERROR: missing parent AMD s22"; exit 1; }
test -s TimeXer/checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth || { echo "ERROR: missing parent AMD s33"; exit 1; }
test -s TimeXer/checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth || { echo "ERROR: missing parent AMD s44"; exit 1; }
test -s TimeXer/checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth || { echo "ERROR: missing parent AMD s55"; exit 1; }
test -s TimeXer/checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth || { echo "ERROR: missing parent CVX s11"; exit 1; }
test -s TimeXer/checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth || { echo "ERROR: missing parent CVX s22"; exit 1; }
test -s TimeXer/checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth || { echo "ERROR: missing parent CVX s33"; exit 1; }
test -s TimeXer/checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth || { echo "ERROR: missing parent CVX s44"; exit 1; }
test -s TimeXer/checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth || { echo "ERROR: missing parent CVX s55"; exit 1; }
test -s TimeXer/checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth || { echo "ERROR: missing parent BABA s11"; exit 1; }
test -s TimeXer/checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth || { echo "ERROR: missing parent BABA s22"; exit 1; }
test -s TimeXer/checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth || { echo "ERROR: missing parent BABA s33"; exit 1; }
test -s TimeXer/checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth || { echo "ERROR: missing parent BABA s44"; exit 1; }
test -s TimeXer/checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth || { echo "ERROR: missing parent BABA s55"; exit 1; }
echo "  all 25 price-only parents present"

# ############################################################################
# TIMEXER, previous-day news: graph and sentiment (50 runs), in the background
# ############################################################################
(
cd "$REPO_ROOT/TimeXer"
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_lead0.csv \
  --model_id GRAPH_L0_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_lead0.csv \
  --model_id SENT_L0_T_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_T_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_lead0.csv \
  --model_id GRAPH_L0_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_lead0.csv \
  --model_id SENT_L0_T_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_T_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_lead0.csv \
  --model_id GRAPH_L0_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_lead0.csv \
  --model_id SENT_L0_T_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_T_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_lead0.csv \
  --model_id GRAPH_L0_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_lead0.csv \
  --model_id SENT_L0_T_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_T_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_graph_lead0.csv \
  --model_id GRAPH_L0_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path T_sent_lead0.csv \
  --model_id SENT_L0_T_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_T_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_T/long_term_forecast_PRICE_T_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_T_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_T/ \
  --result_file result_lead0_T.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_lead0.csv \
  --model_id GRAPH_L0_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_lead0.csv \
  --model_id SENT_L0_INTC_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_INTC_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_lead0.csv \
  --model_id GRAPH_L0_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_lead0.csv \
  --model_id SENT_L0_INTC_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_INTC_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_lead0.csv \
  --model_id GRAPH_L0_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_lead0.csv \
  --model_id SENT_L0_INTC_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_INTC_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_lead0.csv \
  --model_id GRAPH_L0_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_lead0.csv \
  --model_id SENT_L0_INTC_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_INTC_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_graph_lead0.csv \
  --model_id GRAPH_L0_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path INTC_sent_lead0.csv \
  --model_id SENT_L0_INTC_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_INTC_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_INTC/long_term_forecast_PRICE_INTC_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_INTC_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_INTC/ \
  --result_file result_lead0_INTC.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_lead0.csv \
  --model_id GRAPH_L0_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_lead0.csv \
  --model_id SENT_L0_AMD_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_AMD_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_lead0.csv \
  --model_id GRAPH_L0_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_lead0.csv \
  --model_id SENT_L0_AMD_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_AMD_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_lead0.csv \
  --model_id GRAPH_L0_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_lead0.csv \
  --model_id SENT_L0_AMD_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_AMD_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_lead0.csv \
  --model_id GRAPH_L0_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_lead0.csv \
  --model_id SENT_L0_AMD_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_AMD_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_graph_lead0.csv \
  --model_id GRAPH_L0_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path AMD_sent_lead0.csv \
  --model_id SENT_L0_AMD_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_AMD_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_AMD/long_term_forecast_PRICE_AMD_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_AMD_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_AMD/ \
  --result_file result_lead0_AMD.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_lead0.csv \
  --model_id GRAPH_L0_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_lead0.csv \
  --model_id SENT_L0_CVX_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_CVX_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_lead0.csv \
  --model_id GRAPH_L0_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_lead0.csv \
  --model_id SENT_L0_CVX_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_CVX_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_lead0.csv \
  --model_id GRAPH_L0_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_lead0.csv \
  --model_id SENT_L0_CVX_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_CVX_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_lead0.csv \
  --model_id GRAPH_L0_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_lead0.csv \
  --model_id SENT_L0_CVX_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_CVX_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_graph_lead0.csv \
  --model_id GRAPH_L0_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path CVX_sent_lead0.csv \
  --model_id SENT_L0_CVX_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_CVX_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_CVX/long_term_forecast_PRICE_CVX_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_CVX_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_CVX/ \
  --result_file result_lead0_CVX.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_lead0.csv \
  --model_id GRAPH_L0_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_lead0.csv \
  --model_id SENT_L0_BABA_s11 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_BABA_s11 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 11 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_lead0.csv \
  --model_id GRAPH_L0_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_lead0.csv \
  --model_id SENT_L0_BABA_s22 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_BABA_s22 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 22 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_lead0.csv \
  --model_id GRAPH_L0_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_lead0.csv \
  --model_id SENT_L0_BABA_s33 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_BABA_s33 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 33 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_lead0.csv \
  --model_id GRAPH_L0_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_lead0.csv \
  --model_id SENT_L0_BABA_s44 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_BABA_s44 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 44 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_graph_lead0.csv \
  --model_id GRAPH_L0_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 70 --dec_in 70 --c_out 1 \
  --des GRAPH_L0_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 64 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --data_path BABA_sent_lead0.csv \
  --model_id SENT_L0_BABA_s55 --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 1 \
  --des SENT_L0_BABA_s55 --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --learning_rate 1e-3 --lradj constant \
  --seed 55 --inverse --itr 1 \
  --graph_feature_count 1 --graph_adapter_dim 64 --graph_gate_init 0 \
  --pretrained_checkpoint ./checkpoints_BABA/long_term_forecast_PRICE_BABA_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_PRICE_BABA_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_lead0_BABA/ \
  --result_file result_lead0_BABA.txt --no-plot_data_splits

) > "logs/9_timexer_${SLURM_JOB_ID}.out" 2>&1 &
PID_TIMEXER=$!

# ############################################################################
# FNSPID TRANSFORMER, previous-day news: sentiment (5 runs x 5 companies), in the background
# ############################################################################
(
cd "$REPO_ROOT/fnspid_transformer"
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 11 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 22 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 33 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 44 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 55 --epochs 100 --out_dir results_forecast_prevday
) > "logs/9_tr_sentiment_${SLURM_JOB_ID}.out" 2>&1 &
PID_TR_SENTIMENT=$!

# ############################################################################
# FNSPID TRANSFORMER, previous-day news: graph (5 runs x 5 companies), in the background
# ############################################################################
(
cd "$REPO_ROOT/fnspid_transformer"
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 11 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 22 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 33 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 44 --epochs 100 --out_dir results_forecast_prevday
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_lead0 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 55 --epochs 100 --out_dir results_forecast_prevday
) > "logs/9_tr_graph_${SLURM_JOB_ID}.out" 2>&1 &
PID_TR_GRAPH=$!

echo "=== RUNNING: TimeXer ${PID_TIMEXER}, Transformer sentiment ${PID_TR_SENTIMENT}, graph ${PID_TR_GRAPH} ==="
set +e
wait $PID_TIMEXER; S1=$?
wait $PID_TR_SENTIMENT; S2=$?
wait $PID_TR_GRAPH; S3=$?
set -e
echo "exit status: TimeXer ${S1}, Transformer sentiment ${S2}, Transformer graph ${S3}"
if [ $S1 -ne 0 ] || [ $S2 -ne 0 ] || [ $S3 -ne 0 ]; then
  exit 1
fi
echo "=== 9 done ==="
