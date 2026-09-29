#!/bin/bash
#SBATCH --job-name=5_ablate_fusion
#SBATCH --output=logs/5_ablate_fusion_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1

# =============================================================================
# Step 5 - Fusion ablation
#
# The graph pathway of step 4 has three components that can be switched off:
#   attention  the graph gets its own cross-attention; without it, the graph
#              factors join the ordinary exogenous price variables
#   gate       a tanh gate on the graph residual that starts at zero; without it,
#              the residual is added at full strength from the first step
#   reduction  the 64 graph values are reduced to 4 before the adapter; without
#              it, all 64 values pass
# For each company and seed, the step trains a price-only base model (ABL_P) and,
# from it, all eight on/off combinations with the price pathway frozen:
#   A   attention                   AR  attention+reduction
#   AG  attention+gate (step 4)     GR  gate+reduction
#   F   attention+gate+reduction    R   reduction
#   G   gate                        C2  none (plain concatenation)
# 5 companies x 5 seeds x (1 + 8) = 225 runs.
#
# Input   TimeXer/dataset/stock/<TK>_graph.csv and <TK>_price_only.csv (step 4)
# Output  TimeXer/checkpoints_ablation_<TK>/, TimeXer/result_ablation_<TK>.txt
# The job stops if a result file already exists, because results are appended.
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/5_ablate_fusion.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

echo "=== INPUT CHECK ==="
test -s TimeXer/dataset/stock/T_graph.csv || { echo "ERROR: T_graph.csv missing - run step 4 first"; exit 1; }
test -s TimeXer/dataset/stock/T_price_only.csv || { echo "ERROR: T_price_only.csv missing - run step 4 first"; exit 1; }
test ! -e TimeXer/result_ablation_T.txt || { echo "ERROR: TimeXer/result_ablation_T.txt exists; results are appended, so remove it first"; exit 1; }
test -s TimeXer/dataset/stock/INTC_graph.csv || { echo "ERROR: INTC_graph.csv missing - run step 4 first"; exit 1; }
test -s TimeXer/dataset/stock/INTC_price_only.csv || { echo "ERROR: INTC_price_only.csv missing - run step 4 first"; exit 1; }
test ! -e TimeXer/result_ablation_INTC.txt || { echo "ERROR: TimeXer/result_ablation_INTC.txt exists; results are appended, so remove it first"; exit 1; }
test -s TimeXer/dataset/stock/AMD_graph.csv || { echo "ERROR: AMD_graph.csv missing - run step 4 first"; exit 1; }
test -s TimeXer/dataset/stock/AMD_price_only.csv || { echo "ERROR: AMD_price_only.csv missing - run step 4 first"; exit 1; }
test ! -e TimeXer/result_ablation_AMD.txt || { echo "ERROR: TimeXer/result_ablation_AMD.txt exists; results are appended, so remove it first"; exit 1; }
test -s TimeXer/dataset/stock/CVX_graph.csv || { echo "ERROR: CVX_graph.csv missing - run step 4 first"; exit 1; }
test -s TimeXer/dataset/stock/CVX_price_only.csv || { echo "ERROR: CVX_price_only.csv missing - run step 4 first"; exit 1; }
test ! -e TimeXer/result_ablation_CVX.txt || { echo "ERROR: TimeXer/result_ablation_CVX.txt exists; results are appended, so remove it first"; exit 1; }
test -s TimeXer/dataset/stock/BABA_graph.csv || { echo "ERROR: BABA_graph.csv missing - run step 4 first"; exit 1; }
test -s TimeXer/dataset/stock/BABA_price_only.csv || { echo "ERROR: BABA_price_only.csv missing - run step 4 first"; exit 1; }
test ! -e TimeXer/result_ablation_BABA.txt || { echo "ERROR: TimeXer/result_ablation_BABA.txt exists; results are appended, so remove it first"; exit 1; }

# The split decided in step 3, read back from the exported representations.
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
echo "train ends ${TRAIN_END} | val ends ${VAL_END} | test after that"

cd "$REPO_ROOT/TimeXer"

# ############################################################################
# T
# ############################################################################

# ==================================================== T, seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_price_only.csv \
  --model_id ABL_P_s11 --des ABL_P_s11 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 11
test -s ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_A_s11 --des ABL_A_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AG_s11 --des ABL_AG_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_F_s11 --des ABL_F_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AR_s11 --des ABL_AR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_GR_s11 --des ABL_GR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_R_s11 --des ABL_R_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_G_s11 --des ABL_G_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_C2_s11 --des ABL_C2_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== T, seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_price_only.csv \
  --model_id ABL_P_s22 --des ABL_P_s22 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 22
test -s ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_A_s22 --des ABL_A_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AG_s22 --des ABL_AG_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_F_s22 --des ABL_F_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AR_s22 --des ABL_AR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_GR_s22 --des ABL_GR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_R_s22 --des ABL_R_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_G_s22 --des ABL_G_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_C2_s22 --des ABL_C2_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== T, seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_price_only.csv \
  --model_id ABL_P_s33 --des ABL_P_s33 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 33
test -s ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_A_s33 --des ABL_A_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AG_s33 --des ABL_AG_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_F_s33 --des ABL_F_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AR_s33 --des ABL_AR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_GR_s33 --des ABL_GR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_R_s33 --des ABL_R_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_G_s33 --des ABL_G_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_C2_s33 --des ABL_C2_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== T, seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_price_only.csv \
  --model_id ABL_P_s44 --des ABL_P_s44 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 44
test -s ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_A_s44 --des ABL_A_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AG_s44 --des ABL_AG_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_F_s44 --des ABL_F_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AR_s44 --des ABL_AR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_GR_s44 --des ABL_GR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_R_s44 --des ABL_R_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_G_s44 --des ABL_G_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_C2_s44 --des ABL_C2_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== T, seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_price_only.csv \
  --model_id ABL_P_s55 --des ABL_P_s55 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 55
test -s ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_A_s55 --des ABL_A_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AG_s55 --des ABL_AG_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_F_s55 --des ABL_F_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_AR_s55 --des ABL_AR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_GR_s55 --des ABL_GR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_R_s55 --des ABL_R_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_G_s55 --des ABL_G_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_T/ --result_file result_ablation_T.txt \
  --data_path T_graph.csv \
  --model_id ABL_C2_s55 --des ABL_C2_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_T/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

python summarize_ablation.py --result_file result_ablation_T.txt || true

# ############################################################################
# INTC
# ############################################################################

# ==================================================== INTC, seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_price_only.csv \
  --model_id ABL_P_s11 --des ABL_P_s11 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 11
test -s ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_A_s11 --des ABL_A_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AG_s11 --des ABL_AG_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_F_s11 --des ABL_F_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AR_s11 --des ABL_AR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_GR_s11 --des ABL_GR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_R_s11 --des ABL_R_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_G_s11 --des ABL_G_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_C2_s11 --des ABL_C2_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== INTC, seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_price_only.csv \
  --model_id ABL_P_s22 --des ABL_P_s22 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 22
test -s ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_A_s22 --des ABL_A_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AG_s22 --des ABL_AG_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_F_s22 --des ABL_F_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AR_s22 --des ABL_AR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_GR_s22 --des ABL_GR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_R_s22 --des ABL_R_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_G_s22 --des ABL_G_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_C2_s22 --des ABL_C2_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== INTC, seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_price_only.csv \
  --model_id ABL_P_s33 --des ABL_P_s33 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 33
test -s ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_A_s33 --des ABL_A_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AG_s33 --des ABL_AG_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_F_s33 --des ABL_F_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AR_s33 --des ABL_AR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_GR_s33 --des ABL_GR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_R_s33 --des ABL_R_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_G_s33 --des ABL_G_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_C2_s33 --des ABL_C2_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== INTC, seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_price_only.csv \
  --model_id ABL_P_s44 --des ABL_P_s44 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 44
test -s ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_A_s44 --des ABL_A_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AG_s44 --des ABL_AG_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_F_s44 --des ABL_F_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AR_s44 --des ABL_AR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_GR_s44 --des ABL_GR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_R_s44 --des ABL_R_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_G_s44 --des ABL_G_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_C2_s44 --des ABL_C2_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== INTC, seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_price_only.csv \
  --model_id ABL_P_s55 --des ABL_P_s55 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 55
test -s ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_A_s55 --des ABL_A_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AG_s55 --des ABL_AG_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_F_s55 --des ABL_F_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_AR_s55 --des ABL_AR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_GR_s55 --des ABL_GR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_R_s55 --des ABL_R_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_G_s55 --des ABL_G_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_INTC/ --result_file result_ablation_INTC.txt \
  --data_path INTC_graph.csv \
  --model_id ABL_C2_s55 --des ABL_C2_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_INTC/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

python summarize_ablation.py --result_file result_ablation_INTC.txt || true

# ############################################################################
# AMD
# ############################################################################

# ==================================================== AMD, seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_price_only.csv \
  --model_id ABL_P_s11 --des ABL_P_s11 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 11
test -s ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_A_s11 --des ABL_A_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AG_s11 --des ABL_AG_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_F_s11 --des ABL_F_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AR_s11 --des ABL_AR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_GR_s11 --des ABL_GR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_R_s11 --des ABL_R_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_G_s11 --des ABL_G_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_C2_s11 --des ABL_C2_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== AMD, seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_price_only.csv \
  --model_id ABL_P_s22 --des ABL_P_s22 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 22
test -s ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_A_s22 --des ABL_A_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AG_s22 --des ABL_AG_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_F_s22 --des ABL_F_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AR_s22 --des ABL_AR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_GR_s22 --des ABL_GR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_R_s22 --des ABL_R_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_G_s22 --des ABL_G_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_C2_s22 --des ABL_C2_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== AMD, seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_price_only.csv \
  --model_id ABL_P_s33 --des ABL_P_s33 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 33
test -s ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_A_s33 --des ABL_A_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AG_s33 --des ABL_AG_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_F_s33 --des ABL_F_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AR_s33 --des ABL_AR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_GR_s33 --des ABL_GR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_R_s33 --des ABL_R_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_G_s33 --des ABL_G_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_C2_s33 --des ABL_C2_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== AMD, seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_price_only.csv \
  --model_id ABL_P_s44 --des ABL_P_s44 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 44
test -s ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_A_s44 --des ABL_A_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AG_s44 --des ABL_AG_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_F_s44 --des ABL_F_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AR_s44 --des ABL_AR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_GR_s44 --des ABL_GR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_R_s44 --des ABL_R_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_G_s44 --des ABL_G_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_C2_s44 --des ABL_C2_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== AMD, seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_price_only.csv \
  --model_id ABL_P_s55 --des ABL_P_s55 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 55
test -s ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_A_s55 --des ABL_A_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AG_s55 --des ABL_AG_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_F_s55 --des ABL_F_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_AR_s55 --des ABL_AR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_GR_s55 --des ABL_GR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_R_s55 --des ABL_R_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_G_s55 --des ABL_G_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_AMD/ --result_file result_ablation_AMD.txt \
  --data_path AMD_graph.csv \
  --model_id ABL_C2_s55 --des ABL_C2_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_AMD/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

python summarize_ablation.py --result_file result_ablation_AMD.txt || true

# ############################################################################
# CVX
# ############################################################################

# ==================================================== CVX, seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_price_only.csv \
  --model_id ABL_P_s11 --des ABL_P_s11 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 11
test -s ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_A_s11 --des ABL_A_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AG_s11 --des ABL_AG_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_F_s11 --des ABL_F_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AR_s11 --des ABL_AR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_GR_s11 --des ABL_GR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_R_s11 --des ABL_R_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_G_s11 --des ABL_G_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_C2_s11 --des ABL_C2_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== CVX, seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_price_only.csv \
  --model_id ABL_P_s22 --des ABL_P_s22 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 22
test -s ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_A_s22 --des ABL_A_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AG_s22 --des ABL_AG_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_F_s22 --des ABL_F_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AR_s22 --des ABL_AR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_GR_s22 --des ABL_GR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_R_s22 --des ABL_R_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_G_s22 --des ABL_G_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_C2_s22 --des ABL_C2_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== CVX, seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_price_only.csv \
  --model_id ABL_P_s33 --des ABL_P_s33 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 33
test -s ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_A_s33 --des ABL_A_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AG_s33 --des ABL_AG_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_F_s33 --des ABL_F_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AR_s33 --des ABL_AR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_GR_s33 --des ABL_GR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_R_s33 --des ABL_R_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_G_s33 --des ABL_G_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_C2_s33 --des ABL_C2_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== CVX, seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_price_only.csv \
  --model_id ABL_P_s44 --des ABL_P_s44 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 44
test -s ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_A_s44 --des ABL_A_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AG_s44 --des ABL_AG_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_F_s44 --des ABL_F_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AR_s44 --des ABL_AR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_GR_s44 --des ABL_GR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_R_s44 --des ABL_R_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_G_s44 --des ABL_G_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_C2_s44 --des ABL_C2_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== CVX, seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_price_only.csv \
  --model_id ABL_P_s55 --des ABL_P_s55 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 55
test -s ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_A_s55 --des ABL_A_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AG_s55 --des ABL_AG_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_F_s55 --des ABL_F_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_AR_s55 --des ABL_AR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_GR_s55 --des ABL_GR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_R_s55 --des ABL_R_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_G_s55 --des ABL_G_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_CVX/ --result_file result_ablation_CVX.txt \
  --data_path CVX_graph.csv \
  --model_id ABL_C2_s55 --des ABL_C2_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_CVX/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

python summarize_ablation.py --result_file result_ablation_CVX.txt || true

# ############################################################################
# BABA
# ############################################################################

# ==================================================== BABA, seed 11
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_price_only.csv \
  --model_id ABL_P_s11 --des ABL_P_s11 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 11
test -s ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_A_s11 --des ABL_A_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AG_s11 --des ABL_AG_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_F_s11 --des ABL_F_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AR_s11 --des ABL_AR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_GR_s11 --des ABL_GR_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_R_s11 --des ABL_R_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_G_s11 --des ABL_G_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_C2_s11 --des ABL_C2_s11 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 11 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s11_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s11_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== BABA, seed 22
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_price_only.csv \
  --model_id ABL_P_s22 --des ABL_P_s22 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 22
test -s ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_A_s22 --des ABL_A_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AG_s22 --des ABL_AG_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_F_s22 --des ABL_F_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AR_s22 --des ABL_AR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_GR_s22 --des ABL_GR_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_R_s22 --des ABL_R_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_G_s22 --des ABL_G_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_C2_s22 --des ABL_C2_s22 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 22 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s22_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s22_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== BABA, seed 33
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_price_only.csv \
  --model_id ABL_P_s33 --des ABL_P_s33 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 33
test -s ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_A_s33 --des ABL_A_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AG_s33 --des ABL_AG_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_F_s33 --des ABL_F_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AR_s33 --des ABL_AR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_GR_s33 --des ABL_GR_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_R_s33 --des ABL_R_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_G_s33 --des ABL_G_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_C2_s33 --des ABL_C2_s33 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 33 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s33_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s33_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== BABA, seed 44
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_price_only.csv \
  --model_id ABL_P_s44 --des ABL_P_s44 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 44
test -s ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_A_s44 --des ABL_A_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AG_s44 --des ABL_AG_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_F_s44 --des ABL_F_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AR_s44 --des ABL_AR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_GR_s44 --des ABL_GR_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_R_s44 --des ABL_R_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_G_s44 --des ABL_G_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_C2_s44 --des ABL_C2_s44 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 44 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s44_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s44_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# ==================================================== BABA, seed 55
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_price_only.csv \
  --model_id ABL_P_s55 --des ABL_P_s55 \
  --enc_in 6 --dec_in 6 \
  --learning_rate 1e-4 --seed 55
test -s ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth || { echo "ERROR: base checkpoint not written: ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth"; exit 1; }

# A: attention
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_A_s55 --des ABL_A_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AG: attention+gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AG_s55 --des ABL_AG_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# F: attention+gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_F_s55 --des ABL_F_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# AR: attention+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_AR_s55 --des ABL_AR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode attention --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# GR: gate+reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_GR_s55 --des ABL_GR_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# R: reduction
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_R_s55 --des ABL_R_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 4 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# G: gate
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_G_s55 --des ABL_G_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

# C2: none
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/stock/ --model TimeXer --data custom \
  --features MS --target adj_close --freq d \
  --seq_len 10 --label_len 5 --pred_len 1 \
  --e_layers 1 --d_layers 1 --factor 3 --c_out 1 \
  --patch_len 5 --d_model 32 --d_ff 32 --n_heads 4 \
  --batch_size 16 --train_epochs 50 --patience 10 --num_workers 0 \
  --lradj constant --inverse --itr 1 --no-plot_data_splits \
  --train_end "$TRAIN_END" --val_end "$VAL_END" \
  --checkpoints ./checkpoints_ablation_BABA/ --result_file result_ablation_BABA.txt \
  --data_path BABA_graph.csv \
  --model_id ABL_C2_s55 --des ABL_C2_s55 \
  --enc_in 70 --dec_in 70 \
  --learning_rate 1e-3 --seed 55 \
  --graph_feature_count 64 --graph_gate_init 0 \
  --graph_fusion_mode concat --no_graph_gate --graph_adapter_dim 64 \
  --pretrained_checkpoint ./checkpoints_ablation_BABA/long_term_forecast_ABL_P_s55_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_ABL_P_s55_0/checkpoint.pth \
  --freeze_price_pathway --validate_initial_model

python summarize_ablation.py --result_file result_ablation_BABA.txt || true

echo
echo "=== step 5 done: TimeXer/result_ablation_<TK>.txt ==="
