#!/bin/bash
#SBATCH --job-name=8e_fnspid_onestep
#SBATCH --output=logs/8e_fnspid_onestep_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# =============================================================================
# Step 8e - FNSPID Transformer on the one-step task
#
# The Transformer and recipe of step 8c on the main task of step 4: a 10-day
# window, the next trading day's adjusted close, target-day news on the last row
# (join lead 1), and the 302 test days per company. The Transformer reads the
# series it forecasts, so its inputs are volume, open, and adjusted close.
#   price_only  <TK>_price_only.csv (step 4)
#   sentiment   <TK>_sent.csv (step 7)
#   graph       <TK>_graph.csv (step 4)
# TimeXer runs for all three inputs exist from steps 4 and 7. The three inputs run
# side by side on one GPU, one log each (logs/8e_<input>_<job>.out).
#
# Output  fnspid_transformer/results_forecast_onestep/<input>_s<seed>/<TK>_predictions.csv
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/8e_fnspid_transformer_onestep.sh
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

test ! -e fnspid_transformer/results_forecast_onestep || { echo "ERROR: fnspid_transformer/results_forecast_onestep exists; archive it first."; exit 1; }
for f in T INTC AMD CVX BABA; do
  for s in _price_only _sent _graph; do
    test -s "TimeXer/dataset/stock/${f}${s}.csv" || { echo "ERROR: missing TimeXer/dataset/stock/${f}${s}.csv"; exit 1; }
  done
done
echo "  all 15 input files present"

# ############################################################################
# price_only (5 runs x 5 companies), in the background
# ############################################################################
(
cd "$REPO_ROOT/fnspid_transformer"
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode price_only --seed 11 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode price_only --seed 22 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode price_only --seed 33 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode price_only --seed 44 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode price_only --seed 55 --epochs 100 --out_dir results_forecast_onestep
) > "logs/8e_price_only_${SLURM_JOB_ID}.out" 2>&1 &
PID_PRICE_ONLY=$!

# ############################################################################
# sentiment (5 runs x 5 companies), in the background
# ############################################################################
(
cd "$REPO_ROOT/fnspid_transformer"
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 11 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 22 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 33 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 44 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _sent --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 1 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode sentiment --seed 55 --epochs 100 --out_dir results_forecast_onestep
) > "logs/8e_sentiment_${SLURM_JOB_ID}.out" 2>&1 &
PID_SENTIMENT=$!

# ############################################################################
# graph (5 runs x 5 companies), in the background
# ############################################################################
(
cd "$REPO_ROOT/fnspid_transformer"
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 11 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 22 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 33 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 44 --epochs 100 --out_dir results_forecast_onestep
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,adj_close --target adj_close --graph_dims 64 --input_length 10 --output_length 1 \
  --val_end "$VAL_END" --mode graph --seed 55 --epochs 100 --out_dir results_forecast_onestep
) > "logs/8e_graph_${SLURM_JOB_ID}.out" 2>&1 &
PID_GRAPH=$!

echo "=== RUNNING: price_only ${PID_PRICE_ONLY}, sentiment ${PID_SENTIMENT}, graph ${PID_GRAPH} ==="
set +e
wait $PID_PRICE_ONLY; S1=$?
wait $PID_SENTIMENT; S2=$?
wait $PID_GRAPH; S3=$?
set -e
echo "exit status: price_only ${S1}, sentiment ${S2}, graph ${S3}"
if [ $S1 -ne 0 ] || [ $S2 -ne 0 ] || [ $S3 -ne 0 ]; then
  exit 1
fi
echo "=== 8e done ==="
