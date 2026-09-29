#!/bin/bash
#SBATCH --job-name=8c_fnspid_forecast
#SBATCH --output=logs/8c_fnspid_forecast_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# =============================================================================
# Step 8c - FNSPID Transformer and TimeXer on the three-day task
#
# The model and training recipe of step 8b on a genuine forecast: 50 rows in, the
# raw close of the next three rows out, scored on the same 300 x 3 test targets
# as step 6. fnspid_transformer/run_forecast.py documents what differs from the
# authors' code and why (target after the window, min-max scaling fitted on
# training rows, the thesis split).
#   price_only  volume, open, close             <TK>_price_only_h3.csv
#   graph       + the 64 graph dimensions       <TK>_graph_h3.csv
#
# Output  fnspid_transformer/results_forecast/<mode>_s<seed>/<TK>_predictions.csv
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/8c_fnspid_transformer_forecast.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"

read -r TRAIN_END VAL_END < <(python - <<'PYEOF'
import csv
last = {}
with open("gnn_outputs_shared/daily_ticker_embeddings.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["date"] > last.get(row["split"], ""):
            last[row["split"]] = row["date"]
if not last["train"] < last["val"] < last["test"]:
    raise SystemExit("ERROR: embedding splits are not chronologically ordered.")
print(last["train"], last["val"])
PYEOF
)
echo "=== SPLIT INHERITED FROM STAGE 3: train ends ${TRAIN_END}, val ends ${VAL_END} ==="
echo "Their recipe has no validation step, so every row up to ${VAL_END} is a training row."

cd "$REPO_ROOT/fnspid_transformer"

# ==================================================== seed 11
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --val_end "$VAL_END" --mode price_only --seed 11 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 64 --val_end "$VAL_END" --mode graph --seed 11 --epochs 100 --out_dir results_forecast

# ==================================================== seed 22
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --val_end "$VAL_END" --mode price_only --seed 22 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 64 --val_end "$VAL_END" --mode graph --seed 22 --epochs 100 --out_dir results_forecast

# ==================================================== seed 33
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --val_end "$VAL_END" --mode price_only --seed 33 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 64 --val_end "$VAL_END" --mode graph --seed 33 --epochs 100 --out_dir results_forecast

# ==================================================== seed 44
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --val_end "$VAL_END" --mode price_only --seed 44 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 64 --val_end "$VAL_END" --mode graph --seed 44 --epochs 100 --out_dir results_forecast

# ==================================================== seed 55
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _price_only_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --val_end "$VAL_END" --mode price_only --seed 55 --epochs 100 --out_dir results_forecast
python run_forecast.py --data_dir ../TimeXer/dataset/stock --suffix _graph_h3 --stocks T,INTC,AMD,CVX,BABA \
  --columns volume,open,close --graph_dims 64 --val_end "$VAL_END" --mode graph --seed 55 --epochs 100 --out_dir results_forecast

echo "=== 8c done ==="
