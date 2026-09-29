#!/bin/bash
#SBATCH --job-name=8b_fnspid_exact
#SBATCH --output=logs/8b_fnspid_exact_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# =============================================================================
# Step 8b - The FNSPID Transformer experiment as published, on the thesis companies
#
# Their code and protocol on T, INTC, AMD, CVX and BABA, with three inputs:
#   nonsentiment  Volume, Open, Close           (their "A-Non.")
#   sentiment     + their Scaled_sentiment      (their "A-Sen.")
#   graph         + the 64 graph dimensions of step 3
# One pass over the five companies with the model carried forward, 100 epochs per
# company, each company evaluated after its own pass, five seeds.
#
# In this protocol the target is the close of the last row of the input window,
# a value the model receives. Each evaluation file therefore also records copy_*
# (repeat that close: MSE 0, R2 1) and prev_* (repeat the close before it). The
# results are comparable with the paper's Table 3, not with TimeXer; step 8c is
# the comparison with TimeXer.
#
# Input   fnspid_transformer/data_ours/<TK>.csv (fnspid_transformer/build_data.py)
# Output  fnspid_transformer/results_exact/<mode>_s<seed>/<TK>_eval_data.csv
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/8b_fnspid_transformer_exact.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT/fnspid_transformer"

# ==================================================== seed 11
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --mode nonsentiment --seed 11 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 11 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --graph_dims 64 --mode graph --seed 11 --epochs 100 --out_dir results_exact

# ==================================================== seed 22
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --mode nonsentiment --seed 22 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 22 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --graph_dims 64 --mode graph --seed 22 --epochs 100 --out_dir results_exact

# ==================================================== seed 33
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --mode nonsentiment --seed 33 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 33 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --graph_dims 64 --mode graph --seed 33 --epochs 100 --out_dir results_exact

# ==================================================== seed 44
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --mode nonsentiment --seed 44 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 44 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --graph_dims 64 --mode graph --seed 44 --epochs 100 --out_dir results_exact

# ==================================================== seed 55
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --mode nonsentiment --seed 55 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 55 --epochs 100 --out_dir results_exact
python run_exact.py --data_dir data_ours --stocks T,INTC,AMD,CVX,BABA --eval_stocks T,INTC,AMD,CVX,BABA \
  --columns Volume,Open,Close --graph_dims 64 --mode graph --seed 55 --epochs 100 --out_dir results_exact

echo "=== 8b done ==="
