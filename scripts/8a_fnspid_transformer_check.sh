#!/bin/bash
#SBATCH --job-name=8a_fnspid_check
#SBATCH --output=logs/8a_fnspid_check_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1

# =============================================================================
# Step 8a - Reproduction check of the FNSPID Transformer
#
# Runs the FNSPID authors' Transformer (fnspid_transformer/run_exact.py is their
# run.py at commit 5873ff8, with their tst package) on the five stocks their
# paper evaluates (KO, AMD, TSM, GOOG, WMT), with their data files (commit
# eecdcdb) in their training order, without and with their sentiment score, for
# five seeds. Compare with row "5 / Transformer" of their Table 3:
# A-Non. MAE .01883, MSE .00060, R2 .86659; A-Sen. MAE .01801, MSE .00058,
# R2 .87260 (min-max-scaled close).
#
# Input   fnspid_transformer/data_theirs/ (fnspid_transformer/build_data.py)
# Output  fnspid_transformer/results_check/<mode>_s<seed>/<STOCK>_eval_data.csv
#
# Submit from the repository root with the Python environment active:
#   sbatch scripts/8a_fnspid_transformer_check.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(pwd)"
cd "$REPO_ROOT/fnspid_transformer"

python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close --mode nonsentiment --seed 11 --epochs 100 --out_dir results_check
python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 11 --epochs 100 --out_dir results_check

python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close --mode nonsentiment --seed 22 --epochs 100 --out_dir results_check
python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 22 --epochs 100 --out_dir results_check

python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close --mode nonsentiment --seed 33 --epochs 100 --out_dir results_check
python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 33 --epochs 100 --out_dir results_check

python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close --mode nonsentiment --seed 44 --epochs 100 --out_dir results_check
python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 44 --epochs 100 --out_dir results_check

python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close --mode nonsentiment --seed 55 --epochs 100 --out_dir results_check
python run_exact.py --data_dir data_theirs --stocks KO,AMD,TSM,GOOG,WMT --eval_stocks KO,AMD,TSM,GOOG,WMT \
  --columns Volume,Open,Close,Scaled_sentiment --mode sentiment --seed 55 --epochs 100 --out_dir results_check

echo "=== 8a done ==="
