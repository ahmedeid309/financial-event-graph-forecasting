#!/bin/bash
# =============================================================================
# Submit steps 3, 4 and 4b as one dependent chain
#
# Run on the login node from the repository root once the graph exists
# (step 2) or while its build job is still queued:
#   bash scripts/run_chain.sh
#
# Each step is submitted with an afterok dependency, so it starts only if the
# step before it succeeded; after a failure the remaining steps stay queued in
# state DependencyNeverSatisfied instead of running on incomplete inputs.
# The steps are serial because step 4b loads the price-only checkpoints and
# joined files of step 4, and steps 4 and 4b both read the split of step 3.
#
# Outputs of a previous run are first moved to archive/run_<timestamp>/:
# summarize_runs.py averages every metrics file it finds and TimeXer appends to
# its result files, so old outputs would mix with new ones. The text-embedding
# cache and the example cache are kept; both are keyed on content (article
# text and graph fingerprint), so they cannot go stale.
# =============================================================================

set -euo pipefail

STAMP=$(date +%Y%m%d_%H%M)
ARCHIVE="archive/run_${STAMP}"

echo "=== PRECONDITIONS ==="
if squeue -u "$USER" -h -o '%j' | grep -q '^1_extract'; then
  echo "ERROR: extraction jobs are still running; step 2 would read a partial partition."
  exit 1
fi

BUILD_JOB=$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /^2_build/ {print $1; exit}')
if [ -n "$BUILD_JOB" ]; then
  echo "  graph build job ${BUILD_JOB} is running; the chain waits for it"
  DEP="--dependency=afterok:${BUILD_JOB}"
else
  test -s ekg_final/articles.csv || {
    echo "ERROR: no graph in ekg_final/ and no build running. Submit scripts/2_build_graph.sh first."
    exit 1
  }
  echo "  using the graph in ekg_final/"
  DEP=""
fi

echo
echo "=== ARCHIVING THE PREVIOUS RUN -> ${ARCHIVE} ==="
mkdir -p "$ARCHIVE"
for item in gnn_outputs_shared gnn_outputs_text \
            TimeXer/checkpoints_{T,INTC,AMD,CVX,BABA} \
            TimeXer/result_{T,INTC,AMD,CVX,BABA}.txt \
            TimeXer/result_text_{T,INTC,AMD,CVX,BABA}.txt; do
  if [ -e "$item" ]; then
    mkdir -p "$ARCHIVE/$(dirname "$item")"
    mv "$item" "$ARCHIVE/$item"
    echo "  moved  $item"
  fi
done

# Steps 4 and 4b regenerate the joined TimeXer inputs; clearing them means a
# parity check can never compare a fresh file with a stale twin.
rm -f TimeXer/dataset/stock/*_graph.csv \
      TimeXer/dataset/stock/*_price_only.csv \
      TimeXer/dataset/stock/*_text64.csv \
      TimeXer/dataset/stock/*_text1024.csv
echo "  cleared the joined inputs in TimeXer/dataset/stock/"

echo
echo "=== SUBMITTING THE CHAIN ==="
# shellcheck disable=SC2086
J3=$(sbatch --parsable $DEP scripts/3_train_gnn.sh)
echo "  ${J3}  3_train_gnn         ${DEP:-(starts immediately)}"
J4=$(sbatch --parsable --dependency=afterok:"$J3" scripts/4_train_timexer.sh)
echo "  ${J4}  4_train_timexer     after ${J3} succeeds"
J4B=$(sbatch --parsable --dependency=afterok:"$J4" scripts/4b_train_text_baseline.sh)
echo "  ${J4B}  4b_text_baseline    after ${J4} succeeds"

echo
echo "Cancel with:  scancel ${J3} ${J4} ${J4B}"
echo "Logs:         logs/3_train_gnn_${J3}.out, logs/4_train_timexer_${J4}.out,"
echo "              logs/4b_text_baseline_${J4B}.out"
echo "Previous run archived at ${ARCHIVE}"
