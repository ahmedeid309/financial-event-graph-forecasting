#!/bin/bash
# =============================================================================
# Step 0 - Third-party code and files that this repository does not redistribute
#
# Run once from the repository root on a machine with network access:
#   bash scripts/0_setup_external.sh
#
# 1. TimeXer (github.com/thuml/TimeXer) at commit 7601190, with the changes of
#    this thesis (timexer_overlay/) copied over it. Upstream TimeXer carries no
#    license file, so this repository keeps only the files it changes or adds.
# 2. From the FNSPID repository (github.com/Zdong104/FNSPID_Financial_News_Dataset,
#    CC BY-NC 4.0), at the two commits of 19 February 2024 used in the thesis:
#      5873ff8  the Transformer package (tst/) and the six released weight files
#      eecdcdb  the per-company data files with the authors' sentiment scores
#    The files land in fnspid_transformer/tst/, fnspid_transformer/their_weights_5873ff8/,
#    fnspid_sentiment/ (the five thesis companies) and external/fnspid_data_eecdcdb/
#    (the five companies of the FNSPID paper).
# =============================================================================

set -euo pipefail

# ------------------------------------------------------------------ TimeXer
if [ ! -d TimeXer/.git ]; then
  git clone https://github.com/thuml/TimeXer.git TimeXer
fi
git -C TimeXer checkout --quiet 76011909357972bd55a27adba2e1be994d81b327
cp timexer_overlay/run.py TimeXer/run.py
cp timexer_overlay/exp/exp_long_term_forecasting.py TimeXer/exp/exp_long_term_forecasting.py
cp timexer_overlay/models/TimeXer.py TimeXer/models/TimeXer.py
cp timexer_overlay/data_provider/data_loader.py TimeXer/data_provider/data_loader.py
cp timexer_overlay/utils/tools.py TimeXer/utils/tools.py
cp timexer_overlay/make_price_only_csv.py TimeXer/make_price_only_csv.py
cp timexer_overlay/summarize_ablation.py TimeXer/summarize_ablation.py
mkdir -p TimeXer/dataset/stock

# ------------------------------------------------------------------ FNSPID
if [ ! -d external/FNSPID/.git ]; then
  git clone https://github.com/Zdong104/FNSPID_Financial_News_Dataset.git external/FNSPID
fi
mkdir -p fnspid_transformer/tst fnspid_transformer/their_weights_5873ff8 fnspid_sentiment external/fnspid_data_eecdcdb

# Transformer package, commit 5873ff8
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/__init__.py > fnspid_transformer/tst/__init__.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/decoder.py > fnspid_transformer/tst/decoder.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/encoder.py > fnspid_transformer/tst/encoder.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/loss.py > fnspid_transformer/tst/loss.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/multiHeadAttention.py > fnspid_transformer/tst/multiHeadAttention.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/positionwiseFeedForward.py > fnspid_transformer/tst/positionwiseFeedForward.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/transformer.py > fnspid_transformer/tst/transformer.py
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/tst/utils.py > fnspid_transformer/tst/utils.py

# Released weights, commit 5873ff8
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/model_saved/Nonsentiment_5_4layers.pt > fnspid_transformer/their_weights_5873ff8/Nonsentiment_5_4layers.pt
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/model_saved/Nonsentiment_25_4layers.pt > fnspid_transformer/their_weights_5873ff8/Nonsentiment_25_4layers.pt
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/model_saved/Nonsentiment_50_4layers.pt > fnspid_transformer/their_weights_5873ff8/Nonsentiment_50_4layers.pt
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/model_saved/Sentiment_5_4layers.pt > fnspid_transformer/their_weights_5873ff8/Sentiment_5_4layers.pt
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/model_saved/Sentiment_25_4layers.pt > fnspid_transformer/their_weights_5873ff8/Sentiment_25_4layers.pt
git -C external/FNSPID show 5873ff8:dataset_test/Transformer-for-Time-Series-Prediction/model_saved/Sentiment_50_4layers.pt > fnspid_transformer/their_weights_5873ff8/Sentiment_50_4layers.pt

# Data files of the five thesis companies (prices and Sentiment_gpt), commit eecdcdb
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/T.csv > fnspid_sentiment/fnspid_T.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/INTC.csv > fnspid_sentiment/fnspid_INTC.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/AMD.csv > fnspid_sentiment/fnspid_AMD.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/CVX.csv > fnspid_sentiment/fnspid_CVX.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/BABA.csv > fnspid_sentiment/fnspid_BABA.csv

# Data files of the five companies of the FNSPID paper, commit eecdcdb
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/KO.csv > external/fnspid_data_eecdcdb/KO.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/AMD.csv > external/fnspid_data_eecdcdb/AMD.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/TSM.csv > external/fnspid_data_eecdcdb/TSM.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/GOOG.csv > external/fnspid_data_eecdcdb/GOOG.csv
git -C external/FNSPID show eecdcdb:dataset_test/Transformer-for-Time-Series-Prediction/data/WMT.csv > external/fnspid_data_eecdcdb/WMT.csv

echo "External code and files are in place."
