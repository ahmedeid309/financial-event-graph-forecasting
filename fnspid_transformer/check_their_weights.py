#!/usr/bin/env python3
"""Score FNSPID's own trained Transformer weights with this port's data path.

Their repository (commit 5873ff8, the version committed with Table 3's results)
contains the trained models model_saved/{Nonsentiment,Sentiment}_{5,25,50}_4layers.pt.
Scoring each on WMT with run_exact.py's data path reproduces the committed WMT
evaluation file of the same scope and mode (result of 28 Sep 2026: all six agree to
six significant digits; the largest relative difference is 5.95e-07, at the
rounding level of 32-bit arithmetic). That verifies that the port prepares,
windows, scales, feeds and scores the data exactly as their code does, and that
their weights load into the copied model package with every parameter name and
shape matching (strict loading).

Only WMT can be checked this way: the committed weights are the state saved after a
scope's training pass, and the outputs of the other evaluated stocks were produced
by intermediate states that were not stored. (In the released 50-stock list one
more stock follows WMT; the committed 50-stock weights nevertheless reproduce the
committed WMT output, so the saved state is the one that produced it.)

  python check_their_weights.py --weights_dir their_weights_5873ff8
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import run_exact

# Committed evaluation files test_result_<scope>/WMT_<mode>_*/..._eval_data.csv at 5873ff8: MAE, MSE, R2.
COMMITTED_WMT = {
    (5, 'Nonsentiment'): (0.017172618, 0.00047138004, 0.8463940831331748),
    (5, 'Sentiment'): (0.015240835, 0.0004211287, 0.8627692012868061),
    (25, 'Nonsentiment'): (0.006432822, 8.406329e-05, 0.9726067776722045),
    (25, 'Sentiment'): (0.011777276, 0.00017209111, 0.943921657918252),
    (50, 'Nonsentiment'): (0.0061903833, 7.314656e-05, 0.9761641505590287),
    (50, 'Sentiment'): (0.0037065835, 3.5340665e-05, 0.9884837392532265),
}
COLUMNS = {'Nonsentiment': ['Volume', 'Open', 'Close'],
           'Sentiment': ['Volume', 'Open', 'Close', 'Scaled_sentiment']}


def evaluate(model, cols, stock):
    data = pd.read_csv(f'data_theirs/{stock}.csv')[cols].values
    _, test, _ = run_exact.data_processor(data)
    pred, true = [], []
    with torch.no_grad():
        for x, y in test:
            pred.append(model(x).numpy())
            true.append(y.numpy())
    pred, true = np.concatenate(pred).ravel(), np.concatenate(true).ravel()
    return mean_absolute_error(true, pred), mean_squared_error(true, pred), r2_score(true, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', required=True)
    args = parser.parse_args()
    run_exact.device = torch.device('cpu')
    for scope in (5, 25, 50):
        for mode in ('Nonsentiment', 'Sentiment'):
            cols = COLUMNS[mode]
            model = run_exact.Transformer(len(cols), 32, 1, 8, 8, 8, 4, attention_size=30, dropout=0.1,
                                          chunk_mode=None, pe='regular')
            state = torch.load(f'{args.weights_dir}/{mode}_{scope}_4layers.pt', map_location='cpu')
            model.load_state_dict(state, strict=True)   # raises on any missing or unexpected parameter
            model.eval()
            mae, mse, r2 = evaluate(model, cols, 'WMT')
            ref = COMMITTED_WMT[(scope, mode)]
            match = abs(mae - ref[0]) < 1e-8 and abs(mse - ref[1]) < 1e-10 and abs(r2 - ref[2]) < 1e-6
            note = 'MATCH' if match else 'MISMATCH'
            print(f'scope {scope:2d} {mode:12s} WMT  ours: MAE {mae:.9f} MSE {mse:.11f} R2 {r2:.10f}')
            print(f'{"":21s}committed: MAE {ref[0]:.9f} MSE {ref[1]:.11f} R2 {ref[2]:.10f}  -> {note}')


if __name__ == '__main__':
    main()
