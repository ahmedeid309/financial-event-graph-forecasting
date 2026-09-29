#!/usr/bin/env python3
"""FNSPID's Transformer on a genuine three-day forecast, scored on stage 6's samples.

run_exact.py reproduces the dataset paper's experiment, whose target is a value
already inside the input window. This script keeps the paper's model and training
recipe and changes only what is needed for a forecast that can be compared with
TimeXer on identical test samples.

UNCHANGED from run_exact.py: the tst package and every model hyperparameter
(4 encoder + 4 decoder layers, d_model 32, q = v = 8, 8 heads, attention_size 30,
dropout 0.1, regular positional encoding, sigmoid output read at the last time
step), min-max scaling, 50-row input windows with stride 1, MSE loss, SGD with
lr 1e-3 and momentum 0.9, batch 64 without shuffling, 100 epochs, and one pass
over the five stocks in list order with the model carried from one stock to the
next, each stock evaluated right after its own pass. Inputs are the paper's
feature set, volume, open and close, plus the news block.

CHANGED, and why:
  target   the closes of the three rows AFTER the window, the task the paper
           describes ("50 days of information and predicted 3 days in the
           future") and the one their May-2025 revision (commit 94e477b) adopts.
           The model therefore outputs 3 values instead of 1.
  scaling  min-max fitted on training rows only, as in their May-2025 revision.
           Fitting on the whole series gives the model the test period's price
           range in advance.
  split    the thesis's shared split. Their recipe has no validation step, so
           every row up to VAL_END is a training row; the 302 rows after it are
           the test period, the same rows TimeXer is tested on.
  samples  test windows are built the way TimeXer's loader builds them: sample i
           reads rows [b-50, b) and is scored on the closes of rows b, b+1, b+2,
           with b = first test row + i. That gives the same 300 x 3 targets as
           stage 6, which the analysis script verifies element by element.
  data     stage 6's joined files (TimeXer/dataset/stock/<TK>_{graph,price_only}_h3.csv):
           the repaired prices, the raw close as target, and the graph embedding
           of the last day being predicted on the window's final row, exactly as
           stage 6 fed TimeXer.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tst import Transformer

# Defaults reproduce the three-day task of stage 6 / 8c / 8d. The one-step task of the
# thesis (stage 4) is run with --input_length 10 --output_length 1 --target adj_close.
INPUT_LENGTH = 50
OUTPUT_LENGTH = 3


def build_model(d_input):
    return Transformer(d_input, 32, OUTPUT_LENGTH, 8, 8, 8, 4, attention_size=30, dropout=0.1,
                       chunk_mode=None, pe='regular').to(device)


def main():
    global device, INPUT_LENGTH, OUTPUT_LENGTH
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--suffix', required=True, help='file suffix, e.g. _graph_h3')
    parser.add_argument('--stocks', required=True, help='training order, comma separated')
    parser.add_argument('--columns', required=True, help='base columns; the target must be third')
    parser.add_argument('--target', default='close', help='price column to forecast; must be the third column')
    parser.add_argument('--input_length', type=int, default=INPUT_LENGTH)
    parser.add_argument('--output_length', type=int, default=OUTPUT_LENGTH)
    parser.add_argument('--graph_dims', type=int, default=0)
    parser.add_argument('--val_end', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device =', device)

    columns = args.columns.split(',') + [f'gnn_{i}' for i in range(args.graph_dims)]
    if columns[2] != args.target:
        raise SystemExit(f"'{args.target}' must be the third column, as in the original feature order")
    INPUT_LENGTH, OUTPUT_LENGTH = args.input_length, args.output_length
    run_dir = os.path.join(args.out_dir, f'{args.mode}_s{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, 'model.pt')
    if os.path.exists(model_path):
        raise SystemExit(f'{model_path} already exists; this run would continue a previous one')

    for name in args.stocks.split(','):
        frame = pd.read_csv(os.path.join(args.data_dir, f'{name}{args.suffix}.csv'))
        data = frame[columns].values.astype(np.float64)
        n_train = int((frame['date'] <= args.val_end).sum())
        n_test = len(frame) - n_train

        scaler = MinMaxScaler().fit(data[:n_train])
        scaled = scaler.transform(data)
        close_min, close_max = scaler.data_min_[2], scaler.data_max_[2]

        X_train, y_train = [], []
        for i in range(n_train - INPUT_LENGTH - OUTPUT_LENGTH + 1):
            X_train.append(scaled[i:i + INPUT_LENGTH])
            y_train.append(scaled[i + INPUT_LENGTH:i + INPUT_LENGTH + OUTPUT_LENGTH, 2])
        X_test, y_test_raw, dates = [], [], []
        for i in range(n_test - OUTPUT_LENGTH + 1):
            b = n_train + i
            X_test.append(scaled[b - INPUT_LENGTH:b])
            y_test_raw.append(data[b:b + OUTPUT_LENGTH, 2])
            dates.append(frame['date'].iloc[b])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test_raw = np.array(X_test), np.array(y_test_raw)
        print(f"{name}: {len(columns)} columns, train rows {n_train} -> {len(X_train)} windows, "
              f"test rows {n_test} -> {len(X_test)} samples")

        loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                          torch.tensor(y_train, dtype=torch.float32).to(device)),
                            batch_size=64, shuffle=False)

        model = build_model(len(columns))
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()
        for epoch in range(args.epochs):
            running = 0.0
            with tqdm(total=len(loader.dataset), desc=f"[{name} epoch {epoch + 1:3d}/{args.epochs}]",
                      mininterval=30) as pbar:
                for x, y in loader:
                    optimizer.zero_grad()
                    loss = loss_function(y, model(x))
                    loss.backward()
                    optimizer.step()
                    running += loss.item()
                    pbar.update(x.shape[0])
                pbar.set_postfix({'loss': running / len(loader)})
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        pred_raw = pred_scaled * (close_max - close_min) + close_min
        above = float((y_test_raw > close_max).mean())
        rows = []
        for i, day in enumerate(dates):
            for h in range(OUTPUT_LENGTH):
                rows.append({'sample': i, 'first_target_date': day, 'step': h + 1,
                             'true': y_test_raw[i, h], 'pred': pred_raw[i, h]})
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, f'{name}_predictions.csv'), index=False)
        err = pred_raw - y_test_raw
        sst = ((y_test_raw - y_test_raw.mean()) ** 2).sum()
        print(f"{name} {args.mode}: MSE {np.mean(err ** 2):.5f}  R^2 {1 - (err ** 2).sum() / sst:.4f}  "
              f"(share of test closes above the training maximum: {above:.1%})")


if __name__ == "__main__":
    main()
