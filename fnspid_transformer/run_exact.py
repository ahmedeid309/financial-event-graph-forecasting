#!/usr/bin/env python3
"""FNSPID's Transformer experiment, as it produced Table 3 of the dataset paper.

Source: github.com/Zdong104/FNSPID_Financial_News_Dataset, commit 5873ff8
(2024-02-19), dataset_test/Transformer-for-Time-Series-Prediction/run.py. The
per-stock evaluation files behind Table 3 (dated 3 and 18 February 2024) were
committed with this version. The model package tst/ is copied from the same
commit and is identical to the repository's current version.

Kept exactly as in that file: min-max scaling fitted on the whole series, the
chronological 85/15 split, 50-row windows with stride 1, the target, the model
(4 encoder + 4 decoder layers, d_model 32, q = v = 8, 8 heads, attention_size 30,
dropout 0.1, regular positional encoding, sigmoid output read at the last time
step), MSE loss, SGD with lr 1e-3 and momentum 0.9, batch 64 without shuffling,
one pass over the stocks in list order with the model carried from one stock to
the next, and evaluation on the scaled close.

Changed (each marked CHANGED below):
  * the input columns and stock lists come from the command line, so one script
    serves every arm; the original hardcodes a sentiment and a no-sentiment pair.
  * epochs. The committed file trains 1 epoch per invocation and relies on
    reloading the saved model across repeated invocations, a total that cannot
    be recovered from the repository. The paper states 100 epochs; that is used.
  * a seed is set and each run writes its own model file. The original loads any
    model file already on disk, so separate runs would silently continue from
    one another.
  * plotting is removed. It does not affect any number, and the original fails
    when plot_saved/ does not exist.
  * each evaluation also records two references on the same targets (below).

THE TARGET. create_sequences sets y to the close of the LAST row of the input
window, data[i + input_length - 1, 2], and the model returns its output at that
same time step. The value being "predicted" is therefore part of the input. Two
references are written next to every result so this is measured, not asserted:
  copy_last_input   the close already in the window's last row (the target)
  previous_close    the close of the row before it (a one-day persistence rule)
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tst import Transformer


def create_sequences(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length):
        X.append(data[i:(i + input_length)])
        y.append(data[i + input_length - 1,
                 2:3])  # 2 is the index of 'Close' in input_features 2:3 to make the shape as (data_length,1)
    X = np.array(X)
    y = np.array(y)
    return X, y


def data_processor(data):
    # Scaling the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Creating sequences
    input_length = 50
    output_length = 3

    # Split training data into training and validation sets
    split_ratio = 0.85
    split = int(split_ratio * len(scaled_data))
    data_train = scaled_data[:split]
    data_test = scaled_data[split:]

    X_train, y_train = create_sequences(data_train, input_length, output_length)
    X_test, y_test = create_sequences(data_test, input_length, output_length)

    print('X_train: ', X_train.shape, 'X_test', X_test.shape, 'y_train', y_train.shape, 'y_test', y_test.shape)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 64
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_test, scaler


def train_model(dataloader_train, symbol, model_path, d_input, epochs):
    # Model parameters
    d_output = 1
    d_model = 32  # Lattent dim
    q = 8  # Query size
    v = 8  # Value size
    h = 8  # Number of heads
    N = 4  # Number of encoder and decoder to stack
    attention_size = 30  # Attention window size
    dropout = 0.1  # Dropout rate
    pe = 'regular'  # Positional encoding
    chunk_mode = None

    model = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout,
                        chunk_mode=chunk_mode, pe=pe).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for idx_epoch in range(epochs):  # CHANGED: epochs from the command line (paper: 100)
        running_loss = 0
        with tqdm(total=len(dataloader_train.dataset), desc=f"[{symbol} epoch {idx_epoch + 1:3d}/{epochs}]",
                  mininterval=30) as pbar:
            for idx_batch, (x, y) in enumerate(dataloader_train):
                optimizer.zero_grad()
                y_pred = model(x.to(device))
                loss = loss_function(y.to(device), y_pred)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.update(x.shape[0])
            pbar.set_postfix({'loss': running_loss / len(dataloader_train)})

    torch.save(model.state_dict(), model_path)
    print(f"saved model into {model_path}")
    return model


def metrics(true, pred):
    return mean_absolute_error(true, pred), mean_squared_error(true, pred), r2_score(true, pred)


def eval_model(data, model, dataloader_test, symbol, mode, out_dir):
    scaler = MinMaxScaler()
    scaler.fit_transform(data)
    predictions, actuals, inputs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader_test:
            modelout = model(x.to(device))
            predictions.append(modelout.cpu().numpy())
            actuals.append(y.cpu().numpy())
            inputs.append(x.cpu().numpy())
    predictions_np = np.concatenate(predictions, axis=0)
    actuals_np = np.concatenate(actuals, axis=0)
    inputs_np = np.concatenate(inputs, axis=0)
    y_pred_reshaped = predictions_np.reshape(actuals_np.shape)
    y_test_flattened = actuals_np.flatten()
    y_pred_flattened = y_pred_reshaped.flatten()

    mae, mse, r2 = metrics(y_test_flattened, y_pred_flattened)
    print(f"{symbol} {mode}: MSE: {mse}, MAE: {mae}, R^2: {r2}")

    # CHANGED: the two references on the same scaled targets.
    copy_last = inputs_np[:, -1, 2]
    prev_close = inputs_np[:, -2, 2]
    copy_mae, copy_mse, copy_r2 = metrics(y_test_flattened, copy_last)
    prev_mae, prev_mse, prev_r2 = metrics(y_test_flattened, prev_close)
    print(f"{symbol} reference copy_last_input: MSE {copy_mse:.3e} R^2 {copy_r2:.6f} | "
          f"previous_close: MSE {prev_mse:.3e} R^2 {prev_r2:.6f}")

    # Origin-scale values as the original writes them; its zero-padded array had a
    # hardcoded width of 4 or 3, generalised here to the input width. CHANGED.
    y_test_expanded = np.zeros((y_test_flattened.shape[0], data.shape[1]))
    y_pred_expanded = np.zeros((y_pred_flattened.shape[0], data.shape[1]))
    y_test_expanded[:, 2] = y_test_flattened
    y_pred_expanded[:, 2] = y_pred_flattened
    y_test_origin = scaler.inverse_transform(y_test_expanded)[:, 2]
    y_pred_origin = scaler.inverse_transform(y_pred_expanded)[:, 2]

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'True_Data': y_test_flattened, 'Predicted_Data': y_pred_flattened,
                  'True_Data_origin': y_test_origin, 'Predicted_Data_origin': y_pred_origin}
                 ).to_csv(os.path.join(out_dir, f'{symbol}_predicted_data.csv'), index=False)
    pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'R2': [r2],
                  'copy_MAE': [copy_mae], 'copy_MSE': [copy_mse], 'copy_R2': [copy_r2],
                  'prev_MAE': [prev_mae], 'prev_MSE': [prev_mse], 'prev_R2': [prev_r2],
                  'n_test': [len(y_test_flattened)]}
                 ).to_csv(os.path.join(out_dir, f'{symbol}_eval_data.csv'), index=False)


def main():
    global device
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--stocks', required=True, help='training order, comma separated')
    parser.add_argument('--eval_stocks', required=True)
    parser.add_argument('--columns', required=True, help="base columns; 'Close' must be third")
    parser.add_argument('--graph_dims', type=int, default=0, help='append gnn_0..gnn_{n-1}')
    parser.add_argument('--mode', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    # CHANGED: seeding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device =', device)

    columns = args.columns.split(',') + [f'gnn_{i}' for i in range(args.graph_dims)]
    if columns[2] != 'Close':
        raise SystemExit("the original code reads the target from column index 2; 'Close' must be third")
    run_dir = os.path.join(args.out_dir, f'{args.mode}_s{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    # CHANGED: one model file per run, which must not exist yet.
    model_path = os.path.join(run_dir, 'model.pt')
    if os.path.exists(model_path):
        raise SystemExit(f'{model_path} already exists; this run would continue a previous one')

    eval_names = args.eval_stocks.split(',')
    for name in args.stocks.split(','):
        csv_data = pd.read_csv(os.path.join(args.data_dir, f'{name}.csv'))
        data = csv_data[columns].values
        print(name, 'columns', len(columns), 'rows', len(data))
        dataloader_train, dataloader_test, scaler = data_processor(data)
        model = train_model(dataloader_train, name, model_path, len(columns), args.epochs)
        if name in eval_names:
            eval_model(data, model, dataloader_test, name, args.mode, run_dir)


if __name__ == "__main__":
    main()
