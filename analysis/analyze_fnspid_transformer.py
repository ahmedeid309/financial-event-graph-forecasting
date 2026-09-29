#!/usr/bin/env python3
"""Stage 8: FNSPID's Transformer, three views.

  8a  reproduction check   their code, their five stocks, vs their Table 3
  8b  exact replication    their code and protocol on our five companies
                           (price / their sentiment / our graph)
  8c  genuine forecast     their model and recipe, 50 -> 3 raw close, scored on
                           stage 6's test samples beside TimeXer and the random walk

Run from the project root after 8a, 8b and 8c have finished.
"""
import math
import os
import statistics as st

import numpy as np
import pandas as pd

SEEDS = [11, 22, 33, 44, 55]
OURS = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
THEIRS = ['KO', 'AMD', 'TSM', 'GOOG', 'WMT']
ROOT = 'fnspid_transformer'

# Their committed single runs (commit 5873ff8, test_result_5/*_eval_data.csv): MAE, MSE, R2.
COMMITTED = {
    'nonsentiment': {'KO': (0.028839491, 0.0012890922, 0.7005782128173682),
                     'AMD': (0.015243475, 0.00035703258, 0.9575133687633116),
                     'TSM': (0.014071903, 0.00028929816, 0.9618878022481062),
                     'GOOG': (0.011669976, 0.00014543167, -6.517279071559719),
                     'WMT': (0.017172618, 0.00047138004, 0.8463940831331748)},
    'sentiment': {'KO': (0.018874438, 0.0005849263, 0.864137213159692),
                  'AMD': (0.010594043, 0.0001609329, 0.9808490954730725),
                  'TSM': (0.030010208, 0.00095602305, 0.8740533287152794),
                  'GOOG': (0.00216678, 7.3345445e-06, 0.6208816111416205),
                  'WMT': (0.015240835, 0.0004211287, 0.8627692012868061)},
}
TABLE3 = {'nonsentiment': (0.01883, 0.00060, 0.86659), 'sentiment': (0.01801, 0.00058, 0.87260)}


def sd(values):
    return st.stdev(values) if len(values) > 1 else 0.0


def paired(a, b):
    """% change of mean MSE a vs b, wins (a < b), paired t on the differences."""
    d = [x - y for x, y in zip(a, b)]
    s = sd(d)
    t = st.mean(d) / (s / math.sqrt(len(d))) if s > 0 else float('nan')
    return 100 * (st.mean(a) - st.mean(b)) / st.mean(b), sum(x < 0 for x in d), len(d), t


def eval_rows(results, mode, stock):
    rows = []
    for seed in SEEDS:
        path = f'{ROOT}/{results}/{mode}_s{seed}/{stock}_eval_data.csv'
        if os.path.exists(path):
            rows.append(pd.read_csv(path).iloc[0])
    return rows


def part_8a():
    print('=' * 100)
    print('8a  REPRODUCTION CHECK: their code, their data, their five stocks (scaled close)')
    print('=' * 100)
    for mode in ('nonsentiment', 'sentiment'):
        print(f'\n  {mode}')
        print(f"  {'stock':6s} {'R2 mean':>9s} {'sd':>7s} {'MSE':>10s} {'MAE':>9s} | {'their R2':>9s} {'their MSE':>10s} | "
              f"{'prev-close R2':>13s} {'copy R2':>8s}")
        means = {}
        for stock in THEIRS:
            rows = eval_rows('results_check', mode, stock)
            if not rows:
                continue
            r2 = [r.R2 for r in rows]
            means[stock] = (st.mean(r.MAE for r in rows), st.mean(r.MSE for r in rows), st.mean(r2))
            c = COMMITTED[mode][stock]
            print(f"  {stock:6s} {st.mean(r2):9.4f} {sd(r2):7.4f} {means[stock][1]:10.6f} {means[stock][0]:9.5f} | "
                  f"{c[2]:9.4f} {c[1]:10.6f} | {st.mean(r.prev_R2 for r in rows):13.4f} {st.mean(r.copy_R2 for r in rows):8.4f}"
                  f"   ({len(rows)} seeds)")
        four = [s for s in THEIRS if s != 'GOOG' and s in means]
        if len(four) == 4:
            mae, mse, r2 = (st.mean(means[s][k] for s in four) for k in range(3))
            t = TABLE3[mode]
            print(f"  Table-3 style (mean over KO, AMD, TSM, WMT; GOOG dropped): "
                  f"MAE {mae:.5f} MSE {mse:.5f} R2 {r2:.5f}   | paper: MAE {t[0]:.5f} MSE {t[1]:.5f} R2 {t[2]:.5f}")


def part_8b():
    print('\n' + '=' * 100)
    print('8b  EXACT REPLICATION ON OUR COMPANIES (their protocol; target = close already in the window)')
    print('=' * 100)
    print(f"  {'company':8s} {'arm':13s} {'R2 mean':>9s} {'sd':>7s} {'MSE (mm)':>10s} {'MAE (mm)':>9s} | "
          f"{'prev-close R2':>13s} {'copy R2':>8s}")
    pooled = {}
    for tk in OURS:
        mse = {}
        for mode in ('nonsentiment', 'sentiment', 'graph'):
            rows = eval_rows('results_exact', mode, tk)
            if not rows:
                continue
            r2 = [r.R2 for r in rows]
            mse[mode] = [r.MSE for r in rows]
            pooled.setdefault(mode, []).append(st.mean(r2))
            print(f"  {tk:8s} {mode:13s} {st.mean(r2):9.4f} {sd(r2):7.4f} {st.mean(mse[mode]):10.6f} "
                  f"{st.mean(r.MAE for r in rows):9.5f} | {st.mean(r.prev_R2 for r in rows):13.4f} "
                  f"{st.mean(r.copy_R2 for r in rows):8.4f}")
        for arm in ('sentiment', 'graph'):
            if arm in mse and 'nonsentiment' in mse and len(mse[arm]) == len(mse['nonsentiment']):
                pct, wins, n, t = paired(mse[arm], mse['nonsentiment'])
                print(f"  {'':8s} {arm + ' vs price':22s} MSE {pct:+7.2f}%  better in {wins}/{n}  t {t:6.2f}")
        print()
    for mode, values in pooled.items():
        print(f"  mean R2 over the five companies, {mode:13s} {st.mean(values):.4f}")


def timexer_arrays(arm, tk, seed):
    d = (f'TimeXer/results/long_term_forecast_{arm}_{tk}_s{seed}_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1'
         f'_df32_expand2_dc4_fc3_ebtimeF_dtTrue_{arm}_{tk}_s{seed}_0')
    return np.load(d + '/true.npy').reshape(-1), np.load(d + '/pred.npy').reshape(-1)


def score(true, pred):
    err = pred - true
    return {'mse': float(np.mean(err ** 2)), 'mae': float(np.mean(np.abs(err))),
            'r2': float(1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum())}


def part_8c():
    print('\n' + '=' * 100)
    print('8c  GENUINE 3-DAY FORECAST: their Transformer vs TimeXer, identical test samples (raw close)')
    print('=' * 100)
    print(f"  {'company':8s} {'model':26s} {'R2 mean':>9s} {'sd':>7s} {'MSE (price)':>12s} {'MSE (mm)':>10s}")
    summary = {}
    for tk in OURS:
        frame = pd.read_csv(f'TimeXer/dataset/stock/{tk}_graph_h3.csv')
        close = frame['close'].values
        rng = close.max() - close.min()
        n_train = len(frame) - 302
        rw_true, rw_pred = [], []
        for i in range(300):
            b = n_train + i
            rw_true += list(close[b:b + 3]); rw_pred += [close[b - 1]] * 3
        rw = score(np.array(rw_true), np.array(rw_pred))
        print(f"  {tk:8s} {'random walk':26s} {rw['r2']:9.4f} {'--':>7s} {rw['mse']:12.4f} {rw['mse'] / rng ** 2:10.6f}")
        summary.setdefault('random walk', []).append(rw['r2'])
        mses = {}
        for label, kind, arm in (('TimeXer price-only', 'tx', 'PRICE_M'), ('TimeXer graph', 'tx', 'GRAPH_M'),
                                 ('FNSPID Transformer price', 'tr', 'price_only'),
                                 ('FNSPID Transformer graph', 'tr', 'graph')):
            runs = []
            for seed in SEEDS:
                if kind == 'tx':
                    true, pred = timexer_arrays(arm, tk, seed)
                else:
                    path = f'{ROOT}/results_forecast/{arm}_s{seed}/{tk}_predictions.csv'
                    if not os.path.exists(path):
                        continue
                    p = pd.read_csv(path)
                    true, pred = p['true'].values, p['pred'].values
                    tx_true, _ = timexer_arrays('PRICE_M', tk, seed)
                    assert np.abs(true - tx_true).max() < 1e-3, f'{tk} {arm} s{seed}: targets differ from stage 6'
                runs.append(score(true, pred))
            if not runs:
                continue
            r2 = [r['r2'] for r in runs]
            mses[label] = [r['mse'] for r in runs]
            summary.setdefault(label, []).append(st.mean(r2))
            print(f"  {tk:8s} {label:26s} {st.mean(r2):9.4f} {sd(r2):7.4f} {st.mean(mses[label]):12.4f} "
                  f"{st.mean(mses[label]) / rng ** 2:10.6f}   (runs beating random walk: "
                  f"{sum(m < rw['mse'] for m in mses[label])}/{len(runs)})")
        if 'FNSPID Transformer graph' in mses and 'FNSPID Transformer price' in mses:
            pct, wins, n, t = paired(mses['FNSPID Transformer graph'], mses['FNSPID Transformer price'])
            print(f"  {'':8s} Transformer: graph vs price    MSE {pct:+7.2f}%  better in {wins}/{n}  t {t:6.2f}")
        for tr, tx in (('FNSPID Transformer price', 'TimeXer price-only'), ('FNSPID Transformer graph', 'TimeXer graph')):
            if tr in mses:
                ratio = st.mean(mses[tr]) / st.mean(mses[tx])
                print(f"  {'':8s} {tr[19:]:5s} Transformer vs TimeXer: MSE x{ratio:.2f}  "
                      f"(best Transformer run {min(mses[tr]):.4f}, worst TimeXer run {max(mses[tx]):.4f})")
        print()
    print('  MEAN R2 over the five companies')
    for label, values in summary.items():
        print(f"    {label:26s} {st.mean(values):.4f}   range [{min(values):.4f}, {max(values):.4f}]")


if __name__ == '__main__':
    part_8a()
    part_8b()
    part_8c()
