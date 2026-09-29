#!/usr/bin/env python3
"""Diebold-Mariano tests on the daily test losses of the one-step experiment.

For each company and pair of forecasters A, B the daily loss differential is
d_t = mean over the five seeds of (e_A,s,t^2 - e_B,s,t^2), t = 1..302 test days,
where seed s of A and seed s of B share the same price-only base model. The
statistic uses a Newey-West (Bartlett) long-run variance with bandwidth
floor(4 (T/100)^(2/9)) = 5, and the Harvey-Leybourne-Newbold small-sample
correction, compared with Student t on T-1 degrees of freedom (two-sided).
The random walk repeats the previous adjusted close and has no seeds.
Also reported: lag-1 autocorrelation of d_t and the Ljung-Box p value at 10 lags,
which show why a HAC variance is needed.
"""
import json, math
import numpy as np, pandas as pd
from scipy import stats

TK = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
NAME = {'T': 'AT&T', 'INTC': 'Intel', 'AMD': 'AMD', 'CVX': 'Chevron', 'BABA': 'Alibaba'}
SEEDS = [11, 22, 33, 44, 55]
TAIL = '_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_'

def sq_err(arm, tk):
    out = []
    for s in SEEDS:
        d = f'TimeXer/results/long_term_forecast_{arm}_{tk}_s{s}{TAIL}{arm}_{tk}_s{s}_0'
        t, p = np.load(d + '/true.npy').reshape(-1), np.load(d + '/pred.npy').reshape(-1)
        out.append((p - t) ** 2)
    return np.array(out), t

def rw_sq_err(tk, true):
    full = pd.read_csv(f'TimeXer/dataset/stock/{tk}_graph.csv')['adj_close'].values
    n = len(full) - 302
    assert np.abs(full[n:] - true).max() < 1e-3
    return (full[n - 1:-1] - full[n:]) ** 2

def dm(d, h=1):
    T = len(d); m = d.mean(); x = d - m
    L = int(math.floor(4 * (T / 100) ** (2 / 9)))
    lrv = x @ x / T
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / T
    stat = m / math.sqrt(lrv / T)
    stat *= math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(stat), T - 1)
    ac1 = float(np.corrcoef(d[1:], d[:-1])[0, 1])
    lb = stats.chi2.sf(T * (T + 2) * sum(np.corrcoef(d[k:], d[:-k])[0, 1] ** 2 / (T - k) for k in range(1, 11)), 10)
    return stat, p, ac1, lb, L

pairs = [('GRAPH', 'PRICE'), ('GRAPH', 'TEXT64'), ('GRAPH', 'TEXT1024'), ('GRAPH', 'SENT'), ('GRAPH', 'RW'), ('PRICE', 'RW'), ('SENT', 'PRICE'),
         ('TEXT64', 'PRICE'), ('TEXT1024', 'PRICE'),
         ('GRAPH_L0', 'PRICE'), ('SENT_L0', 'PRICE'), ('GRAPH_L0', 'GRAPH'), ('SENT_L0', 'SENT')]
res = {}
print(f"{'comparison':18s} {'company':8s} {'mean d':>11s} {'rel %':>7s} {'DM':>7s} {'p':>7s} {'ac1(d)':>7s} {'LB p':>7s}")
for a, b in pairs:
    for tk in TK:
        ea, true = sq_err(a, tk)
        la = ea.mean(0)
        lb_ = rw_sq_err(tk, true) if b == 'RW' else sq_err(b, tk)[0].mean(0)
        d = la - lb_
        stat, p, ac1, lbp, L = dm(d)
        rel = 100 * d.mean() / lb_.mean()
        res[f'{a} vs {b}|{tk}'] = dict(mean_d=float(d.mean()), rel_pct=float(rel), dm=float(stat), p=float(p), ac1=ac1, ljung_box_p=float(lbp), bandwidth=L)
        print(f"{a+' vs '+b:18s} {NAME[tk]:8s} {d.mean():11.5f} {rel:+7.2f} {stat:7.2f} {p:7.4f} {ac1:7.3f} {lbp:7.4f}")
json.dump(res, open('fnspid_transformer/dm_tests_onestep.json', 'w'), indent=1)
