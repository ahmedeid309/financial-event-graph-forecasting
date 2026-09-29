#!/usr/bin/env python3
"""The dataset paper's configuration against the thesis's, on the thesis's ONE-STEP task.

Companion of analyze_fnspid_grid.py (three-day task). Here: 10-day window, the next
trading day's adjusted close, the 302 test days per company of stage 4 (thesis
Table 4.18), five seeds per cell:

                      prices only          + sentiment score        + graph
  FNSPID Transformer  8e  price_only       8e  sentiment            8e  graph
  TimeXer (AG)        4   PRICE            7   SENT                 4   GRAPH

plus the random walk adj_close(t-1) -> adj_close(t) on the same days.

Statistics are those of analyze_fnspid_grid.py: TimeXer variants paired within seed
(they share the price-only parent), everything else unpaired (ratio of seed-mean
MSEs, cross-pairs won out of 25 = Mann-Whitney U with its exact p, Welch's t).
Before anything is reported, the TimeXer and random-walk numbers are checked
against thesis Table 4.18, and every Transformer target against TimeXer's.

Writes fnspid_transformer/grid_results_onestep.json and prints the tables.
"""
import json
import math
import statistics as st

import numpy as np
import pandas as pd
from scipy import stats

TK = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
NAME = {'T': 'AT&T', 'INTC': 'Intel', 'AMD': 'AMD', 'CVX': 'Chevron', 'BABA': 'Alibaba'}
SEEDS = [11, 22, 33, 44, 55]
CELLS = [('TX-P', 'TimeXer', 'prices', 'PRICE'), ('TX-S', 'TimeXer', 'sentiment', 'SENT'),
         ('TX-G', 'TimeXer', 'graph', 'GRAPH'), ('TR-P', 'Transformer', 'prices', 'price_only'),
         ('TR-S', 'Transformer', 'sentiment', 'sentiment'), ('TR-G', 'Transformer', 'graph', 'graph')]
TAIL = '_TimeXer_custom_ftMS_sl10_ll5_pl1_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_'
N_TEST = 302
# Thesis Table 4.18 (27-28 Sep build): R2 and MSE in price units, for the correctness gate.
TABLE_4_18 = {
    'T':    {'rw': (0.9696, 0.07523), 'TX-P': (0.9678, 0.07983), 'TX-G': (0.9691, 0.07668), 'TX-S': (0.9674, 0.08074)},
    'INTC': {'rw': (0.9792, 0.62906), 'TX-P': (0.9753, 0.74981), 'TX-G': (0.9780, 0.66739), 'TX-S': (0.9750, 0.75866)},
    'AMD':  {'rw': (0.9815, 8.29299), 'TX-P': (0.9803, 8.81661), 'TX-G': (0.9832, 7.51195), 'TX-S': (0.9806, 8.71260)},
    'CVX':  {'rw': (0.9328, 5.75049), 'TX-P': (0.9246, 6.45668), 'TX-G': (0.9399, 5.14703), 'TX-S': (0.9271, 6.24133)},
    'BABA': {'rw': (0.9473, 6.47861), 'TX-P': (0.9440, 6.87316), 'TX-G': (0.9460, 6.62904), 'TX-S': (0.9446, 6.80912)},
}


def tx_dir(arm, tk, seed):
    mid = f'{arm}_{tk}_s{seed}'
    return f'TimeXer/results/long_term_forecast_{mid}{TAIL}{mid}_0'


def load(cell, tk, seed):
    code, model, rep, arm = cell
    if model == 'TimeXer':
        d = tx_dir(arm, tk, seed)
        return np.load(d + '/true.npy').reshape(-1), np.load(d + '/pred.npy').reshape(-1)
    p = pd.read_csv(f'fnspid_transformer/results_forecast_onestep/{arm}_s{seed}/{tk}_predictions.csv')
    return p['true'].values, p['pred'].values


def gates(arm, tk):
    out = {}
    path = f'TimeXer/{"result_sent" if arm == "SENT" else "result"}_{tk}.txt'
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f'long_term_forecast_{arm}_{tk}_s'):
            seed = int(line.split('_TimeXer_')[0].rsplit('_s', 1)[1])
            if 'graph_gates:[' in lines[i + 1]:
                out[seed] = float(lines[i + 1].split('graph_gates:[')[1].split(']')[0])
    return out


def score(true, pred):
    err = pred - true
    return dict(mse=float(np.mean(err ** 2)), mae=float(np.mean(np.abs(err))),
                r2=float(1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()))


def paired(a, b):
    pct = [100 * (x / y - 1) for x, y in zip(a, b)]
    d = [x - y for x, y in zip(a, b)]
    sd = st.stdev(d)
    t = st.mean(d) / (sd / math.sqrt(len(d))) if sd > 0 else float('nan')
    return dict(pct=st.mean(pct), pct_of_means=100 * (st.mean(a) / st.mean(b) - 1),
                wins=sum(x < 0 for x in d), n=len(d), t=t,
                p=float(2 * stats.t.sf(abs(t), len(d) - 1)) if sd > 0 else float('nan'))


def unpaired(a, b):
    u = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact')
    w = stats.ttest_ind(a, b, equal_var=False)
    return dict(ratio=st.mean(a) / st.mean(b), pct=100 * (st.mean(a) / st.mean(b) - 1),
                pair_wins=sum(1 for x in a for y in b if x < y), pairs=len(a) * len(b),
                mw_p=float(u.pvalue), welch_t=float(w.statistic), welch_p=float(w.pvalue))


def main():
    res = {'companies': {}, 'pooled': {}}
    for tk in TK:
        full = pd.read_csv(f'TimeXer/dataset/stock/{tk}_graph.csv')['adj_close'].values
        rng = float(full.max() - full.min())
        n_train = len(full) - N_TEST
        rw_true = full[n_train:n_train + N_TEST]
        rw_pred = full[n_train - 1:n_train + N_TEST - 1]
        ref_true, _ = load(CELLS[0], tk, 11)
        assert np.abs(ref_true - rw_true).max() < 1e-3, f'{tk}: TimeXer targets differ from the adjusted close'
        rw = score(rw_true, rw_pred)
        rw.update(mse_mm=rw['mse'] / rng ** 2, mae_mm=rw['mae'] / rng)
        gate = TABLE_4_18[tk]['rw']
        assert abs(rw['r2'] - gate[0]) < 6e-5 and abs(rw['mse'] - gate[1]) < 6e-6, f'{tk}: random walk does not reproduce Table 4.18'
        company = {'range': rng, 'random_walk': rw, 'cells': {}}
        for cell in CELLS:
            runs = {}
            for seed in SEEDS:
                true, pred = load(cell, tk, seed)
                assert len(true) == N_TEST and np.abs(true - ref_true).max() < 1e-3, f'{tk} {cell[0]} s{seed}: targets differ'
                runs[seed] = score(true, pred)
            mse = [runs[s]['mse'] for s in SEEDS]
            r2 = [runs[s]['r2'] for s in SEEDS]
            entry = dict(model=cell[1], input=cell[2], mse_by_seed=mse, r2_by_seed=r2,
                         r2=st.mean(r2), r2_sd=st.stdev(r2), mse=st.mean(mse),
                         mae=st.mean(runs[s]['mae'] for s in SEEDS),
                         mse_mm=st.mean(mse) / rng ** 2, mae_mm=st.mean(runs[s]['mae'] for s in SEEDS) / rng,
                         beats_random_walk=sum(m < rw['mse'] for m in mse))
            if cell[0] in TABLE_4_18[tk]:
                ref = TABLE_4_18[tk][cell[0]]
                assert abs(entry['r2'] - ref[0]) < 6e-5 and abs(entry['mse'] - ref[1]) < 6e-6, \
                    f'{tk} {cell[0]}: does not reproduce Table 4.18 ({entry["r2"]:.4f}, {entry["mse"]:.5f})'
            if cell[1] == 'TimeXer' and cell[2] != 'prices':
                g = gates(cell[3], tk)
                entry['at_base_model'] = sum(1 for s in SEEDS if g.get(s) == 0.0)
            company['cells'][cell[0]] = entry
        m = {k: v['mse_by_seed'] for k, v in company['cells'].items()}
        company['comparisons'] = {
            'TX-S vs TX-P (paired)': paired(m['TX-S'], m['TX-P']),
            'TX-G vs TX-P (paired)': paired(m['TX-G'], m['TX-P']),
            'TX-G vs TX-S (paired)': paired(m['TX-G'], m['TX-S']),
            'TR-S vs TR-P': unpaired(m['TR-S'], m['TR-P']),
            'TR-G vs TR-P': unpaired(m['TR-G'], m['TR-P']),
            'TR-G vs TR-S': unpaired(m['TR-G'], m['TR-S']),
            'TX-P vs TR-P': unpaired(m['TX-P'], m['TR-P']),
            'TX-S vs TR-S': unpaired(m['TX-S'], m['TR-S']),
            'TX-G vs TR-G': unpaired(m['TX-G'], m['TR-G']),
            'TX-G vs TR-S (ours vs theirs)': unpaired(m['TX-G'], m['TR-S']),
        }
        res['companies'][tk] = company

    pooled = {'mean_r2': {'random walk': st.mean(res['companies'][t]['random_walk']['r2'] for t in TK)}, 'comparisons': {}}
    for cell in CELLS:
        pooled['mean_r2'][cell[0]] = st.mean(res['companies'][t]['cells'][cell[0]]['r2'] for t in TK)
    for key in res['companies']['T']['comparisons']:
        rows = [res['companies'][t]['comparisons'][key] for t in TK]
        agg = {'mean_pct': st.mean(r['pct'] for r in rows)}
        if 'wins' in rows[0]:
            agg.update(mean_pct_of_means=st.mean(r['pct_of_means'] for r in rows),
                       wins=sum(r['wins'] for r in rows), n=sum(r['n'] for r in rows))
        else:
            agg.update(pair_wins=sum(r['pair_wins'] for r in rows), pairs=sum(r['pairs'] for r in rows))
        pooled['comparisons'][key] = agg
    res['pooled'] = pooled
    json.dump(res, open('fnspid_transformer/grid_results_onestep.json', 'w'), indent=1)

    print('Correctness gate passed: TimeXer PRICE/GRAPH/SENTIMENT and the random walk reproduce Table 4.18.\n')
    print('R2 (mean of 5 seeds) on the one-step task')
    print(f"{'company':9s} {'RW':>7s} " + ' '.join(f'{c[0]:>7s}' for c in CELLS))
    for tk in TK:
        c = res['companies'][tk]
        print(f"{NAME[tk]:9s} {c['random_walk']['r2']:7.4f} " + ' '.join(f"{c['cells'][x[0]]['r2']:7.4f}" for x in CELLS))
    print(f"{'mean':9s} {pooled['mean_r2']['random walk']:7.4f} " + ' '.join(f"{pooled['mean_r2'][x[0]]:7.4f}" for x in CELLS))
    print('\ncomparisons (MSE of first vs second)')
    for key in res['companies']['T']['comparisons']:
        print(f'  {key}')
        for tk in TK:
            r = res['companies'][tk]['comparisons'][key]
            if 'wins' in r:
                print(f"    {NAME[tk]:8s} {r['pct_of_means']:+8.2f}% (seed means)  wins {r['wins']}/5  t {r['t']:7.2f}  p {r['p']:.4f}")
            else:
                print(f"    {NAME[tk]:8s} x{r['ratio']:6.3f} ({r['pct']:+7.1f}%)  cross-pairs {r['pair_wins']:2d}/25  MW p {r['mw_p']:.4f}")
        print(f"    pooled   {pooled['comparisons'][key]}")
    print('\nruns beating the random walk:', {x[0]: sum(res['companies'][t]['cells'][x[0]]['beats_random_walk'] for t in TK) for x in CELLS})


if __name__ == '__main__':
    main()
