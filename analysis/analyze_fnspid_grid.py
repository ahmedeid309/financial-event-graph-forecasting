#!/usr/bin/env python3
"""The dataset paper's configuration against the thesis's, on one common task.

Three-day task (50-day window, raw close of the next three trading days), the 300
test samples per company of stage 6, five seeds per cell:

                      prices only          + sentiment score        + graph
  FNSPID Transformer  8c  price_only       8d  sentiment            8c  graph
  TimeXer (AG)        6   PRICE_M          8d  SENT_M               6   GRAPH_M

plus the random walk, close(t+h) = close(t), on the same samples.

Statistics follow Section 3.8.3 of the thesis where runs are paired:
  * TimeXer news variants share their price-only parent within a seed, so they are
    compared by the mean of per-seed percentage changes (Eq. 3.32), wins, and a
    two-sided paired t on the raw MSE differences (Eq. 3.33, df = 4).
Runs of different forecasters are not paired (no shared parent, and the
Transformer's initialisation changes with its input width), so those comparisons
use the ratio of seed-mean MSEs, the number of cross-pairs won out of 25
(the Mann-Whitney U statistic) with its exact two-sided p value, and Welch's t.
The same unpaired statistics are used between Transformer inputs.

Every target array is checked against stage 6's before anything is reported.
Writes fnspid_transformer/grid_results.json and prints the tables.
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
CELLS = [('TX-P', 'TimeXer', 'prices', 'PRICE_M'), ('TX-S', 'TimeXer', 'sentiment', 'SENT_M'),
         ('TX-G', 'TimeXer', 'graph', 'GRAPH_M'), ('TR-P', 'Transformer', 'prices', 'price_only'),
         ('TR-S', 'Transformer', 'sentiment', 'sentiment'), ('TR-G', 'Transformer', 'graph', 'graph')]
TAIL = '_TimeXer_custom_ftMS_sl50_ll25_pl3_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_'


def tx_dir(arm, tk, seed):
    mid = f'{arm}_{tk}_s{seed}'
    return f'TimeXer/results/long_term_forecast_{mid}{TAIL}{mid}_0'


def load(cell, tk, seed):
    code, model, rep, arm = cell
    if model == 'TimeXer':
        d = tx_dir(arm, tk, seed)
        return np.load(d + '/true.npy').reshape(-1), np.load(d + '/pred.npy').reshape(-1)
    p = pd.read_csv(f'fnspid_transformer/results_forecast/{arm}_s{seed}/{tk}_predictions.csv')
    return p['true'].values, p['pred'].values


def gates(arm, tk):
    out = {}
    lines = open(f'TimeXer/{"result_matched_sent" if arm == "SENT_M" else "result_matched"}_{tk}.txt').read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f'long_term_forecast_{arm}_{tk}_s'):
            seed = int(line.split('_TimeXer_')[0].rsplit('_s', 1)[1])
            metric = lines[i + 1]
            if 'graph_gates:[' in metric:
                out[seed] = float(metric.split('graph_gates:[')[1].split(']')[0])
    return out


def score(true, pred):
    err = pred - true
    return dict(mse=float(np.mean(err ** 2)), mae=float(np.mean(np.abs(err))),
                r2=float(1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()))


def paired(a, b):
    """a vs b, same seeds: mean per-seed % change (Eq. 3.32), wins, paired t (Eq. 3.33)."""
    pct = [100 * (x / y - 1) for x, y in zip(a, b)]
    d = [x - y for x, y in zip(a, b)]
    sd = st.stdev(d)
    t = st.mean(d) / (sd / math.sqrt(len(d))) if sd > 0 else float('nan')
    return dict(pct=st.mean(pct), pct_of_means=100 * (st.mean(a) / st.mean(b) - 1),
                wins=sum(x < 0 for x in d), n=len(d), t=t,
                p=float(2 * stats.t.sf(abs(t), len(d) - 1)) if sd > 0 else float('nan'))


def unpaired(a, b):
    """a vs b, independent runs: ratio of means, cross-pairs won by a, exact Mann-Whitney p, Welch t."""
    u = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact')
    wins = sum(1 for x in a for y in b if x < y)
    w = stats.ttest_ind(a, b, equal_var=False)
    return dict(ratio=st.mean(a) / st.mean(b), pct=100 * (st.mean(a) / st.mean(b) - 1),
                pair_wins=wins, pairs=len(a) * len(b), mw_p=float(u.pvalue),
                welch_t=float(w.statistic), welch_p=float(w.pvalue))


def main():
    res = {'companies': {}, 'pooled': {}}
    for tk in TK:
        full = pd.read_csv(f'TimeXer/dataset/stock/{tk}_graph.csv')['close'].values
        rng = float(full.max() - full.min())
        n_train = len(full) - 302
        rw_true = np.concatenate([full[n_train + i:n_train + i + 3] for i in range(300)])
        rw_pred = np.concatenate([[full[n_train + i - 1]] * 3 for i in range(300)])
        ref_true, _ = load(CELLS[0], tk, 11)
        assert np.abs(ref_true - rw_true).max() < 1e-3, f'{tk}: stage-6 targets differ from the raw close'
        company = {'range': rng, 'random_walk': score(rw_true, rw_pred), 'cells': {}}
        company['random_walk'].update(mse_mm=company['random_walk']['mse'] / rng ** 2,
                                      mae_mm=company['random_walk']['mae'] / rng)
        for cell in CELLS:
            runs = {}
            for seed in SEEDS:
                true, pred = load(cell, tk, seed)
                assert len(true) == 900 and np.abs(true - ref_true).max() < 1e-3, f'{tk} {cell[0]} s{seed}: targets differ'
                runs[seed] = score(true, pred)
            mse = [runs[s]['mse'] for s in SEEDS]
            r2 = [runs[s]['r2'] for s in SEEDS]
            entry = dict(model=cell[1], input=cell[2], mse_by_seed=mse, r2_by_seed=r2,
                         r2=st.mean(r2), r2_sd=st.stdev(r2), mse=st.mean(mse),
                         mae=st.mean(runs[s]['mae'] for s in SEEDS),
                         mse_mm=st.mean(mse) / rng ** 2, mae_mm=st.mean(runs[s]['mae'] for s in SEEDS) / rng,
                         beats_random_walk=sum(m < company['random_walk']['mse'] for m in mse))
            if cell[1] == 'TimeXer' and cell[2] != 'prices':
                g = gates(cell[3], tk)
                entry['gates'] = [g[s] for s in SEEDS]
                entry['at_base_model'] = sum(1 for s in SEEDS if g[s] == 0.0)
            company['cells'][cell[0]] = entry
        c = company['cells']
        m = {k: v['mse_by_seed'] for k, v in c.items()}
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

    pooled = {'mean_r2': {}, 'comparisons': {}}
    pooled['mean_r2']['random walk'] = st.mean(res['companies'][t]['random_walk']['r2'] for t in TK)
    for cell in CELLS:
        pooled['mean_r2'][cell[0]] = st.mean(res['companies'][t]['cells'][cell[0]]['r2'] for t in TK)
    for key in res['companies']['T']['comparisons']:
        rows = [res['companies'][t]['comparisons'][key] for t in TK]
        agg = {'mean_pct': st.mean(r['pct'] for r in rows)}
        if 'wins' in rows[0]:
            agg.update(mean_pct_of_means=st.mean(r['pct_of_means'] for r in rows),
                       wins=sum(r['wins'] for r in rows), n=sum(r['n'] for r in rows))
        else:
            agg.update(pair_wins=sum(r['pair_wins'] for r in rows), pairs=sum(r['pairs'] for r in rows),
                       companies_all_25=sum(r['pair_wins'] == 25 for r in rows),
                       companies_none=sum(r['pair_wins'] == 0 for r in rows))
        pooled['comparisons'][key] = agg
    for code in ('TX-S', 'TX-G'):
        pooled[f'{code} at base model'] = sum(res['companies'][t]['cells'][code]['at_base_model'] for t in TK)
    res['pooled'] = pooled
    json.dump(res, open('fnspid_transformer/grid_results.json', 'w'), indent=1)

    # ------------------------------------------------------------------ print
    print('R2 (mean of 5 seeds) on the three-day task; MSE in price units')
    print(f"{'company':9s} {'RW':>7s} " + ' '.join(f'{c[0]:>7s}' for c in CELLS))
    for tk in TK:
        c = res['companies'][tk]
        print(f"{NAME[tk]:9s} {c['random_walk']['r2']:7.4f} " + ' '.join(f"{c['cells'][x[0]]['r2']:7.4f}" for x in CELLS))
    print(f"{'mean':9s} {pooled['mean_r2']['random walk']:7.4f} " + ' '.join(f"{pooled['mean_r2'][x[0]]:7.4f}" for x in CELLS))
    print('\nfull metrics')
    for tk in TK:
        c = res['companies'][tk]; rw = c['random_walk']
        print(f"  {NAME[tk]:8s} random walk      R2 {rw['r2']:.4f}            MSE {rw['mse']:10.5f} MAE {rw['mae']:7.4f} | mm MSE {rw['mse_mm']:.6f} MAE {rw['mae_mm']:.5f}")
        for x in CELLS:
            e = c['cells'][x[0]]
            extra = f"  base-model {e['at_base_model']}/5" if 'at_base_model' in e else ''
            print(f"  {'':8s} {x[1][:11]:11s} {x[2]:9s} R2 {e['r2']:.4f} sd {e['r2_sd']:.4f} MSE {e['mse']:10.5f} MAE {e['mae']:7.4f} | "
                  f"mm MSE {e['mse_mm']:.6f} MAE {e['mae_mm']:.5f}  beats RW {e['beats_random_walk']}/5{extra}")
    print('\ncomparisons (MSE of first vs second)')
    for key in res['companies']['T']['comparisons']:
        print(f'  {key}')
        for tk in TK:
            r = res['companies'][tk]['comparisons'][key]
            if 'wins' in r:
                print(f"    {NAME[tk]:8s} {r['pct']:+8.2f}% per-seed mean, {r['pct_of_means']:+7.2f}% of seed means  "
                      f"wins {r['wins']}/5  paired t {r['t']:7.2f}  p {r['p']:.4f}")
            else:
                print(f"    {NAME[tk]:8s} x{r['ratio']:5.2f} ({r['pct']:+7.1f}%)  cross-pairs won {r['pair_wins']:2d}/25  "
                      f"MW p {r['mw_p']:.4f}  Welch t {r['welch_t']:7.2f} p {r['welch_p']:.4f}")
        print(f"    pooled   {pooled['comparisons'][key]}")
    print(f"\nTimeXer runs ending at their base model: sentiment {pooled['TX-S at base model']}/25, graph {pooled['TX-G at base model']}/25")


if __name__ == '__main__':
    main()
