#!/usr/bin/env python3
"""One-step task with PREVIOUS-DAY news only (stage 9), beside the same-day version.

The forecast of day t uses news representations up to day t-1 (join lead 0): each
daily representation covers its own day and the calendar day before, so the news
comes from days t-2 and t-1. Everything else is the one-step task of Table 4.18.

Grid (five seeds per cell, the 302 test days of Table 4.18):
                      prices only             + sentiment (prev. day)   + graph (prev. day)
  TimeXer (AG)        stage 4 PRICE           stage 9 SENT_L0           stage 9 GRAPH_L0
  FNSPID Transformer  stage 8e price_only     stage 9 sentiment         stage 9 graph
Price-only runs use no news, so they are the same runs as in the same-day grid.

Also compares, paired within seed on the same price-only parent, each TimeXer news
variant with previous-day news against the same variant with same-day news.
Statistics and correctness gates as in analyze_fnspid_grid_onestep.py.
Writes fnspid_transformer/grid_results_prevday.json.
"""
import json
import statistics as st

import numpy as np
import pandas as pd

import analyze_fnspid_grid_onestep as A

CELLS = [('TX-P', 'TimeXer', 'prices', 'PRICE'), ('TX-S', 'TimeXer', 'sentiment', 'SENT_L0'),
         ('TX-G', 'TimeXer', 'graph', 'GRAPH_L0'), ('TR-P', 'Transformer', 'prices', 'onestep/price_only'),
         ('TR-S', 'Transformer', 'sentiment', 'prevday/sentiment'), ('TR-G', 'Transformer', 'graph', 'prevday/graph')]


def load(cell, tk, seed):
    code, model, rep, arm = cell
    if model == 'TimeXer':
        d = A.tx_dir(arm, tk, seed)
        return np.load(d + '/true.npy').reshape(-1), np.load(d + '/pred.npy').reshape(-1)
    folder, name = arm.split('/')
    p = pd.read_csv(f'fnspid_transformer/results_forecast_{folder}/{name}_s{seed}/{tk}_predictions.csv')
    return p['true'].values, p['pred'].values


def gates_lead0(arm, tk):
    out = {}
    lines = open(f'TimeXer/result_lead0_{tk}.txt').read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f'long_term_forecast_{arm}_{tk}_s') and 'graph_gates:[' in lines[i + 1]:
            seed = int(line.split('_TimeXer_')[0].rsplit('_s', 1)[1])
            out[seed] = float(lines[i + 1].split('graph_gates:[')[1].split(']')[0])
    return out


def mse_runs(arm, tk):
    return [A.score(*(lambda d: (np.load(d + '/true.npy').reshape(-1), np.load(d + '/pred.npy').reshape(-1)))(A.tx_dir(arm, tk, s)))['mse'] for s in A.SEEDS]


def main():
    res = {'companies': {}, 'pooled': {}}
    for tk in A.TK:
        full = pd.read_csv(f'TimeXer/dataset/stock/{tk}_graph.csv')['adj_close'].values
        rng = float(full.max() - full.min())
        n_train = len(full) - A.N_TEST
        rw = A.score(full[n_train:n_train + A.N_TEST], full[n_train - 1:n_train + A.N_TEST - 1])
        ref = A.TABLE_4_18[tk]['rw']
        assert abs(rw['r2'] - ref[0]) < 6e-5 and abs(rw['mse'] - ref[1]) < 6e-6, f'{tk}: random walk off Table 4.18'
        ref_true, _ = load(CELLS[0], tk, 11)
        company = {'random_walk': rw, 'cells': {}}
        for cell in CELLS:
            runs = {}
            for seed in A.SEEDS:
                true, pred = load(cell, tk, seed)
                assert len(true) == A.N_TEST and np.abs(true - ref_true).max() < 1e-3, f'{tk} {cell[0]} s{seed}: targets differ'
                runs[seed] = A.score(true, pred)
            mse = [runs[s]['mse'] for s in A.SEEDS]; r2 = [runs[s]['r2'] for s in A.SEEDS]
            entry = dict(model=cell[1], input=cell[2], mse_by_seed=mse, r2_by_seed=r2, r2=st.mean(r2),
                         r2_sd=st.stdev(r2), mse=st.mean(mse), mae=st.mean(runs[s]['mae'] for s in A.SEEDS),
                         beats_random_walk=sum(m < rw['mse'] for m in mse))
            if cell[0] == 'TX-P':
                r = A.TABLE_4_18[tk]['TX-P']
                assert abs(entry['r2'] - r[0]) < 6e-5 and abs(entry['mse'] - r[1]) < 6e-6, f'{tk}: PRICE off Table 4.18'
            if cell[0] in ('TX-S', 'TX-G'):
                g = gates_lead0(cell[3], tk)
                entry['at_base_model'] = sum(1 for s in A.SEEDS if g.get(s) == 0.0)
            company['cells'][cell[0]] = entry
        m = {k: v['mse_by_seed'] for k, v in company['cells'].items()}
        same_day = {'GRAPH': mse_runs('GRAPH', tk), 'SENT': mse_runs('SENT', tk)}
        company['comparisons'] = {
            'TX-S vs TX-P (paired)': A.paired(m['TX-S'], m['TX-P']),
            'TX-G vs TX-P (paired)': A.paired(m['TX-G'], m['TX-P']),
            'TX-G vs TX-S (paired)': A.paired(m['TX-G'], m['TX-S']),
            'TR-S vs TR-P': A.unpaired(m['TR-S'], m['TR-P']),
            'TR-G vs TR-P': A.unpaired(m['TR-G'], m['TR-P']),
            'TX-P vs TR-P': A.unpaired(m['TX-P'], m['TR-P']),
            'TX-S vs TR-S': A.unpaired(m['TX-S'], m['TR-S']),
            'TX-G vs TR-G': A.unpaired(m['TX-G'], m['TR-G']),
            'TX-G vs TR-S (ours vs theirs)': A.unpaired(m['TX-G'], m['TR-S']),
            'GRAPH prev-day vs same-day (paired)': A.paired(m['TX-G'], same_day['GRAPH']),
            'SENT prev-day vs same-day (paired)': A.paired(m['TX-S'], same_day['SENT']),
        }
        res['companies'][tk] = company
    pooled = {'mean_r2': {'random walk': st.mean(res['companies'][t]['random_walk']['r2'] for t in A.TK)}, 'comparisons': {}}
    for cell in CELLS:
        pooled['mean_r2'][cell[0]] = st.mean(res['companies'][t]['cells'][cell[0]]['r2'] for t in A.TK)
    for key in res['companies']['T']['comparisons']:
        rows = [res['companies'][t]['comparisons'][key] for t in A.TK]
        agg = {'mean_pct': st.mean(r['pct'] for r in rows)}
        if 'wins' in rows[0]:
            agg.update(mean_pct_of_means=st.mean(r['pct_of_means'] for r in rows), wins=sum(r['wins'] for r in rows))
        else:
            agg.update(pair_wins=sum(r['pair_wins'] for r in rows), pairs=sum(r['pairs'] for r in rows))
        pooled['comparisons'][key] = agg
    pooled['at_base_model'] = {c: sum(res['companies'][t]['cells'][c]['at_base_model'] for t in A.TK) for c in ('TX-S', 'TX-G')}
    res['pooled'] = pooled
    json.dump(res, open('fnspid_transformer/grid_results_prevday.json', 'w'), indent=1)

    print('Correctness gate passed: TimeXer PRICE and the random walk reproduce Table 4.18.\n')
    print('R2 (mean of 5 seeds), one-step task, PREVIOUS-DAY news')
    print(f"{'company':9s} {'RW':>7s} " + ' '.join(f'{c[0]:>7s}' for c in CELLS))
    for tk in A.TK:
        c = res['companies'][tk]
        print(f"{A.NAME[tk]:9s} {c['random_walk']['r2']:7.4f} " + ' '.join(f"{c['cells'][x[0]]['r2']:7.4f}" for x in CELLS))
    print(f"{'mean':9s} {pooled['mean_r2']['random walk']:7.4f} " + ' '.join(f"{pooled['mean_r2'][x[0]]:7.4f}" for x in CELLS))
    print('\ncomparisons (MSE of first vs second)')
    for key in res['companies']['T']['comparisons']:
        print(f'  {key}')
        for tk in A.TK:
            r = res['companies'][tk]['comparisons'][key]
            if 'wins' in r:
                print(f"    {A.NAME[tk]:8s} {r['pct_of_means']:+8.2f}% (seed means)  wins {r['wins']}/5  t {r['t']:7.2f}  p {r['p']:.4f}")
            else:
                print(f"    {A.NAME[tk]:8s} x{r['ratio']:6.3f} ({r['pct']:+7.1f}%)  cross-pairs {r['pair_wins']:2d}/25  MW p {r['mw_p']:.4f}")
        print(f"    pooled   {pooled['comparisons'][key]}")
    print('\nruns beating the random walk:', {x[0]: sum(res['companies'][t]['cells'][x[0]]['beats_random_walk'] for t in A.TK) for x in CELLS})
    print('TimeXer runs retaining their base model:', pooled['at_base_model'])


if __name__ == '__main__':
    main()
