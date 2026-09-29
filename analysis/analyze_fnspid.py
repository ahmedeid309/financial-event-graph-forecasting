#!/usr/bin/env python3
"""Recompute this thesis's forecasting results in the dataset paper's metric set.

Two protocols:
  ONE-STEP  stage 4  : 10-day window, 1 day ahead, adjusted close   (thesis design)
  MATCHED   stage 6  : 50-day window, 3 days ahead, raw close       (paper's task)

Metrics: R^2 (invariant to affine rescaling, so directly comparable), MSE and MAE
in price units, and MSE/MAE after min-max scaling to [0,1] as the paper reports
them. A random-walk baseline is computed for each protocol on its own target
series: adj(t-1)->adj(t) for one-step, close(t)->close(t+3) for matched.
"""
import ast, csv, os, struct, statistics as st

TICK = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
SEEDS = [11, 22, 33, 44, 55]
NAME = {'T': 'AT&T', 'INTC': 'Intel', 'AMD': 'AMD', 'CVX': 'Chevron', 'BABA': 'Alibaba'}
RES = 'TimeXer/results'

def read_npy(path):
    with open(path, 'rb') as f:
        assert f.read(6) == b'\x93NUMPY'
        maj = f.read(1)[0]; f.read(1)
        hl = struct.unpack('<H' if maj == 1 else '<I', f.read(2 if maj == 1 else 4))[0]
        h = ast.literal_eval(f.read(hl).decode('latin1').strip())
        n = 1
        for d in h['shape']:
            n *= d
        return list(struct.unpack('<%df' % n, f.read(4 * n)))

def setting(mid, sl, ll, pl):
    return (f'long_term_forecast_{mid}_TimeXer_custom_ftMS_sl{sl}_ll{ll}_pl{pl}'
            f'_dm32_nh4_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_{mid}_0')

def stats(true, pred):
    n = len(true); mu = sum(true) / n
    ssr = sum((a - b) ** 2 for a, b in zip(true, pred))
    sst = sum((a - mu) ** 2 for a in true)
    return dict(mse=ssr / n, mae=sum(abs(a - b) for a, b in zip(true, pred)) / n,
                r2=1 - ssr / sst)

def load_series(tk, col):
    rows = list(csv.DictReader(open(f'TimeXer/dataset/stock/{tk}_graph.csv')))
    return [float(r[col]) for r in rows]

def protocol(label, arms, sl, ll, pl, target_col):
    """Metrics for one protocol.

    The random walk is reconstructed on the model's own sample structure rather
    than on a raw series tail. With pred_len > 1 the stored arrays are
    (samples, pred_len, 1) and flatten to samples*pred_len values in which each
    date recurs, so a naive tail slice would both use the wrong window and lose
    the multi-step overlap. For sample i the targets are full[b : b+pl] where
    b = border1 + i + sl, and the persistence prediction is full[b-1] repeated.
    """
    print('\n' + '=' * 94)
    print(f'{label}   window {sl} -> {pl} day(s) ahead, target = {target_col}')
    print('=' * 94)
    print(f"{'company':9s} {'arm':12s} {'R2':>8s} {'sd':>7s} | {'MSE(price)':>11s} {'MAE(price)':>10s} | "
          f"{'MSE(mm)':>10s} {'MAE(mm)':>9s}")
    agg = {}
    for tk in TICK:
        full = load_series(tk, target_col)
        rng = max(full) - min(full)
        n = len(full); num_test = 302
        border1 = n - num_test - sl
        nsamp = num_test - pl + 1
        true_rw, pred_rw = [], []
        for i in range(nsamp):
            b = border1 + i + sl
            true_rw += full[b:b + pl]
            pred_rw += [full[b - 1]] * pl
        rw = stats(true_rw, pred_rw)
        agg.setdefault('random walk', []).append(rw['r2'])
        print(f"{tk:9s} {'random walk':12s} {rw['r2']:8.4f} {'--':>7s} | {rw['mse']:11.5f} "
              f"{rw['mae']:10.4f} | {rw['mse']/rng**2:10.6f} {rw['mae']/rng:9.5f}")
        for a in arms:
            runs = []
            for sd in SEEDS:
                d = os.path.join(RES, setting(f'{a}_{tk}_s{sd}', sl, ll, pl))
                if os.path.isdir(d):
                    tr = read_npy(d + '/true.npy'); pr = read_npy(d + '/pred.npy')
                    assert len(tr) == nsamp * pl, (tk, a, len(tr), nsamp * pl)
                    runs.append(stats(tr, pr))
            if not runs:
                continue
            r2s = [x['r2'] for x in runs]
            agg.setdefault(a, []).append(st.mean(r2s))
            print(f"{'':9s} {a:12s} {st.mean(r2s):8.4f} {st.stdev(r2s) if len(r2s)>1 else 0:7.4f} | "
                  f"{st.mean(x['mse'] for x in runs):11.5f} {st.mean(x['mae'] for x in runs):10.4f} | "
                  f"{st.mean(x['mse'] for x in runs)/rng**2:10.6f} "
                  f"{st.mean(x['mae'] for x in runs)/rng:9.5f}")
        print()
    print(f"  MEAN R2 over the five companies")
    for k, v in agg.items():
        print(f"    {k:14s} {st.mean(v):.4f}   range [{min(v):.4f}, {max(v):.4f}]")
    return agg

one = protocol('ONE-STEP PROTOCOL (this thesis, stage 4)',
               ['PRICE', 'GRAPH', 'TEXT64', 'TEXT1024'], 10, 5, 1, 'adj_close')
mat = protocol('MATCHED PROTOCOL (dataset paper task, stage 6)',
               ['PRICE_M', 'GRAPH_M'], 50, 25, 3, 'close')

print('\n' + '=' * 94)
print('DATASET PAPER, Table 3 (50 stocks, ChatGPT-sentiment column), for reference')
print('=' * 94)
print(f"  {'Transformer':14s} R2 0.98785   MSE 0.00005   MAE 0.00544")
print(f"  {'LSTM':14s} R2 0.85585   MSE 0.00170   MAE 0.02493")
print(f"  {'GRU':14s} R2 0.82767   MSE 0.00209   MAE 0.02769")
print(f"  {'TimesNet':14s} R2 0.73819   MSE 0.00106   MAE 0.02577")
print(f"  {'CNN':14s} R2 0.73355   MSE 0.00289   MAE 0.03550")
print(f"  {'RNN':14s} R2 0.61744   MSE 0.00389   MAE 0.04154")
print('\n  Their MSE/MAE are on min-max scaled prices, so compare against the MSE(mm)/MAE(mm)')
print('  columns above, and their R2 against the R2 column. No naive baseline is reported')
print('  in that work, which is why the random-walk row is included here.')
