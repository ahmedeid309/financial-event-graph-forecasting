#!/usr/bin/env python3
"""LaTeX tables for the one-step comparison with previous-day news only.

Generated from fnspid_transformer/grid_results_prevday.json (analysis/analyze_fnspid_grid_prevday.py).
Writes fnspid_transformer/latex_tables_prevday.tex. No number is typed by hand.
"""
import json

TK = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
NAME = {'T': 'AT\\&T', 'INTC': 'Intel', 'AMD': 'AMD', 'CVX': 'Chevron', 'BABA': 'Alibaba'}
r = json.load(open('fnspid_transformer/grid_results_prevday.json'))
C, P = r['companies'], r['pooled']
out = []
w = out.append


def bold(s, on):
    return f'\\textbf{{{s}}}' if on else s


def signed(x, digits=2):
    if round(x, digits) == 0:
        return f'{0:.{digits}f}'
    return f'{x:+.{digits}f}'.replace('-', '$-$').replace('+', '$+$')


w('% ---------------------------------------------------------------- Table: previous-day R2 grid')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{Coefficient of determination on the one-step task with previous-day news only, for both '
  'forecasters and all three inputs, mean over five seeds. The task and test days are those of '
  'Table~\\ref{tab:fnspid-grid-r2-onestep}, but the news representation on the last observed row is that of '
  'the last observed day, so the forecast of day $t$ uses news from days $t-2$ and $t-1$ only. The random walk '
  'and the price-only columns use no news and are those of Table~\\ref{tab:fnspid-grid-r2-onestep}. '
  '$^{\\dagger}$Configuration of the FNSPID study. $^{\\ddagger}$Configuration of this thesis.}')
w('  \\label{tab:fnspid-grid-r2-prevday}')
w('  \\begin{tabular}{lccccccc}')
w('    \\toprule')
w('     & & \\multicolumn{3}{c}{TimeXer} & \\multicolumn{3}{c}{FNSPID Transformer} \\\\')
w('    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}')
w('    Company & Random walk & \\textsc{Price} & \\textsc{Sentiment} & \\textsc{Graph}$^{\\ddagger}$ '
  '& \\textsc{Price} & \\textsc{Sentiment}$^{\\dagger}$ & \\textsc{Graph} \\\\')
w('    \\midrule')
for tk in TK:
    c = C[tk]['cells']
    vals = [C[tk]['random_walk']['r2']] + [c[k]['r2'] for k in ('TX-P', 'TX-S', 'TX-G', 'TR-P', 'TR-S', 'TR-G')]
    w(f'    {NAME[tk]} & ' + ' & '.join(f'{v:.4f}' for v in vals) + ' \\\\')
w('    \\midrule')
m = P['mean_r2']
w('    Mean & ' + ' & '.join(f'{m[k]:.4f}' for k in ('random walk', 'TX-P', 'TX-S', 'TX-G', 'TR-P', 'TR-S', 'TR-G')) + ' \\\\')
w('    \\bottomrule')
w('  \\end{tabular}')
w('\\end{table}')
w('')

w('% ---------------------------------------------------------------- Table: previous-day forecaster comparison')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{TimeXer against the FNSPID Transformer on the one-step task with previous-day news only, in the '
  'format of Table~\\ref{tab:fnspid-grid-forecaster-onestep}. Each entry is the ratio of seed-mean MSE, TimeXer '
  'variant divided by Transformer variant, so values below one favor TimeXer. Parentheses give the number of the '
  '25 cross-pairs of runs in which the TimeXer run has the lower MSE. Bold marks differences significant at the '
  'two-sided 5\\% level by the exact Mann--Whitney test.}')
w('  \\label{tab:fnspid-grid-forecaster-prevday}')
w('  \\begin{tabular}{lcccc}')
w('    \\toprule')
w('     & This thesis vs.\\ FNSPID & \\multicolumn{3}{c}{Same news input in both forecasters} \\\\')
w('    \\cmidrule(lr){2-2}\\cmidrule(lr){3-5}')
w('     & TimeXer + \\textsc{Graph} vs.\\ & & & \\\\')
w('    Company & Transformer + \\textsc{Sentiment} & \\textsc{Price} & \\textsc{Sentiment} & \\textsc{Graph} \\\\')
w('    \\midrule')
keys = ['TX-G vs TR-S (ours vs theirs)', 'TX-P vs TR-P', 'TX-S vs TR-S', 'TX-G vs TR-G']
for tk in TK:
    cells = []
    for k in keys:
        x = C[tk]['comparisons'][k]
        cells.append(bold(f"{x['ratio']:.2f} ({x['pair_wins']})", x['mw_p'] < 0.05))
    w(f'    {NAME[tk]} & ' + ' & '.join(cells) + ' \\\\')
w('    \\midrule')
w('    Cross-pairs won & ' + ' & '.join(f"{P['comparisons'][k]['pair_wins']}/{P['comparisons'][k]['pairs']}" for k in keys) + ' \\\\')
w('    \\bottomrule')
w('  \\end{tabular}')
w('\\end{table}')
w('')

w('% ---------------------------------------------------------------- Table: previous-day vs target-day news in TimeXer')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{TimeXer with previous-day news, paired within seed. The first two column groups compare '
  '\\textsc{Graph} and \\textsc{Sentiment} with previous-day news against \\textsc{Price}; the last compares '
  '\\textsc{Graph} with previous-day news against \\textsc{Graph} with target-day news. $\\Delta$ is the MSE change '
  'computed from the seed-mean MSEs, wins count the seeds on which the first variant has the lower MSE, and '
  'bold marks $|t| > 2.78$, the unadjusted two-sided 5\\% threshold across the five training seeds.}')
w('  \\label{tab:prevday-paired}')
w('  \\begin{tabular}{lrcrcrc}')
w('    \\toprule')
w('     & \\multicolumn{2}{c}{\\textsc{Graph} vs.\\ \\textsc{Price}} & \\multicolumn{2}{c}{\\textsc{Sentiment} vs.\\ '
  '\\textsc{Price}} & \\multicolumn{2}{c}{Previous vs.\\ target day} \\\\')
w('    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}')
w('    Company & $\\Delta$ (\\%) & Wins & $\\Delta$ (\\%) & Wins & $\\Delta$ (\\%) & Wins \\\\')
w('    \\midrule')
pk = ['TX-G vs TX-P (paired)', 'TX-S vs TX-P (paired)', 'GRAPH prev-day vs same-day (paired)']
for tk in TK:
    cells = []
    for k in pk:
        x = C[tk]['comparisons'][k]
        cells.append(bold(signed(x['pct_of_means']), abs(x['t']) > 2.78) + f" & {x['wins']}/5")
    w(f'    {NAME[tk]} & ' + ' & '.join(cells) + ' \\\\')
w('    \\midrule')
pooled = [P['comparisons'][k] for k in pk]
w('    Pooled & ' + ' & '.join(f"{signed(x['mean_pct_of_means'])} & {x['wins']}/25" for x in pooled) + ' \\\\')
w('    \\bottomrule')
w('  \\end{tabular}')
w('\\end{table}')

open('fnspid_transformer/latex_tables_prevday.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
