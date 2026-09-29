#!/usr/bin/env python3
"""LaTeX tables for the one-step version of the Section 4.6.4 comparison.

Generated from fnspid_transformer/grid_results_onestep.json (analyze_fnspid_grid_onestep.py).
Writes fnspid_transformer/latex_tables_onestep.tex. No number is typed by hand.
"""
import json

TK = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
NAME = {'T': 'AT\\&T', 'INTC': 'Intel', 'AMD': 'AMD', 'CVX': 'Chevron', 'BABA': 'Alibaba'}
r = json.load(open('fnspid_transformer/grid_results_onestep.json'))
C, P = r['companies'], r['pooled']
out = []
w = out.append


def bold(s, on):
    return f'\\textbf{{{s}}}' if on else s


w('% ---------------------------------------------------------------- Table: one-step R2 grid')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{Coefficient of determination on the one-step task for both forecasters and all three news '
  'inputs, mean over five seeds. The task is that of Table~\\ref{tab:REPLACE-4-18}: a 10-day window, the '
  'next trading day\'s adjusted close as target, and the same 302 test days per company. The random walk '
  'repeats the preceding day\'s adjusted close. The random walk and the TimeXer columns are those of '
  'Table~\\ref{tab:REPLACE-4-18}. $^{\\dagger}$Configuration of the FNSPID study. '
  '$^{\\ddagger}$Configuration of this thesis.}')
w('  \\label{tab:fnspid-grid-r2-onestep}')
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

w('% ---------------------------------------------------------------- Table: one-step forecaster comparison')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{TimeXer against the FNSPID Transformer on the one-step task, in the format of '
  'Table~\\ref{tab:fnspid-grid-forecaster}. Each entry is the ratio of seed-mean MSE, TimeXer variant divided '
  'by Transformer variant, so values below one favor TimeXer. Parentheses give the number of the 25 '
  'cross-pairs of runs in which the TimeXer run has the lower MSE. Bold marks differences significant at the '
  'two-sided 5\\% level by the exact Mann--Whitney test.}')
w('  \\label{tab:fnspid-grid-forecaster-onestep}')
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

open('fnspid_transformer/latex_tables_onestep.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
