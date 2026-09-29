#!/usr/bin/env python3
"""LaTeX tables for thesis Section 4.6.4, generated from grid_results.json.

Run from the project root after analyze_fnspid_grid.py. Writes
fnspid_transformer/latex_tables.tex. No number in the tables is typed by hand.
"""
import json

TK = ['T', 'INTC', 'AMD', 'CVX', 'BABA']
NAME = {'T': 'AT\\&T', 'INTC': 'Intel', 'AMD': 'AMD', 'CVX': 'Chevron', 'BABA': 'Alibaba'}
r = json.load(open('fnspid_transformer/grid_results.json'))
C = r['companies']
P = r['pooled']
out = []
w = out.append


def pct(x, digits=2):
    s = f'{x:+.{digits}f}'.replace('-', '$-$')
    return s


def bold(s, on):
    return f'\\textbf{{{s}}}' if on else s


# ------------------------------------------------------------- R2 grid
w('% ---------------------------------------------------------------- Table: R2 grid')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{Coefficient of determination on the three-day task for both forecasters and all three news inputs, '
  'mean over five seeds. All variants are evaluated on the same 300 test windows per company. The random walk '
  'repeats the last observed close for all three days. $^{\\dagger}$Configuration of the FNSPID study. '
  '$^{\\ddagger}$Configuration of this thesis.}')
w('  \\label{tab:fnspid-grid-r2}')
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

# -------------------------------------------------- forecaster comparison
w('% ---------------------------------------------------------------- Table: forecaster comparison')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{TimeXer against the FNSPID Transformer on the three-day task. Each entry is the ratio of seed-mean '
  'MSE, TimeXer variant divided by Transformer variant, so values below one favor TimeXer. In parentheses, the '
  'number of the 25 cross-pairs of runs in which the TimeXer run has the lower MSE. With 25 of 25, the exact '
  'two-sided Mann--Whitney $p$ value is 0.0079, the smallest attainable with five runs per variant. Bold marks '
  'differences significant at the two-sided 5\\% level.}')
w('  \\label{tab:fnspid-grid-forecaster}')
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

# ------------------------------------------------------- input effects
w('% ---------------------------------------------------------------- Table: input effects')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{Effect of the news input within each forecaster on the three-day task, as the percentage change '
  'in seed-mean MSE of the first-named input relative to the second; negative values favor the first-named '
  'input. TimeXer variants share their base model and are paired within seed (wins out of five, as in '
  'Table~\\ref{tab:REPLACE-4-21}); Transformer variants are not paired (cross-pairs won out of 25). The change of '
  '\\textsc{Graph} against \\textsc{Price} within TimeXer is reported in Table~\\ref{tab:REPLACE-4-21}. Bold marks '
  'differences significant at the two-sided 5\\% level (paired $t$ for TimeXer, exact Mann--Whitney for the '
  'Transformer).}')
w('  \\label{tab:fnspid-grid-inputs}')
w('  \\begin{tabular}{lcccc}')
w('    \\toprule')
w('     & \\multicolumn{2}{c}{TimeXer} & \\multicolumn{2}{c}{FNSPID Transformer} \\\\')
w('    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}')
w('     & \\textsc{Sentiment} & \\textsc{Graph} & \\textsc{Sentiment} & \\textsc{Graph} \\\\')
w('    Company & vs.\\ \\textsc{Price} & vs.\\ \\textsc{Sentiment} & vs.\\ \\textsc{Price} & vs.\\ \\textsc{Price} \\\\')
w('    \\midrule')
for tk in TK:
    a = C[tk]['comparisons']['TX-S vs TX-P (paired)']
    b = C[tk]['comparisons']['TX-G vs TX-S (paired)']
    c = C[tk]['comparisons']['TR-S vs TR-P']
    d = C[tk]['comparisons']['TR-G vs TR-P']
    cells = [bold(f"{pct(a['pct_of_means'])}\\% ({a['wins']}/5)", a['p'] < 0.05),
             bold(f"{pct(b['pct_of_means'])}\\% ({b['wins']}/5)", b['p'] < 0.05),
             bold(f"{pct(c['pct'], 1)}\\% ({c['pair_wins']}/25)", c['mw_p'] < 0.05),
             bold(f"{pct(d['pct'], 1)}\\% ({d['pair_wins']}/25)", d['mw_p'] < 0.05)]
    w(f'    {NAME[tk]} & ' + ' & '.join(cells) + ' \\\\')
w('    \\midrule')
pa = P['comparisons']['TX-S vs TX-P (paired)']; pb = P['comparisons']['TX-G vs TX-S (paired)']
pc = P['comparisons']['TR-S vs TR-P']; pd_ = P['comparisons']['TR-G vs TR-P']
w(f"    Mean & {pct(pa['mean_pct_of_means'])}\\% ({pa['wins']}/25) & {pct(pb['mean_pct_of_means'])}\\% ({pb['wins']}/25) "
  f"& {pct(pc['mean_pct'], 1)}\\% ({pc['pair_wins']}/125) & {pct(pd_['mean_pct'], 1)}\\% ({pd_['pair_wins']}/125) \\\\")
w('    \\bottomrule')
w('  \\end{tabular}')
w('\\end{table}')
w('')

# ---------------------------------------------- optional full-metrics table
w('% ---------------------------------------------------------------- Optional appendix table: full metrics')
w('\\begin{table}[htbp]')
w('  \\centering')
w('  \\small')
w('  \\caption{Full metrics of the variants added in Section~\\ref{sec:results-fnspid-rerun}, in the format of '
  'Table~\\ref{tab:REPLACE-4-20}. SD is the standard deviation of $R^2$ across five seeds.}')
w('  \\label{tab:fnspid-grid-full}')
w('  \\begin{tabular}{llcccccc}')
w('    \\toprule')
w('     & & & & \\multicolumn{2}{c}{Price units} & \\multicolumn{2}{c}{Min--max scaled} \\\\')
w('    \\cmidrule(lr){5-6}\\cmidrule(lr){7-8}')
w('    Company & Variant & $R^2$ & SD & MSE & MAE & MSE & MAE \\\\')
w('    \\midrule')
labels = [('TX-S', 'TimeXer, \\textsc{Sentiment}'), ('TR-P', 'Transformer, \\textsc{Price}'),
          ('TR-S', 'Transformer, \\textsc{Sentiment}'), ('TR-G', 'Transformer, \\textsc{Graph}')]
for i, tk in enumerate(TK):
    for j, (k, lab) in enumerate(labels):
        e = C[tk]['cells'][k]
        first = NAME[tk] if j == 0 else ''
        w(f"    {first} & {lab} & {e['r2']:.4f} & {e['r2_sd']:.4f} & {e['mse']:.5f} & {e['mae']:.4f} & "
          f"{e['mse_mm']:.6f} & {e['mae_mm']:.5f} \\\\")
    if i < len(TK) - 1:
        w('    \\addlinespace')
w('    \\bottomrule')
w('  \\end{tabular}')
w('\\end{table}')

open('fnspid_transformer/latex_tables.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
