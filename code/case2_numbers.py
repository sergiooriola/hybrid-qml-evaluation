r"""Emit every number the Case Study II chapter and Annex D quote.

Single source of truth for the thesis tables. Run after any change to the
grid and diff the output against the .tex files:

    python case2_numbers.py > ../thesis/_numbers.txt

Sections of the output map one-to-one onto thesis tables:

    [D.1]  tab:case2-friedman-omnibus     Friedman omnibus per (eta, r)
    [D.2]  tab:case2-holm-sig-count       Holm-significant pairs per (eta, r)
    [5.1]  tab:case2-grid                 mean BA per model per ratio
    [5.4]  tab:case2-robustness           balanced vs extreme, slope
    [D.3]  tab:case2-auc-contrast         BA vs AUC at the two extremes
    [aux]  degradation split, win counts, tie audit, per-model drops
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qml_imbalance import TIE_DECIMALS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, 'results', 'case2_grid.json')

# Analysis target. Defaults to the archived grid; pass --grid PATH (or set
# CASE2_GRID) to point the same analysis at a grid you regenerated yourself.
if "--grid" in sys.argv:
    GRID = sys.argv[sys.argv.index("--grid") + 1]
elif os.environ.get("CASE2_GRID"):
    GRID = os.environ["CASE2_GRID"]


D = json.load(open(GRID))
RUNS = D['runs']
C = D['config']
RATIOS = C['ratios']
NOISES = C['noise_levels']
QUANTUM = list(C['quantum_models'])
CLASSICAL = list(C['classical_models'])
ALL_MODELS = QUANTUM + CLASSICAL


def cell(noise, ratio, metric='balanced_accuracy'):
    """(models, matrix) with matrix[seed, model] for one (eta, r) cell."""
    sub = [r for r in RUNS if r['noise'] == noise and r['ratio'] == ratio]
    models = sorted({r['model'] for r in sub})
    seeds = sorted({r['seed'] for r in sub})
    M = np.full((len(seeds), len(models)), np.nan)
    idx = {(r['model'], r['seed']): r for r in sub}
    for j, m in enumerate(models):
        for i, s in enumerate(seeds):
            M[i, j] = idx[(m, s)]['metrics'][metric]
    return models, M


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise(M, models, round_diffs=True):
    pairs = list(itertools.combinations(range(len(models)), 2))
    ps = []
    for a, b in pairs:
        d = M[:, a] - M[:, b]
        if round_diffs:
            d = np.round(d, TIE_DECIMALS)
        if np.all(d == 0):
            ps.append(1.0)
        else:
            try:
                ps.append(wilcoxon(d, zero_method='wilcox',
                                   alternative='two-sided')[1])
            except ValueError:
                ps.append(1.0)
    return pairs, np.asarray(ps), holm(ps)


def mean_ba(noise, ratio, model, metric='balanced_accuracy'):
    v = [r['metrics'][metric] for r in RUNS
         if r['noise'] == noise and r['ratio'] == ratio and r['model'] == model]
    return float(np.mean(v))


# -----------------------------------------------------------------------
print('=' * 72)
print('[D.1] tab:case2-friedman-omnibus   +   [D.2] tab:case2-holm-sig-count')
print('=' * 72)
sig = {}
for r in RATIOS:
    row = [f'    ${r:.2f}$']
    counts = []
    for nz in NOISES:
        models, M = cell(nz, r)
        chi2, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
        _, _, adj = pairwise(M, models)
        n = int((adj < 0.05).sum())
        counts.append(n)
        sig[(nz, r)] = n
        mant, exp = f'{p:.1e}'.split('e')
        row.append(f'${chi2:.2f}$ & $ {mant} \\times 10^{{{int(exp)}}}$')
    print(' & '.join([row[0], row[1], '', row[2]]) + r' \\')
print()
print('  D.2 body:')
for r in RATIOS:
    print(f'    ${r:.2f}$ & ${sig[(NOISES[0], r)]}$ & '
          f'${sig[(NOISES[1], r)]}$ \\\\')

# -----------------------------------------------------------------------
print()
print('=' * 72)
print('[5.1] tab:case2-grid   mean BA per model per ratio')
print('=' * 72)
for nz in NOISES:
    print(f'\n  eta = {nz}')
    print('    model'.ljust(22) + ''.join(f'{r:>8.2f}' for r in RATIOS))
    for m in ALL_MODELS:
        vals = [mean_ba(nz, r, m) for r in RATIOS]
        print('    ' + m.ljust(18) + ''.join(f'{v:>8.3f}' for v in vals))

# -----------------------------------------------------------------------
print()
print('=' * 72)
print('[5.4] tab:case2-robustness   BA at r=0.50, r=0.99, slope')
print('=' * 72)
for nz in NOISES:
    print(f'\n  eta = {nz}')
    rows = []
    for m in ALL_MODELS:
        a, b = mean_ba(nz, 0.50, m), mean_ba(nz, 0.99, m)
        rows.append((a - b, m, a, b))
    for drop, m, a, b in sorted(rows):
        kind = 'Q' if m in QUANTUM else 'C'
        print(f'    {kind} {m:<18} {a:.3f}  {b:.3f}   slope {drop:.3f}')

# -----------------------------------------------------------------------
print()
print('=' * 72)
print('DEGRADATION SPLIT   (how much of the total drop sits above r=0.90)')
print('=' * 72)
for nz in NOISES:
    fr = []
    for m in ALL_MODELS:
        tot = mean_ba(nz, 0.50, m) - mean_ba(nz, 0.99, m)
        top = mean_ba(nz, 0.90, m) - mean_ba(nz, 0.99, m)
        if tot > 0.01:
            fr.append(top / tot)
    print(f'  eta = {nz}: mean {100*np.mean(fr):.1f}%  '
          f'median {100*np.median(fr):.1f}%  over {len(fr)} models')

# -----------------------------------------------------------------------
print()
print('=' * 72)
print('CERTIFIED QUANTUM WINS  (Holm p<0.05 and mean diff favouring quantum)')
print('=' * 72)
total_q, total_c = 0, 0
for nz in NOISES:
    for r in RATIOS:
        models, M = cell(nz, r)
        pairs, _, adj = pairwise(M, models)
        for k, (a, b) in enumerate(pairs):
            ma, mb = models[a], models[b]
            if (ma in QUANTUM) == (mb in QUANTUM) or adj[k] >= 0.05:
                continue
            d = M[:, a].mean() - M[:, b].mean()
            qwin = (d > 0) if ma in QUANTUM else (d < 0)
            q, c = (ma, mb) if ma in QUANTUM else (mb, ma)
            if qwin:
                total_q += 1
                print(f'  eta={nz} r={r:.2f}  {q} > {c}   '
                      f'p_holm={adj[k]:.4f}')
            else:
                total_c += 1
print(f'\n  quantum-favouring certified pairs : {total_q}')
print(f'  classical-favouring certified pairs: {total_c}')

# -----------------------------------------------------------------------
print()
print('=' * 72)
print('TIE-POLICY AUDIT   (rounded differences vs raw arrays)')
print('=' * 72)
spurious = 0
flips = []
npairs = 0
for nz in NOISES:
    for r in RATIOS:
        models, M = cell(nz, r)
        pairs, _, adj_new = pairwise(M, models, round_diffs=True)
        _, _, adj_old = pairwise(M, models, round_diffs=False)
        npairs += len(pairs)
        for k, (a, b) in enumerate(pairs):
            d = M[:, a] - M[:, b]
            spurious += int(((d != 0) & (np.abs(d) < 1e-9)).sum())
            if (adj_new[k] < 0.05) != (adj_old[k] < 0.05):
                mixed = (models[a] in QUANTUM) != (models[b] in QUANTUM)
                flips.append((nz, r, models[a], models[b],
                              adj_old[k], adj_new[k], mixed))
print(f'  spurious non-zero differences (|d| < 1e-9): {spurious}')
print(f'  pairs tested                             : {npairs}')
print(f'  significance flips                       : {len(flips)}')
for f in flips:
    tag = 'QUANTUM-vs-CLASSICAL' if f[6] else 'same family'
    print(f'    eta={f[0]} r={f[1]:.2f}  {f[2]} vs {f[3]}: '
          f'{f[4]:.6f} -> {f[5]:.6f}  [{tag}]')

# -----------------------------------------------------------------------
print()
print('=' * 72)
print('DEGENERATE-PREDICTOR FLAG  (pre-registered at BA < 0.55)')
print('=' * 72)
worst = min(((mean_ba(nz, r, m), m, nz, r)
             for nz in NOISES for r in RATIOS for m in ALL_MODELS))
print(f'  lowest cell mean: {worst[0]:.3f}  ({worst[1]}, eta={worst[2]}, '
      f'r={worst[3]})')
below = [(mean_ba(nz, r, m), m, nz, r)
         for nz in NOISES for r in RATIOS for m in ALL_MODELS
         if mean_ba(nz, r, m) < 0.55]
print(f'  cells below 0.55: {len(below)}')
for v, m, nz, r in sorted(below):
    print(f'    {m} eta={nz} r={r:.2f}  {v:.3f}')
