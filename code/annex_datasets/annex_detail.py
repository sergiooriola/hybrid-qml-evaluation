r"""Detail behind the annex advantage table: which quantum wins, and where.

Prints, per dataset and noise level, the certified quantum wins grouped by
the classical baseline they are against, and flags every cell in which the
classical model of the pair is itself below the pre-registered
degenerate-predictor threshold of 0.55 balanced accuracy.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qml_imbalance import TIE_DECIMALS  # noqa: E402

DATASETS = ["sinus", "corners", "spiral", "circles"]
STRONG = {"svm_rbf", "rf"}
FLAG = 0.55


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


for name in DATASETS:
    D = json.load(open(os.path.join(HERE, "results", f"annex_{name}.json")))
    R, C = D["runs"], D["config"]
    Q = set(C["quantum_models"])
    for nz in C["noise_levels"]:
        wins_by_c = {}
        strong_detail = []
        degenerate_cells = []
        qw = cw = 0
        for r in C["ratios"]:
            sub = [x for x in R if x["noise"] == nz and x["ratio"] == r]
            models = sorted({x["model"] for x in sub})
            seeds = sorted({x["seed"] for x in sub})
            idx = {(x["model"], x["seed"]): x for x in sub}
            M = np.array([[idx[(m, s)]["metrics"]["balanced_accuracy"]
                           for m in models] for s in seeds])
            means = {m: M[:, j].mean() for j, m in enumerate(models)}
            low = [m for m in models if m not in Q and means[m] < FLAG]
            if low:
                degenerate_cells.append((r, {m: round(means[m], 3)
                                             for m in low}))
            pairs = list(itertools.combinations(range(len(models)), 2))
            ps = []
            for a, b in pairs:
                d = np.round(M[:, a] - M[:, b], TIE_DECIMALS)
                ps.append(1.0 if np.all(d == 0)
                          else wilcoxon(d, zero_method="wilcox")[1])
            adj = holm(ps)
            for k, (a, b) in enumerate(pairs):
                ma, mb = models[a], models[b]
                if (ma in Q) == (mb in Q) or adj[k] >= 0.05:
                    continue
                diff = M[:, a].mean() - M[:, b].mean()
                qwin = (diff > 0) if ma in Q else (diff < 0)
                q, c = (ma, mb) if ma in Q else (mb, ma)
                if qwin:
                    qw += 1
                    wins_by_c[c] = wins_by_c.get(c, 0) + 1
                    if c in STRONG:
                        strong_detail.append(
                            (r, q, c, round(means[q], 3), round(means[c], 3),
                             round(means[q] - means[c], 3), round(adj[k], 4)))
                else:
                    cw += 1
        print(f"\n### {name}  eta={nz}   Q wins {qw}   C wins {cw}")
        print("    quantum wins by baseline:",
              {k: v for k, v in sorted(wins_by_c.items(),
                                       key=lambda kv: -kv[1])})
        if strong_detail:
            print("    against the strong tier:")
            for r, q, c, bq, bc, d, p in strong_detail:
                print(f"      r={r:.2f}  {q} > {c}   {bq} vs {bc} "
                      f"(+{d})  p_holm={p}")
        if degenerate_cells:
            print("    classical baselines below the 0.55 flag:")
            for r, d in degenerate_cells:
                print(f"      r={r:.2f}  {d}")
