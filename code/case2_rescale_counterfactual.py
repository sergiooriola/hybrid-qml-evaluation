"""What the quantum-only rescaling was worth, counted in certified wins.

`case2_rescale_ablation.py` measures the asymmetry in balanced accuracy. This
script answers the question that matters for the thesis's conclusion: how many
certified quantum wins the study would have reported had the affine rescaler
stayed on the quantum arm only.

Both arms are taken under the *current* split, so the count is not confounded
with any later change to `make_imbalanced_split`. Quantum rows come from the
main grid (the quantum arm was rescaled under both conventions, so its numbers
are the same either way). Classical rows come from the un-rescaled arm stored
by `case2_rescale_ablation.py`. The pre-registered protocol is then run
unchanged on the resulting 12x12 per-seed matrix of each (noise, ratio) cell.

Run from the repository root, after both grids exist:
    python code/case2_rescale_counterfactual.py
"""
import itertools
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from qml_imbalance import TIE_DECIMALS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "code", "results", "case2_grid.json")
ABL = os.path.join(ROOT, "code", "results", "case2_rescale_ablation.json")
OUT = os.path.join(ROOT, "code", "results", "case2_rescale_counterfactual.json")


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj, running = np.empty(m), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def certified_quantum_wins(M, models, quantum):
    """Holm-certified pairs in one cell where a quantum model leads."""
    pairs = list(itertools.combinations(range(len(models)), 2))
    ps = []
    for a, b in pairs:
        d = np.round(M[:, a] - M[:, b], TIE_DECIMALS)
        if np.all(d == 0):
            ps.append(1.0)
        else:
            try:
                ps.append(wilcoxon(d, zero_method="wilcox",
                                   alternative="two-sided")[1])
            except ValueError:
                ps.append(1.0)
    adj = holm(ps)
    out = []
    for (a, b), q in zip(pairs, adj):
        if q >= 0.05:
            continue
        diff = float(np.mean(M[:, a] - M[:, b]))
        hi, lo = (a, b) if diff > 0 else (b, a)
        if models[hi] in quantum and models[lo] not in quantum:
            out.append((models[hi], models[lo], float(q)))
    return out


def main():
    grid = json.load(open(GRID, encoding="utf-8"))
    cfg, runs = grid["config"], grid["runs"]
    abl = json.load(open(ABL, encoding="utf-8"))["runs"]
    models = cfg["quantum_models"] + cfg["classical_models"]
    quantum, seeds = set(cfg["quantum_models"]), cfg["seeds"]

    shared = {(r["model"], r["noise"], r["ratio"], r["seed"]):
              r["metrics"]["balanced_accuracy"] for r in runs}
    bare = {(r["model"], r["noise"], r["ratio"], r["seed"]):
            r["balanced_accuracy"] for r in abl if not r["rescale"]}
    asym = {**shared, **bare}

    totals, detail = {}, {}
    for label, src in (("shared", shared), ("asymmetric", asym)):
        n, rows = 0, []
        for noise in cfg["noise_levels"]:
            for ratio in cfg["ratios"]:
                M = np.array([[src[(m, noise, ratio, s)] for m in models]
                              for s in seeds])
                for hi, lo, q in certified_quantum_wins(M, models, quantum):
                    n += 1
                    rows.append(dict(noise=noise, ratio=ratio, quantum=hi,
                                     classical=lo, p_holm=q))
        totals[label], detail[label] = n, rows
        print(f"{label:11s} rescaling: {n} certified quantum wins")

    delta = totals["asymmetric"] - totals["shared"]
    print(f"\nsharing the rescaler cost the quantum arm {delta} certified "
          f"wins ({totals['asymmetric']} -> {totals['shared']}), "
          f"both counts under the current split")
    json.dump({"totals": totals, "detail": detail},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
