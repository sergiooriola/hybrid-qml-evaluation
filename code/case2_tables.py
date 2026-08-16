"""Regenerate the Case II statistical tables from the cached grid.

Runs the pre-registered protocol (Friedman omnibus, then Holm-corrected
pairwise Wilcoxon) at every imbalance ratio and prints, as LaTeX table
bodies, the quantum-advantage table and its reverse. Reads the grid produced
by ``qml_imbalance.run_grid``; runs no experiments of its own.

Run from the repository root:
    python code/case2_tables.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from qml_imbalance import RunResult, friedman_wilcoxon_holm  # noqa: E402

GRID = os.path.join(ROOT, "code", "results", "case2_grid.json")

# Analysis target. Defaults to the archived grid; pass --grid PATH (or set
# CASE2_GRID) to point the same analysis at a grid you regenerated yourself.
if "--grid" in sys.argv:
    GRID = sys.argv[sys.argv.index("--grid") + 1]
elif os.environ.get("CASE2_GRID"):
    GRID = os.environ["CASE2_GRID"]


QUANTUM = ["data_reuploading", "no_reuploading", "fixed_vqc",
           "angle_vqc", "reuploading_2q", "trainable_vqc"]
CLASSICAL = ["logreg", "svm_rbf", "svm_linear", "knn", "rf", "mlp_small"]
TEX = {m: "\\texttt{" + m.replace("_", "\\_") + "}"
       for m in QUANTUM + CLASSICAL}


def load(metric="balanced_accuracy"):
    """Cached runs regrouped as {noise: [RunResult, ...]}."""
    raw = json.load(open(GRID, encoding="utf-8"))["runs"]
    by_noise = {}
    for r in raw:
        rr = RunResult(model=r["model"], ratio=r["ratio"], seed=r["seed"],
                       metrics=r["metrics"], threshold=r["threshold"],
                       fit_time_s=r["fit_time_s"],
                       n_circuit_evals=r["n_circuit_evals"],
                       n_train_majority=r["n_train_majority"],
                       n_train_minority=r["n_train_minority"])
        by_noise.setdefault(r["noise"], []).append(rr)
    return by_noise


def advantage_counts(results, alpha=0.05, metric="balanced_accuracy"):
    """{(quantum, classical): (n_quantum_wins, n_classical_wins, n_ratios)}."""
    per_ratio = friedman_wilcoxon_holm(results, alpha=alpha, metric=metric)
    ratios = sorted(per_ratio)
    counts = {(q, c): [0, 0, 0] for q in QUANTUM for c in CLASSICAL}
    for r in ratios:
        res = per_ratio[r]
        if res["friedman_p"] >= alpha:
            continue  # omnibus not rejected: no pairwise claim at this ratio
        for (a, b), st in res["pairwise"].items():
            for q, c in ((a, b), (b, a)):
                if q in QUANTUM and c in CLASSICAL:
                    counts[(q, c)][2] += 1
                    if st["reject_holm"]:
                        # mean_diff is defined for the (a, b) ordering
                        d = st["mean_diff"] if (a, b) == (q, c) else -st["mean_diff"]
                        if d > 0:
                            counts[(q, c)][0] += 1
                        else:
                            counts[(q, c)][1] += 1
    return counts, len(ratios)


def table_body(counts, n_ratios, reverse=False):
    idx = 1 if reverse else 0
    lines = []
    for q in QUANTUM:
        cells = []
        for c in CLASSICAL:
            n = counts[(q, c)][idx]
            cell = f"${n}/{n_ratios}$"
            if n:
                cell = f"$\\mathbf{{{n}}}/{n_ratios}$"
            cells.append(cell)
        lines.append(f"    {TEX[q]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


if __name__ == "__main__":
    metric = sys.argv[1] if len(sys.argv) > 1 else "balanced_accuracy"
    by_noise = load()
    for noise in sorted(by_noise):
        res = by_noise[noise]
        per_ratio = friedman_wilcoxon_holm(res, metric=metric)
        counts, n_ratios = advantage_counts(res, metric=metric)
        print(f"\n%%% noise eta = {noise}, metric = {metric}, "
              f"{n_ratios} ratios %%%")
        omnibus = [f"{r}: p={per_ratio[r]['friedman_p']:.2e}"
                   for r in sorted(per_ratio)]
        print("% Friedman omnibus: " + "; ".join(omnibus))
        print("% --- quantum wins ---")
        print(table_body(counts, n_ratios))
        print("% --- classical wins (reverse) ---")
        print(table_body(counts, n_ratios, reverse=True))
        tot_q = sum(v[0] for v in counts.values())
        tot_c = sum(v[1] for v in counts.values())
        print(f"% totals: quantum-favouring {tot_q}, "
              f"classical-favouring {tot_c}, "
              f"of {len(QUANTUM) * len(CLASSICAL) * n_ratios} comparisons")
