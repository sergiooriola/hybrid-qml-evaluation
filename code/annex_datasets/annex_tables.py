"""Recompute the cross-dataset advantage counts of Annex C.

Applies the pre-registered protocol (Friedman omnibus, then Holm-corrected
pairwise Wilcoxon) to each (dataset, noise) cell of the annex grid and reports
how many of the quantum-versus-classical comparisons are certified in each
direction. Reads the cached grids; runs no experiments.

Run from the repository root:
    python code/annex_datasets/annex_tables.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from qml_imbalance import RunResult, friedman_wilcoxon_holm  # noqa: E402

RESULTS = os.path.join(ROOT, "code", "annex_datasets", "results")
DATASETS = ["sinus", "corners", "spiral", "circles"]
QUANTUM = ["data_reuploading", "no_reuploading", "fixed_vqc",
           "angle_vqc", "reuploading_2q", "trainable_vqc"]
CLASSICAL = ["logreg", "svm_rbf", "svm_linear", "knn", "rf", "mlp_small"]
STRONG = {"svm_rbf", "rf"}   # knn is excluded from the strong tier here


def runs_for(dataset, noise):
    path = os.path.join(RESULTS, f"annex_{dataset}.json")
    raw = json.load(open(path, encoding="utf-8"))["runs"]
    return [RunResult(model=r["model"], ratio=r["ratio"], seed=r["seed"],
                      metrics=r["metrics"], threshold=r["threshold"],
                      fit_time_s=r["fit_time_s"],
                      n_circuit_evals=r["n_circuit_evals"],
                      n_train_majority=r["n_train_majority"],
                      n_train_minority=r["n_train_minority"])
            for r in raw if abs(r["noise"] - noise) < 1e-9]


def cell(dataset, noise, alpha=0.05):
    res = friedman_wilcoxon_holm(runs_for(dataset, noise), alpha=alpha)
    ratios = sorted(res)
    q_wins = c_wins = q_strong = 0
    omnibus_rejected = 0
    for r in ratios:
        cur = res[r]
        if cur["friedman_p"] >= alpha:
            continue
        omnibus_rejected += 1
        for (a, b), st in cur["pairwise"].items():
            if not st["reject_holm"]:
                continue
            for q, c in ((a, b), (b, a)):
                if q in QUANTUM and c in CLASSICAL:
                    d = st["mean_diff"] if (a, b) == (q, c) else -st["mean_diff"]
                    if d > 0:
                        q_wins += 1
                        if c in STRONG:
                            q_strong += 1
                    else:
                        c_wins += 1
    total = len(QUANTUM) * len(CLASSICAL) * len(ratios)
    return q_wins, c_wins, q_strong, total, omnibus_rejected, len(ratios)


if __name__ == "__main__":
    print(f"{'dataset':10s} {'noise':>5s} {'Q wins':>7s} {'C wins':>7s} "
          f"{'Q vs strong':>12s} {'of':>6s}  omnibus")
    print("-" * 62)
    for ds in DATASETS:
        for noise in (0.1, 0.3):
            q, c, qs, tot, rej, nr = cell(ds, noise)
            print(f"{ds:10s} {noise:5.1f} {q:7d} {c:7d} {qs:12d} {tot:6d}"
                  f"  {rej}/{nr} rejected")
