"""Test-set ROC curves at the two extreme imbalance ratios.

The scalar ROC AUC of the main grid says how well each model ranks positives
above negatives; it does not say *where* on the operating range the imbalance
is paid. This script recovers the continuous test scores, which the cached
grid does not store, and plots the mean ROC curve per model.

Recovering the scores means re-fitting each (model, noise, ratio, seed)
combination at the two extreme ratios, so the run is the expensive part of
this analysis. The scores are therefore cached in
``results/case2_roc_scores.json``; once that file exists the figure redraws in
a second. Delete it to force a re-run.

Preprocessing matches ``run_one``: every model, quantum or classical, receives
the features rescaled to [-pi, pi].

Run from the repository root:
    python code/case2_roc_curves.py            # draw (re-fitting if needed)
    N_JOBS=-1 python code/case2_roc_curves.py  # ... in parallel
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import qml_imbalance as qi  # noqa: E402

CACHE = os.path.join(ROOT, "code", "results", "case2_roc_scores.json")
FIG = os.path.join(ROOT, "code", "figures", "case2-roc-curves.png")

RATIOS = [0.5, 0.99]
NOISES = [0.1, 0.3]
SEEDS = list(range(12))
FPR_GRID = np.linspace(0.0, 1.0, 201)

QUANTUM = ["data_reuploading", "no_reuploading", "fixed_vqc",
           "angle_vqc", "reuploading_2q", "trainable_vqc"]
CLASSICAL = ["logreg", "svm_rbf", "svm_linear", "knn", "rf", "mlp_small"]
ALL = QUANTUM + CLASSICAL

COLOURS = {
    "data_reuploading": "#d62728", "no_reuploading": "#7f7f7f",
    "fixed_vqc": "#ff7f0e", "angle_vqc": "#e377c2",
    "reuploading_2q": "#000000", "trainable_vqc": "#bcbd22",
    "logreg": "#1f77b4", "svm_rbf": "#2ca02c", "svm_linear": "#98df8a",
    "knn": "#9467bd", "rf": "#8c564b", "mlp_small": "#17becf",
}


def score_one(model_name, noise, ratio, seed):
    """One experiment, returning the continuous test scores.

    Mirrors ``qi.run_one`` step for step, including the shared feature
    rescaling, but returns the scores instead of the aggregated metrics.
    """
    Xb, yb = qi.make_balanced_dataset(n_total=8000, noise=noise, seed=42)
    X_tr, y_tr, X_va, y_va, X_te, y_te = qi.make_imbalanced_split(
        Xb, yb, train_ratio_majority=ratio,
        n_train=500, n_val=400, n_test=1200, seed=seed)
    X_tr_h, y_tr_h, sw = qi.apply_class_weight(X_tr, y_tr)
    X_tr_u, X_va_u, X_te_u = qi.rescale_to_pi(X_tr_h, X_va, X_te)
    model = (qi.QuantumModel(model_name, n_iter=200, lr=0.015)
             if model_name in qi.QuantumModel.SPEC
             else qi.ClassicalModel(model_name))
    state = model.fit(X_tr_u, y_tr_h, sw, X_va_u, y_va, seed=seed)
    return (np.asarray(y_te).tolist(),
            np.asarray(model.predict_score(state, X_te_u)).tolist())


def build_cache():
    tasks = [(m, n, r, s) for n in NOISES for r in RATIOS
             for m in ALL for s in SEEDS]
    n_jobs = int(os.environ.get("N_JOBS", "1"))
    print(f"re-fitting {len(tasks)} runs to recover test scores "
          f"(N_JOBS={n_jobs})")
    t0 = time.time()
    if n_jobs == 1:
        out = []
        for i, t in enumerate(tasks, 1):
            out.append(score_one(*t))
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}  ({time.time() - t0:.0f}s)")
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(score_one)(*t) for t in tasks)
    cache = {"|".join(map(str, t)): {"y": y, "s": s}
             for t, (y, s) in zip(tasks, out)}
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    print(f"cached {len(cache)} score vectors in {time.time() - t0:.0f}s")
    return cache


def mean_roc(curves):
    """Vertical averaging over seeds on a common FPR grid."""
    tprs, aucs = [], []
    for fpr, tpr in curves:
        t = np.interp(FPR_GRID, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(sk_auc(fpr, tpr))
    tprs = np.vstack(tprs)
    return tprs.mean(0), tprs.std(0), float(np.mean(aucs))


def main():
    cache = (json.load(open(CACHE, encoding="utf-8"))
             if os.path.exists(CACHE) else build_cache())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for i, noise in enumerate(NOISES):
        for j, ratio in enumerate(RATIOS):
            ax = axes[i][j]
            for mname in ALL:
                curves = []
                for seed in SEEDS:
                    e = cache[f"{mname}|{noise}|{ratio}|{seed}"]
                    fpr, tpr, _ = roc_curve(e["y"], e["s"])
                    curves.append((fpr, tpr))
                mu, sd, a = mean_roc(curves)
                ax.plot(FPR_GRID, mu, color=COLOURS[mname], lw=1.6,
                        ls="-" if mname in QUANTUM else "--",
                        label=f"{mname} ({a:.3f})")
                ax.fill_between(FPR_GRID, mu - sd, mu + sd,
                                color=COLOURS[mname], alpha=0.08, lw=0)
            ax.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.005)
            ax.set_title(f"$\\eta = {noise}$,  $r = {ratio}$", fontsize=11)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(fontsize=6.5, loc="lower right", ncol=2, framealpha=0.9)
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG, dpi=160, bbox_inches="tight")
    print(f"written {FIG}")


if __name__ == "__main__":
    main()
