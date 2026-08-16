"""Provenance: what the quantum-only feature rescaling was worth.

This script records the measurement that led to a change in the framework,
and is kept so that the numbers quoted in the thesis can be reproduced.

An earlier version of `run_one` applied the affine rescaler to [-pi, pi] for
quantum models only, on the grounds that gate angles confined to a small
interval probe only a nearly linear region of the encoding, while for a
scikit-learn estimator the step is optional. Classical baselines therefore
received the raw two-moons coordinates. This script measured that asymmetry:
it re-fits the six classical baselines over the full ratio grid *with* the
rescaling and compares them, cell by cell, against the un-rescaled values.

The finding was that five of the six baselines are insensitive to the step
(the random forest exactly so) but the parameter-matched MLP gains several
BA points from it. `run_one` was changed accordingly and now rescales for
every model, so both arms of the comparison see identical inputs.

The script fits each classical baseline twice under the current split, once
with the rescaling and once without, rather than comparing against a stored
snapshot of an earlier grid. Both arms therefore always reflect the
framework as it stands, and the measurement stays valid across later
changes to the split or to the environment. Only classical baselines are
re-fitted, which takes milliseconds each.

Run from the repository root:
    python code/case2_rescale_ablation.py
"""
import json
import os
import time

import numpy as np

from qml_imbalance import (ClassicalModel, RunResult, apply_class_weight,
                           compute_metrics, make_balanced_dataset,
                           make_imbalanced_split, rescale_to_pi,
                           tune_threshold)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Read only the grid config, so the ablation always uses the live settings.
GRID = os.path.join(ROOT, "code", "results", "case2_grid.json")
OUT = os.path.join(ROOT, "code", "results", "case2_rescale_ablation.json")

CLASSICAL = ["logreg", "svm_rbf", "svm_linear", "knn", "rf", "mlp_small"]


def run_one_scaling(model, ratio, seed, X_base, y_base, rescale,
                    n_train=500, n_val=400, n_test=1200):
    """`run_one`, with the rescaler switched on or off by `rescale`."""
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_imbalanced_split(
        X_base, y_base, train_ratio_majority=ratio,
        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
    )
    X_tr_h, y_tr_h, sw_tr_h = apply_class_weight(X_tr, y_tr)
    if rescale:
        X_tr_use, X_va_use, X_te_use = rescale_to_pi(X_tr_h, X_va, X_te)
    else:
        X_tr_use, X_va_use, X_te_use = X_tr_h, X_va, X_te

    t0 = time.time()
    state = model.fit(X_tr_use, y_tr_h, sw_tr_h, X_va_use, y_va, seed=seed)
    fit_time = time.time() - t0

    threshold, _ = tune_threshold(model.predict_score(state, X_va_use), y_va)
    metrics = compute_metrics(y_te, model.predict_score(state, X_te_use),
                              threshold=threshold)
    return RunResult(model=model.name, ratio=ratio, seed=seed,
                     metrics=metrics, threshold=threshold,
                     fit_time_s=fit_time, n_circuit_evals=0,
                     n_train_majority=int(np.sum(y_tr == 0)),
                     n_train_minority=int(np.sum(y_tr == 1)))


def main():
    if not os.path.exists(GRID):
        raise SystemExit(f"no grid at {GRID}; run qml_imbalance.py first.")
    cfg = json.load(open(GRID, encoding="utf-8"))["config"]
    ratios, seeds = cfg["ratios"], cfg["seeds"]
    noises = cfg["noise_levels"]

    ref, out, t_start = {}, [], time.time()
    for noise in noises:
        X_base, y_base = make_balanced_dataset(
            n_total=cfg["n_total_base"], noise=noise, seed=42)
        for name in CLASSICAL:
            model = ClassicalModel(name)
            for ratio in ratios:
                for seed in seeds:
                    kw = dict(n_train=cfg["n_train"], n_val=cfg["n_val"],
                              n_test=cfg["n_test"])
                    res = run_one_scaling(model, ratio, seed, X_base, y_base,
                                          rescale=True, **kw)
                    out.append(dict(model=name, noise=noise, ratio=ratio,
                                    seed=seed, rescale=True, **res.metrics))
                    bare = run_one_scaling(model, ratio, seed, X_base, y_base,
                                           rescale=False, **kw)
                    out.append(dict(model=name, noise=noise, ratio=ratio,
                                    seed=seed, rescale=False, **bare.metrics))
                    ref[(name, noise, ratio, seed)] = \
                        bare.metrics["balanced_accuracy"]
            print(f"  {name:11s} eta={noise}  done "
                  f"({time.time() - t_start:.0f}s)")

    # ---- comparison table: mean BA per (model, noise), raw vs rescaled ----
    print(f"\n{'model':12s} {'eta':>4s} {'raw BA':>8s} {'rescaled':>9s} "
          f"{'delta':>8s} {'max|d| cell':>12s}")
    print("-" * 60)
    summary = {}
    for name in CLASSICAL:
        for noise in noises:
            a, b, cell = [], [], {}
            for ratio in ratios:
                ra, rb = [], []
                for seed in seeds:
                    ra.append(ref[(name, noise, ratio, seed)])
                    rb.append(next(r["balanced_accuracy"] for r in out
                                   if r["model"] == name and r["noise"] == noise
                                   and r["ratio"] == ratio and r["seed"] == seed
                                   and r["rescale"]))
                a += ra
                b += rb
                cell[ratio] = float(np.mean(rb) - np.mean(ra))
            d = float(np.mean(b) - np.mean(a))
            worst = max(cell.items(), key=lambda kv: abs(kv[1]))
            summary[f"{name}|{noise}"] = dict(raw=float(np.mean(a)),
                                              rescaled=float(np.mean(b)),
                                              delta=d, per_ratio=cell)
            print(f"{name:12s} {noise:4.1f} {np.mean(a):8.4f} "
                  f"{np.mean(b):9.4f} {d:+8.4f} "
                  f"{worst[1]:+8.4f} @r={worst[0]}")

    alld = [v["delta"] for v in summary.values()]
    print(f"\nlargest |mean delta| over all 12 cells: {max(map(abs, alld)):.4f}")

    json.dump({"config": cfg, "runs": out, "summary": summary},
              open(OUT, "w", encoding="utf-8"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
