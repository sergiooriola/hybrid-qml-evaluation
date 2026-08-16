"""Ablation of the two methodological mechanisms at the most extreme ratio.

Reproduces the table of Chapter 5: SVM-RBF and the trainable-encoding VQC at
r = 0.99, with cost-sensitive class weighting (cw) and validation-set
threshold tuning (tt) switched on and off independently. Answers why an RBF
SVM still reaches a high balanced accuracy with only five minority training
samples.

Feature rescaling is applied to every model, matching `run_one`, so the
ablation varies only cw and tt.

SVM-RBF runs on the full twelve seeds. The trainable VQC runs on four seeds
with 150 iterations, which keeps the ablation tractable on a single core.

Run from the repository root:
    python code/case2_ablation.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from qml_imbalance import (ClassicalModel, QuantumModel, apply_class_weight,
                           compute_metrics, make_balanced_dataset,
                           make_imbalanced_split, rescale_to_pi,
                           tune_threshold)

OUT = os.path.join(ROOT, "code", "results", "case2_ablation.json")
RATIO = 0.99
CONFIGS = [("cw+tt", True, True), ("cw only", True, False),
           ("tt only", False, True), ("neither", False, False)]


def one(model, seed, X_base, y_base, use_cw, use_tt,
        n_train=500, n_val=400, n_test=1200):
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_imbalanced_split(
        X_base, y_base, train_ratio_majority=RATIO,
        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
    if use_cw:
        X_tr, y_tr, sw = apply_class_weight(X_tr, y_tr)
    else:
        sw = np.ones(len(y_tr), dtype=float)
    X_tr, X_va, X_te = rescale_to_pi(X_tr, X_va, X_te)

    state = model.fit(X_tr, y_tr, sw, X_va, y_va, seed=seed)
    if use_tt:
        t, _ = tune_threshold(model.predict_score(state, X_va), y_va)
    else:
        t = 0.0
    return compute_metrics(y_te, model.predict_score(state, X_te),
                           threshold=t)["balanced_accuracy"]


def main():
    out, t0 = {}, time.time()
    for noise in (0.1, 0.3):
        X, y = make_balanced_dataset(n_total=8000, noise=noise, seed=42)
        for label, cw, tt in CONFIGS:
            for name, seeds, mk in (
                    ("svm_rbf", range(12), lambda: ClassicalModel("svm_rbf")),
                    ("trainable_vqc", range(4),
                     lambda: QuantumModel("trainable_vqc", n_iter=150))):
                vals = [one(mk(), s, X, y, cw, tt) for s in seeds]
                out[f"{name}|{noise}|{label}"] = float(np.mean(vals))
                print(f"  eta={noise}  {label:8s}  {name:14s} "
                      f"BA={np.mean(vals):.3f}  ({time.time() - t0:.0f}s)")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n% LaTeX table body")
    for label, _, _ in CONFIGS:
        cells = " & ".join(
            f"${out[f'{m}|{n}|{label}']:.3f}$"
            for m, n in (("svm_rbf", 0.1), ("svm_rbf", 0.3),
                         (None, None), ("trainable_vqc", 0.1),
                         ("trainable_vqc", 0.3)) if m)
        c = cells.split(" & ")
        print(f"    {label:22s} & {c[0]} & {c[1]} & & {c[2]} & {c[3]} \\\\")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
