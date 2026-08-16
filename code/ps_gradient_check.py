r"""Primitive-level validation of the parameter-shift gradient.

The head-to-head of Section 4.5 compares autodifferentiation and
parameter-shift trajectories on the Fixed VQC and the Data Reuploading
model, and on the *variational-only* parameters of the Trainable VQC. It
therefore leaves the one code path that is unique to the Trainable VQC
unvalidated: the four trainable encoding weights use the data-dependent
shift and the multiplicative prefactor of the generalised rule, rather
than the plain +-pi/2 shift.

This script closes that gap by checking every parameter of every
architecture against a central finite difference of the same cost
function. Finite differences are an independent numerical reference: they
call only the forward pass and know nothing about shift rules, so an
agreement between the two is evidence about the analytic gradient rather
than about a shared implementation.

Run:  python ps_gradient_check.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from qml_core import cost_np, parameter_shift_gradient
from qml_imbalance import (QuantumModel, make_balanced_dataset,
                           make_imbalanced_split, rescale_to_pi,
                           apply_class_weight, labels_to_pm1)

# Step size for the central difference. With an O(h^2) truncation error and
# an O(eps/h) round-off error on IEEE-754 double, the total error is
# minimised near h ~ eps^(1/3) ~ 6e-6; 1e-5 sits in that basin.
H = 1e-5
N_SAMPLES = 120
N_DRAWS = 5


def check(name: str, seed: int = 0) -> dict:
    spec = QuantumModel.SPEC[name]
    circuit_type, n_params = spec['circuit_type'], spec['n_params']
    rng = np.random.RandomState(seed)
    X, y = make_balanced_dataset(n_total=4000, noise=0.1, seed=42)
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_imbalanced_split(
        X, y, 0.70, n_train=N_SAMPLES, n_val=100, n_test=200, seed=seed)
    X_tr, y_tr, sw = apply_class_weight(X_tr, y_tr)
    X_tr, _, _ = rescale_to_pi(X_tr, X_va, X_te)
    y_pm = labels_to_pm1(y_tr)

    worst_abs, worst_rel, worst_idx = 0.0, 0.0, -1
    for _ in range(N_DRAWS):
        theta = np.pi * rng.rand(n_params)
        g_ps, _, _ = parameter_shift_gradient(theta, X_tr, y_pm,
                                              circuit_type=circuit_type,
                                              sample_weight=sw)
        for k in range(n_params):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += H
            tm[k] -= H
            lp = cost_np(tp, X_tr, y_pm, circuit_type=circuit_type,
                         sample_weight=sw)[0]
            lm = cost_np(tm, X_tr, y_pm, circuit_type=circuit_type,
                         sample_weight=sw)[0]
            g_fd = (lp - lm) / (2 * H)
            a = abs(g_ps[k] - g_fd)
            r = a / max(abs(g_fd), 1e-8)
            if a > worst_abs:
                worst_abs, worst_idx = a, k
            worst_rel = max(worst_rel, r if abs(g_fd) > 1e-6 else 0.0)
    return {'circuit': circuit_type, 'n_params': n_params,
            'max_abs_err': worst_abs, 'max_rel_err': worst_rel,
            'worst_param_index': worst_idx}


def main() -> None:
    rows = []
    for name in ('fixed_vqc', 'trainable_vqc', 'angle_vqc',
                 'data_reuploading', 'reuploading_2q', 'no_reuploading'):
        n = QuantumModel.SPEC[name]['n_params']
        r = check(name)
        r['model'] = name
        rows.append(r)
        print(f"{name:18s} {n:>3} params   "
              f"max |PS - FD| = {r['max_abs_err']:.3e}   "
              f"max rel = {r['max_rel_err']:.3e}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'ps_gradient_check.json')
    with open(out, 'w') as f:
        json.dump({'h': H, 'n_train': N_SAMPLES, 'n_draws': N_DRAWS,
                   'rows': rows}, f, indent=2)
    print(f"\nworst over all architectures: "
          f"{max(r['max_abs_err'] for r in rows):.3e}")
    print(f"written to {out}")


if __name__ == '__main__':
    main()
