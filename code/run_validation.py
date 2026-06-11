"""
run_validation.py
=================
Case Study 1 *validation* sweep. Trains the generalised VQC (``vqc_general``)
over a grid of (n_qubits, n_layers) on two balanced toy datasets from
Rodriguez-Grasa, Ban & Sanz (arXiv:2401.04642v2) and reports test accuracy next
to that paper's published numbers, so the framework can be shown to reproduce
them and a (n_qubits, n_layers) choice can be justified for Case Study 2.

Datasets: corners and circles. These are the only two toy datasets for which
the reference reports BOTH a layer table (its Table IV, 2 qubits, L=5..8) and a
qubit table (its Table III, n=1..8). Its Table III is taken at L=7 (its n=2
entries equal the Table IV L=7 entries exactly), so the qubit sweep here uses
L=7 for a like-for-like comparison.

Two sweeps mirror the two reference tables:

    Layer sweep : n=2 qubits, L=1..8      -> compare with paper Table IV
    Qubit sweep : L=7 layers,  n=1..4     -> compare with paper Table III

Plus a small encoding contrast (re-uploading vs single-encoding at n=2, L=4) to
make the role of data re-uploading explicit.

The data noise is kept low (0.02) so the Bayes ceiling of each dataset (checked
separately with an RBF-SVM: corners ~0.99, circles ~0.95) lies above all of the
reference's reported numbers; accuracy is therefore limited by the model's
capacity (qubits / layers), as in the reference's ideal simulations, not by
label noise.

Output: results/case1_validation.json (checkpointed after every config) plus a
printed comparison table.

Run:
    python -u run_validation.py
    SEEDS=5 ITERS=150 python -u run_validation.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import toy_datasets as td
from vqc_general import VQCConfig, run_config

# --- configuration ----------------------------------------------------------
DATASETS = ("corners", "circles")
NOISE = float(os.environ.get("NOISE", "0.02"))
N_SAMPLES = 2000
N_TRAIN, N_VAL, N_TEST = 400, 200, 300
SEEDS = tuple(range(int(os.environ.get("SEEDS", "3"))))
ITERS = int(os.environ.get("ITERS", "120"))
LR = float(os.environ.get("LR", "0.05"))

LAYER_SWEEP_QUBITS = 2
LAYER_SWEEP_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]
QUBIT_SWEEP_LAYERS = 7
QUBIT_SWEEP_QUBITS = [1, 2, 3, 4]

OUT = os.path.join(os.path.dirname(__file__), "results", "case1_validation.json")

# Paper reference test accuracies (arXiv:2401.04642v2, Appendix E; QNN columns).
PAPER = {
    # Table IV: 2-qubit QNN accuracy by number of layers.
    "table_iv_n2": {
        "sinus":   {5: 0.948, 6: 0.956, 7: 0.966, 8: 0.958},
        "spiral":  {5: 0.952, 6: 0.974, 7: 0.978, 8: 0.980},
        "corners": {5: 0.948, 6: 0.936, 7: 0.934, 8: 0.916},
        "circles": {5: 0.786, 6: 0.808, 7: 0.792, 8: 0.844},
    },
    # Table III: n-to-n QNN accuracy by number of qubits (at L=7).
    "table_iii_byqubits": {
        "corners": {1: 0.886, 2: 0.934, 3: 0.948, 4: 0.952,
                    5: 0.954, 6: 0.956, 7: 0.960, 8: 0.962},
        "circles": {1: 0.698, 2: 0.792, 3: 0.820, 4: 0.858,
                    5: 0.934, 6: 0.944, 7: 0.944, 8: 0.946},
    },
    # Table I: single-qubit QNN accuracy.
    "table_i_n1": {
        "sinus": 0.890, "corners": 0.886, "spiral": 0.800, "circles": 0.698,
    },
}


def _result_dict(res, kind):
    return {
        "kind": kind,
        "dataset": res.dataset,
        "n_qubits": res.cfg.n_qubits,
        "n_layers": res.cfg.n_layers,
        "reupload": res.cfg.reupload,
        "rot": res.cfg.rot,
        "n_params": res.n_params,
        "test_acc_mean": res.test_acc_mean,
        "test_acc_std": res.test_acc_std,
        "test_acc_best": res.test_acc_best,
        "val_acc_mean": res.val_acc_mean,
        "per_seed_test_acc": res.per_seed_test_acc,
        "mean_evals": res.mean_evals,
        "mean_time_s": res.mean_time_s,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    config_meta = {
        "datasets": list(DATASETS), "noise": NOISE, "n_samples": N_SAMPLES,
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "seeds": list(SEEDS), "iters": ITERS, "lr": LR,
        "rot": "su2", "entangle": "cz", "init": "pi_uniform",
        "qubit_sweep_layers": QUBIT_SWEEP_LAYERS,
    }
    records = []
    data = {name: td.make_dataset(name, n_samples=N_SAMPLES, noise=NOISE,
                                  random_state=42)
            for name in DATASETS}

    def save():
        with open(OUT + ".tmp", "w") as f:
            json.dump({"config": config_meta, "paper": PAPER, "runs": records},
                      f, indent=1, default=float)
        os.replace(OUT + ".tmp", OUT)

    def do(name, cfg, kind):
        X, y = data[name]
        res = run_config(X, y, cfg, dataset=name, seeds=SEEDS,
                         n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                         n_iter=ITERS, lr=LR, verbose=True)
        records.append(_result_dict(res, kind))
        save()
        return res

    print("=" * 72, flush=True)
    print(f"LAYER SWEEP  (n={LAYER_SWEEP_QUBITS} qubits, L=1..8)  vs paper Table IV",
          flush=True)
    print("=" * 72, flush=True)
    for name in DATASETS:
        for L in LAYER_SWEEP_LAYERS:
            cfg = VQCConfig(n_qubits=LAYER_SWEEP_QUBITS, n_layers=L,
                            reupload=True, rot="su2")
            do(name, cfg, "layer_sweep")

    print("=" * 72, flush=True)
    print(f"QUBIT SWEEP  (L={QUBIT_SWEEP_LAYERS} layers, n=1..4)  vs paper Table III",
          flush=True)
    print("=" * 72, flush=True)
    for name in DATASETS:
        for nq in QUBIT_SWEEP_QUBITS:
            cfg = VQCConfig(n_qubits=nq, n_layers=QUBIT_SWEEP_LAYERS,
                            reupload=True, rot="su2")
            do(name, cfg, "qubit_sweep")

    print("=" * 72, flush=True)
    print("ENCODING CONTRAST  (n=2, L=4): re-uploading vs single-encoding",
          flush=True)
    print("=" * 72, flush=True)
    for name in DATASETS:
        for reup in (True, False):
            cfg = VQCConfig(n_qubits=2, n_layers=4, reupload=reup, rot="su2")
            do(name, cfg, "encoding_cmp")

    # --- printed comparison tables ------------------------------------------
    def cmp_table(kind, ref_key, ref_index):
        print("\n" + "=" * 72, flush=True)
        title = ("Table IV (n=2, by layers)" if kind == "layer_sweep"
                 else "Table III (L=7, by qubits)")
        print(f"COMPARISON WITH PAPER {title}", flush=True)
        print("=" * 72, flush=True)
        print(f"{'dataset':>8} {ref_index:>3} {'ours mean':>10} {'ours best':>10} "
              f"{'paper':>7} {'diff best':>10}", flush=True)
        for r in records:
            if r["kind"] != kind:
                continue
            key = r["n_layers"] if kind == "layer_sweep" else r["n_qubits"]
            paper = PAPER[ref_key].get(r["dataset"], {}).get(key)
            if paper is None:
                ref, diff = "  -  ", "  -  "
            else:
                ref = f"{paper:.3f}"
                diff = f"{r['test_acc_best'] - paper:+.3f}"
            print(f"{r['dataset']:>8} {key:>3} "
                  f"{r['test_acc_mean']:>10.3f} {r['test_acc_best']:>10.3f} "
                  f"{ref:>7} {diff:>10}", flush=True)

    cmp_table("layer_sweep", "table_iv_n2", "L")
    cmp_table("qubit_sweep", "table_iii_byqubits", "n")
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
