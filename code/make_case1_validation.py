"""
make_case1_validation.py
========================
Turn results/case1_validation.json (produced by run_validation.py) into the
thesis-ready artefacts for the Case Study 1 validation:

    figures/case1-val-layers.png   our accuracy vs paper Table IV (by layers)
    figures/case1-val-qubits.png   our accuracy vs paper Table III (by qubits)
    (LaTeX tables printed to stdout, ready to paste into 03a_case1.tex)

Run after run_validation.py has finished:
    python make_case1_validation.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "results", "case1_validation.json")
FIGDIR = os.path.join(HERE, "figures")


def load():
    with open(JSON) as f:
        d = json.load(f)
    # JSON stringifies integer dict keys; coerce paper reference keys back to int.
    for tbl in ("table_iv_n2", "table_iii_byqubits"):
        d["paper"][tbl] = {ds: {int(k): v for k, v in vals.items()}
                           for ds, vals in d["paper"][tbl].items()}
    return d


def rows(runs, kind):
    return [r for r in runs if r["kind"] == kind]


def layer_curve(runs, dataset):
    rs = sorted([r for r in rows(runs, "layer_sweep") if r["dataset"] == dataset],
                key=lambda r: r["n_layers"])
    return rs


def qubit_curve(runs, dataset):
    rs = sorted([r for r in rows(runs, "qubit_sweep") if r["dataset"] == dataset],
                key=lambda r: r["n_qubits"])
    return rs


# --- figures ----------------------------------------------------------------

def fig_sweep(d, kind, xkey, paper_key, xlabel, fname, title):
    runs = d["runs"]
    paper = d["paper"][paper_key]
    datasets = [name for name in d["config"]["datasets"]]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.0),
                             squeeze=False)
    for ax, name in zip(axes[0], datasets):
        rs = (layer_curve if kind == "layer_sweep" else qubit_curve)(runs, name)
        xs = [r[xkey] for r in rs]
        mean = [r["test_acc_mean"] for r in rs]
        std = [r["test_acc_std"] for r in rs]
        best = [r["test_acc_best"] for r in rs]
        ax.errorbar(xs, mean, yerr=std, marker="o", capsize=3,
                    label="VQC (this work), mean$\\pm$std", color="C0")
        ax.plot(xs, best, marker="^", ls="--", color="C0", alpha=0.6,
                label="VQC (this work), best seed")
        pdict = paper.get(name, {})
        px = sorted(k for k in pdict)
        ax.plot(px, [pdict[k] for k in px], marker="s", ls=":", color="C3",
                label="Reference QNN")
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


# --- LaTeX tables -----------------------------------------------------------

def latex_layer_table(d):
    runs = d["runs"]
    paper = d["paper"]["table_iv_n2"]
    datasets = d["config"]["datasets"]
    Ls = sorted({r["n_layers"] for r in rows(runs, "layer_sweep")})
    print("\n% --- Layer-sweep comparison (n=2 qubits) vs paper Table IV ---")
    print("\\begin{tabular}{l" + "c" * len(Ls) + "}")
    print("\\toprule")
    print("Dataset / source & " + " & ".join(f"$L={L}$" for L in Ls) + " \\\\")
    print("\\midrule")
    for name in datasets:
        rs = {r["n_layers"]: r for r in layer_curve(runs, name)}
        ours = " & ".join(f"{rs[L]['test_acc_best']:.3f}" if L in rs else "--"
                          for L in Ls)
        print(f"{name} (VQC, this work) & {ours} \\\\")
        prow = " & ".join(f"{paper[name][L]:.3f}" if L in paper.get(name, {})
                          else "--" for L in Ls)
        print(f"{name} (reference~\\cite{{rodriguezgrasa2024neural}}) & {prow} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")


def latex_qubit_table(d):
    runs = d["runs"]
    paper = d["paper"]["table_iii_byqubits"]
    datasets = d["config"]["datasets"]
    ns = sorted({r["n_qubits"] for r in rows(runs, "qubit_sweep")})
    print("\n% --- Qubit-sweep comparison (L=7) vs paper Table III ---")
    print("\\begin{tabular}{l" + "c" * len(ns) + "}")
    print("\\toprule")
    print("Dataset / source & " + " & ".join(f"$n={n}$" for n in ns) + " \\\\")
    print("\\midrule")
    for name in datasets:
        rs = {r["n_qubits"]: r for r in qubit_curve(runs, name)}
        ours = " & ".join(f"{rs[n]['test_acc_best']:.3f}" if n in rs else "--"
                          for n in ns)
        print(f"{name} (VQC, this work) & {ours} \\\\")
        prow = " & ".join(f"{paper[name][n]:.3f}" if n in paper.get(name, {})
                          else "--" for n in ns)
        print(f"{name} (reference~\\cite{{rodriguezgrasa2024neural}}) & {prow} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")


def encoding_summary(d):
    rs = rows(d["runs"], "encoding_cmp")
    if not rs:
        return
    print("\n% --- Encoding contrast (n=2, L=4): re-uploading vs single-encoding ---")
    for r in sorted(rs, key=lambda r: (r["dataset"], not r["reupload"])):
        mode = "re-uploading" if r["reupload"] else "single-encoding"
        print(f"  {r['dataset']:>8} {mode:>16}: "
              f"test_acc best={r['test_acc_best']:.3f} "
              f"mean={r['test_acc_mean']:.3f}+/-{r['test_acc_std']:.3f}")


def main():
    if not os.path.exists(JSON):
        sys.exit(f"missing {JSON}; run run_validation.py first")
    d = load()
    fig_sweep(d, "layer_sweep", "n_layers", "table_iv_n2",
              "Number of layers $L$", "case1-val-layers.png",
              "Layer sweep (n=2 qubits): VQC vs reference QNN")
    fig_sweep(d, "qubit_sweep", "n_qubits", "table_iii_byqubits",
              "Number of qubits $n$", "case1-val-qubits.png",
              "Qubit sweep (L=7): VQC vs reference QNN")
    latex_layer_table(d)
    latex_qubit_table(d)
    encoding_summary(d)


if __name__ == "__main__":
    main()
