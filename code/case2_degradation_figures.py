"""Degradation curves for Case II, in a focused and a complete version.

The complete version plots all twelve models and is reported in the annex.
The focused version plots the eight models the discussion of Case II actually
argues over: the three strong classical baselines, the capacity-matched MLP,
logistic regression as the linear reference, and the three quantum
architectures that lead at least one cell of the balanced comparison.

Four models are left out of the focused version and appear only in the
complete one. Three of them (Fixed VQC, Data Reuploading, No-Reuploading) are
the low band; the fourth (linear-kernel SVM) tracks logistic regression so
closely at every ratio that the two curves overlap. Dropping them removes a
third of the ink without removing anything the surrounding text reads off the
figure. Every model keeps the colour it has in the complete version, so a
curve means the same model in both figures.

Run from the repository root:
    python code/case2_degradation_figures.py
"""
import json
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "code", "results", "case2_grid.json")

# Analysis target. Defaults to the archived grid; pass --grid PATH (or set
# CASE2_GRID) to point the same analysis at a grid you regenerated yourself.
if "--grid" in sys.argv:
    GRID = sys.argv[sys.argv.index("--grid") + 1]
elif os.environ.get("CASE2_GRID"):
    GRID = os.environ["CASE2_GRID"]

FIGDIR = os.path.join(ROOT, "code", "figures")

# Full model lists, in the order that fixes the palette. Never reorder these:
# a model's colour is its index here, so the focused figure cannot repaint the
# curves it keeps.
QUANTUM = ["data_reuploading", "no_reuploading", "fixed_vqc",
           "angle_vqc", "reuploading_2q", "trainable_vqc"]
CLASSICAL = ["logreg", "svm_rbf", "svm_linear", "knn", "rf", "mlp_small"]

FOCUS = {"trainable_vqc", "reuploading_2q", "angle_vqc",
         "svm_rbf", "rf", "knn", "mlp_small", "logreg"}

CMAP_Q = plt.cm.viridis(np.linspace(0.15, 0.85, len(QUANTUM)))
CMAP_C = plt.cm.plasma(np.linspace(0.15, 0.85, len(CLASSICAL)))


def load():
    with open(GRID, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["runs"], sorted({r["ratio"] for r in data["runs"]}), \
        sorted({r["noise"] for r in data["runs"]})


def curve(runs, model, noise, ratios):
    means, stds = [], []
    for r in ratios:
        vals = [x["metrics"]["balanced_accuracy"] for x in runs
                if x["model"] == model
                and abs(x["ratio"] - r) < 1e-9
                and abs(x["noise"] - noise) < 1e-9]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    return np.array(means), np.array(stds)


def panel(ax, runs, noise, ratios, models, n_train=500):
    """One noise level.

    The horizontal axis is the number of minority training samples, on a
    log scale, rather than the ratio itself. That is the variable the
    degradation actually tracks: the ratios are log-spaced in it, and on a
    linear-in-$r$ axis nine tenths of the effect is compressed into the last
    fifth of the plot.
    """
    minority = np.array([round(n_train * (1 - r)) for r in ratios], dtype=float)
    for family, names, cmap, style in (
            ("Q", QUANTUM, CMAP_Q, dict(linestyle="-", marker="o",
                                        linewidth=2, markersize=5, alpha=1.0)),
            ("C", CLASSICAL, CMAP_C, dict(linestyle="--", marker="s",
                                          linewidth=1.5, markersize=4,
                                          alpha=0.85))):
        for idx, name in enumerate(names):
            if name not in models:
                continue
            mu, sd = curve(runs, name, noise, ratios)
            ax.plot(minority, mu, color=cmap[idx],
                    label=f"{family}: {name}", **style)
            ax.fill_between(minority, mu - sd, mu + sd,
                            color=cmap[idx], alpha=0.12)
    ax.set_xscale("log")
    ax.invert_xaxis()  # imbalance increases to the right
    ax.set_xticks(minority)
    ax.set_xticklabels([f"{int(m)}" for m in minority], fontsize=7)
    ax.minorticks_off()
    secax = ax.secondary_xaxis("top")
    secax.set_xscale("log")
    secax.set_xticks(minority)
    secax.set_xticklabels([f"{r:g}" for r in ratios], fontsize=7)
    secax.set_xlabel("Train majority ratio  $r$", fontsize=9)
    ax.set_xlabel("Minority training samples  (log scale)")
    ax.set_title(f"noise $\\eta$ = {noise}", pad=26)
    ax.set_ylim([0.4, 1.02])
    ax.grid(True, alpha=0.3)


def build(models, outfile, ncol):
    runs, ratios, noises = load()
    fig, axes = plt.subplots(1, len(noises), figsize=(7 * len(noises), 4.8),
                             sharey=True)
    for ax, noise in zip(np.atleast_1d(axes), noises):
        panel(ax, runs, noise, ratios, models)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(loc="lower left", fontsize=7, ncol=ncol)
    fig.tight_layout()
    out = os.path.join(FIGDIR, outfile)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"written {out}  ({len(models)} models)")


if __name__ == "__main__":
    build(FOCUS, "case2-degradation-main.png", ncol=2)
    build(set(QUANTUM) | set(CLASSICAL), "case2-degradation-full.png", ncol=2)
