r"""Head-to-head bars, AUC degradation curves and the training-set panel.

Extracted from `imbalance_experiments.ipynb` so that every grid-derived
figure of Case Study II regenerates from the results file with one command.
The training-set panel is included because it draws the splits directly and
therefore has to follow any change to `make_imbalanced_split`.

Run:  python code/case2_remaining_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from qml_imbalance import (make_balanced_dataset,        # noqa: E402
                           make_imbalanced_split)

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "results", "case2_grid.json")

# Analysis target. Defaults to the archived grid; pass --grid PATH (or set
# CASE2_GRID) to point the same analysis at a grid you regenerated yourself.
if "--grid" in sys.argv:
    GRID = sys.argv[sys.argv.index("--grid") + 1]
elif os.environ.get("CASE2_GRID"):
    GRID = os.environ["CASE2_GRID"]

FIGDIR = os.path.join(HERE, "figures")

D = json.load(open(GRID, encoding="utf-8"))
RUNS, C = D["runs"], D["config"]
QUANTUM, CLASSICAL = C["quantum_models"], C["classical_models"]
ALL = QUANTUM + CLASSICAL
RATIOS = sorted(C["ratios"])


def stat(noise, ratio, model, metric):
    v = [x["metrics"][metric] for x in RUNS
         if x["noise"] == noise and x["ratio"] == ratio
         and x["model"] == model]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, 0.0)


def head_to_head() -> None:
    extreme = max(RATIOS)
    fig, axes = plt.subplots(1, len(C["noise_levels"]),
                             figsize=(7 * len(C["noise_levels"]), 4.7),
                             sharey=True)
    for ax, noise in zip(np.atleast_1d(axes), C["noise_levels"]):
        pairs = [stat(noise, extreme, m, "balanced_accuracy") for m in ALL]
        means = [p[0] for p in pairs]
        stds = [p[1] for p in pairs]
        n_q = len(QUANTUM)
        colors = ["#2c7fb8"] * n_q + ["#d95f0e"] * (len(ALL) - n_q)
        xs = np.arange(len(ALL))
        ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8,
                   alpha=0.4)
        ax.axvline(n_q - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax.text(n_q / 2 - 0.5, 1.02, "quantum", ha="center",
                fontsize=10, color="#2c7fb8")
        ax.text((n_q + len(ALL)) / 2 - 0.5, 1.02, "classical",
                ha="center", fontsize=10, color="#d95f0e")
        ax.set_xticks(xs)
        ax.set_xticklabels(ALL, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Balanced accuracy at r = 0.99 (mean ± std)")
        ax.set_title(f"noise = {noise}")
        ax.set_ylim([0.4, 1.05])
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Head-to-head at the most imbalanced ratio "
                 f"(r = {extreme:.2f})", y=1.02, fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "case2-headtohead.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def auc_degradation() -> None:
    fig, axes = plt.subplots(1, len(C["noise_levels"]),
                             figsize=(7 * len(C["noise_levels"]), 4.5),
                             sharey=True)
    cq = plt.cm.viridis(np.linspace(0.15, 0.85, len(QUANTUM)))
    cc = plt.cm.plasma(np.linspace(0.15, 0.85, len(CLASSICAL)))
    for ax, noise in zip(np.atleast_1d(axes), C["noise_levels"]):
        for group, cmap, style, kw in (
                (QUANTUM, cq, "-o", dict(linewidth=2, markersize=5,
                                         alpha=1.0)),
                (CLASSICAL, cc, "--s", dict(linewidth=1.5, markersize=4,
                                            alpha=0.85))):
            tag = "Q" if group is QUANTUM else "C"
            for mname, color in zip(group, cmap):
                pairs = [stat(noise, r, mname, "roc_auc") for r in RATIOS]
                mu = np.array([p[0] for p in pairs])
                sd = np.array([p[1] for p in pairs])
                ax.plot(RATIOS, mu, style, color=color,
                        label=f"{tag}: {mname}", **kw)
                ax.fill_between(RATIOS, mu - sd, mu + sd, color=color,
                                alpha=0.15 if tag == "Q" else 0.10)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8,
                   alpha=0.4)
        ax.set_xlabel("Train majority ratio  r")
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"noise = {noise}")
        ax.set_ylim([0.55, 1.01])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, ncol=2)
    plt.suptitle("ROC AUC (threshold-independent) under increasing "
                 "class imbalance", y=1.02, fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "case2-auc-degradation.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def train_sets() -> None:
    # Only the two extremes of the ratio axis are shown. The intermediate
    # panels interpolate visually between them and cost eight further
    # panels to say so.
    shown = [min(RATIOS), max(RATIOS)]
    fig, axes = plt.subplots(
        len(C["noise_levels"]), len(shown),
        figsize=(2.7 * len(shown), 2.7 * len(C["noise_levels"])),
        sharex="row", sharey="row")
    axes = np.atleast_2d(axes)
    for i, noise in enumerate(C["noise_levels"]):
        X, y = make_balanced_dataset(n_total=C["n_total_base"],
                                     noise=noise, seed=42)
        for j, r in enumerate(shown):
            ax = axes[i, j]
            Xt, yt, *_ = make_imbalanced_split(
                X, y, train_ratio_majority=r, n_train=C["n_train"],
                n_val=C["n_val"], n_test=C["n_test"], seed=0)
            n0, n1 = int((yt == 0).sum()), int((yt == 1).sum())
            ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1], s=5, alpha=0.5,
                       c="#1f77b4")
            ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1], s=10, alpha=0.8,
                       c="#d62728")
            ax.set_title(f"η={noise}, r={r:.2f}\n({n0}/{n1})", fontsize=9)
            ax.grid(True, alpha=0.3)
    plt.suptitle("Train sets at each (noise, ratio) point (seed=0)", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "case2-train-sets.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    head_to_head()
    auc_degradation()
    train_sets()
