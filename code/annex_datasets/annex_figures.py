r"""Cross-dataset summary figures for Annex C.

Extracted from `summary.ipynb` so that the two figures the annex includes
regenerate from the four cached grids with one command.

Run:  python code/annex_datasets/annex_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures")
DATASETS = ["sinus", "corners", "spiral", "circles"]

GRIDS = {d: json.load(open(os.path.join(RESULTS, f"annex_{d}.json"),
                           encoding="utf-8"))
         for d in DATASETS}
C = GRIDS[DATASETS[0]]["config"]
QUANTUM, CLASSICAL = C["quantum_models"], C["classical_models"]
ALL = QUANTUM + CLASSICAL
RATIOS = sorted(C["ratios"])
NOISES = C["noise_levels"]


def stat(ds, noise, ratio, model, metric="balanced_accuracy"):
    v = [x["metrics"][metric] for x in GRIDS[ds]["runs"]
         if x["noise"] == noise and x["ratio"] == ratio
         and x["model"] == model]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, 0.0)


def degradation() -> None:
    cq = plt.cm.viridis(np.linspace(0.15, 0.85, len(QUANTUM)))
    cc = plt.cm.plasma(np.linspace(0.15, 0.85, len(CLASSICAL)))
    fig, axes = plt.subplots(len(NOISES), len(DATASETS),
                             figsize=(5.0 * len(DATASETS),
                                      4.4 * len(NOISES)),
                             sharey=True, squeeze=False)
    for i, noise in enumerate(NOISES):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for m, c in zip(QUANTUM, cq):
                p = [stat(ds, noise, r, m) for r in RATIOS]
                mu = np.array([q[0] for q in p])
                sd = np.array([q[1] for q in p])
                ax.plot(RATIOS, mu, "-o", color=c, lw=2, ms=4,
                        label=f"Q: {m}")
                ax.fill_between(RATIOS, mu - sd, mu + sd, color=c,
                                alpha=0.15)
            for m, c in zip(CLASSICAL, cc):
                p = [stat(ds, noise, r, m) for r in RATIOS]
                mu = np.array([q[0] for q in p])
                ax.plot(RATIOS, mu, "--s", color=c, lw=1.3, ms=3,
                        alpha=0.85, label=f"C: {m}")
            ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
            ax.set_ylim([0.4, 1.02])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(ds)
            if j == 0:
                ax.set_ylabel(f"noise={noise}\nBalanced accuracy")
            ax.set_xlabel("majority ratio r")
    axes[0, -1].legend(loc="lower left", fontsize=6, ncol=2)
    plt.suptitle("BA degradation under imbalance, across datasets and "
                 "noise levels", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "summary-degradation.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def head_to_head(noise: float = 0.1) -> None:
    extreme = max(RATIOS)
    n_q = len(QUANTUM)
    colors = ["#2c7fb8"] * n_q + ["#d95f0e"] * (len(ALL) - n_q)
    fig, axes = plt.subplots(1, len(DATASETS),
                             figsize=(4.6 * len(DATASETS), 4.6),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], DATASETS):
        p = [stat(ds, noise, extreme, m) for m in ALL]
        means = [q[0] for q in p]
        stds = [q[1] for q in p]
        xs = np.arange(len(ALL))
        ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85, capsize=2)
        ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
        ax.axvline(n_q - 0.5, color="gray", ls=":", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(ALL, rotation=40, ha="right", fontsize=7)
        ax.set_title(ds)
        ax.set_ylim([0.4, 1.05])
        ax.grid(True, alpha=0.3, axis="y")
    axes[0, 0].set_ylabel(f"BA at r={extreme:.2f} (noise={noise})")
    plt.suptitle("Head-to-head at the most imbalanced ratio, across "
                 "datasets", y=1.03)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "summary-headtohead.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    degradation()
    head_to_head()
