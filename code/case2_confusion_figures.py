r"""Confusion-matrix panels at the most extreme ratio (Annex D figure).

Extracted from `imbalance_experiments.ipynb` so that the figure regenerates
from the grid file with one command and cannot go stale behind a
cache-guarded notebook cell.

Run from the repository root or from code/:
    python code/case2_confusion_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "results", "case2_grid.json")

# Analysis target. Defaults to the archived grid; pass --grid PATH (or set
# CASE2_GRID) to point the same analysis at a grid you regenerated yourself.
if "--grid" in sys.argv:
    GRID = sys.argv[sys.argv.index("--grid") + 1]
elif os.environ.get("CASE2_GRID"):
    GRID = os.environ["CASE2_GRID"]

FIGDIR = os.path.join(HERE, "figures")


def main() -> None:
    D = json.load(open(GRID, encoding="utf-8"))
    R, C = D["runs"], D["config"]
    models = C["quantum_models"] + C["classical_models"]
    extreme = max(C["ratios"])
    seed = C["seeds"][0]

    for noise in C["noise_levels"]:
        ncols = 4
        nrows = (len(models) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes)
        for idx, mname in enumerate(models):
            ax = axes[idx // ncols, idx % ncols]
            hit = [r for r in R if r["model"] == mname
                   and r["noise"] == noise and r["ratio"] == extreme
                   and r["seed"] == seed]
            if not hit:
                ax.set_visible(False)
                continue
            k = hit[0]["metrics"]
            cm = np.array([[k["tn"], k["fp"]], [k["fn"], k["tp"]]])
            ax.imshow(cm, cmap="Blues", aspect="auto")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() * 0.5 else "black",
                        fontsize=11)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred 0", "pred 1"])
            ax.set_yticklabels(["true 0", "true 1"])
            ax.set_title(f"{mname}\nBA={k['balanced_accuracy']:.3f}",
                         fontsize=9)
        for idx in range(len(models), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        plt.suptitle(f"Confusion matrices at noise={noise}, "
                     f"r={extreme:.2f}, seed={seed}", y=1.02, fontsize=12)
        plt.tight_layout()
        out = os.path.join(FIGDIR,
                           f"case2-confusion-n{int(noise * 10)}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("written", out)


if __name__ == "__main__":
    main()
