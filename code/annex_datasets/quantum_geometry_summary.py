"""Balanced-point comparison of the six quantum architectures across geometries.

Reads the cached grids (two-moons from the Case II run, the four additional
geometries from the annex run), keeps the balanced column (r = 0.50) and the
six quantum models, and reports mean +/- std balanced accuracy over the twelve
seeds. Produces the LaTeX table body and the figure used in Case I.

Run from the repository root:
    python code/annex_datasets/quantum_geometry_summary.py
"""
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE2 = os.path.join(ROOT, "code", "results", "case2_grid.json")
ANNEX = os.path.join(ROOT, "code", "annex_datasets", "results", "annex_{}.json")
FIGDIR = os.path.join(ROOT, "code", "figures")

QUANTUM = ["data_reuploading", "no_reuploading", "fixed_vqc",
           "angle_vqc", "reuploading_2q", "trainable_vqc"]
LABEL = {
    "data_reuploading": "Data Reuploading",
    "no_reuploading": "No-Reuploading",
    "fixed_vqc": "Fixed VQC",
    "angle_vqc": "Angle VQC",
    "reuploading_2q": "Reuploading-2Q",
    "trainable_vqc": "Trainable VQC",
}
GEOMETRIES = ["moons", "sinus", "corners", "spiral", "circles"]
GEOM_LABEL = {"moons": "Two-moons", "sinus": "Sinus", "corners": "Corners",
              "spiral": "Spiral", "circles": "Circles"}
NOISES = [0.1, 0.3]
BALANCED = 0.50


def load_runs(geometry):
    path = CASE2 if geometry == "moons" else ANNEX.format(geometry)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def balanced_scores():
    """{(geometry, noise, model): (mean, std, n)} at the balanced ratio."""
    out = {}
    for geom in GEOMETRIES:
        runs = load_runs(geom)
        for noise in NOISES:
            for model in QUANTUM:
                vals = [r["metrics"]["balanced_accuracy"] for r in runs
                        if r["model"] == model
                        and abs(r["ratio"] - BALANCED) < 1e-9
                        and abs(r["noise"] - noise) < 1e-9]
                out[(geom, noise, model)] = (float(np.mean(vals)),
                                             float(np.std(vals, ddof=1)),
                                             len(vals))
    return out


def latex_table(scores):
    lines = []
    for noise in NOISES:
        lines.append(r"    \multicolumn{6}{c}{\textbf{Noise $\eta = %s$}} \\" % noise)
        lines.append(r"    \midrule")
        lines.append("    Architecture & " +
                     " & ".join(GEOM_LABEL[g] for g in GEOMETRIES) + r" \\")
        lines.append(r"    \midrule")
        # order rows by mean accuracy across geometries, strongest first
        order = sorted(
            QUANTUM,
            key=lambda m: -np.mean([scores[(g, noise, m)][0] for g in GEOMETRIES]))
        for model in order:
            cells = []
            for geom in GEOMETRIES:
                mean, std, _ = scores[(geom, noise, model)]
                best = max(scores[(geom, noise, m)][0] for m in QUANTUM)
                cell = f"${mean:.3f} \\pm {std:.3f}$"
                if abs(mean - best) < 1e-12:
                    cell = f"$\\mathbf{{{mean:.3f}}} \\pm {std:.3f}$"
                cells.append(cell)
            lines.append(f"    {LABEL[model]} & " + " & ".join(cells) + r" \\")
        if noise != NOISES[-1]:
            lines.append(r"    \midrule")
    return "\n".join(lines)


def figure(scores):
    """Small multiples: one panel per (noise, geometry).

    Each panel shows the six architectures as dots joined to the chance line
    by a thin stem. Balanced accuracy has a meaningful floor at $0.5$ rather
    than at $0$, so a dot plot anchored on chance is used instead of bars,
    whose length would otherwise be measured from a truncated baseline. The
    colour order matches the quantum palette used in \\cref{sec:case2}, so a
    given architecture keeps its colour across the whole document.
    """
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(QUANTUM)))
    ypos = np.arange(len(QUANTUM))[::-1]  # first model at the top

    fig, axes = plt.subplots(len(NOISES), len(GEOMETRIES),
                             figsize=(12.6, 5.4), sharex=True, sharey=True)
    for row, noise in enumerate(NOISES):
        for col, geom in enumerate(GEOMETRIES):
            ax = axes[row, col]
            ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, zorder=1)
            for i, model in enumerate(QUANTUM):
                mean, std, _ = scores[(geom, noise, model)]
                ax.plot([0.5, mean], [ypos[i], ypos[i]],
                        color=colours[i], linewidth=1.6, alpha=0.55, zorder=2)
                ax.errorbar(mean, ypos[i], xerr=std, fmt="o", markersize=6,
                            color=colours[i], ecolor=colours[i],
                            elinewidth=1.1, capsize=2.5,
                            markeredgecolor="black", markeredgewidth=0.5,
                            zorder=3)
            ax.set_xlim(0.46, 1.0)
            ax.set_ylim(-0.7, len(QUANTUM) - 0.3)
            ax.grid(axis="x", alpha=0.25, zorder=0)
            ax.tick_params(axis="both", labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(GEOM_LABEL[geom], fontsize=11)
            if col == 0:
                ax.set_yticks(ypos)
                ax.set_yticklabels([LABEL[m] for m in QUANTUM], fontsize=9)
                ax.set_ylabel(f"$\\eta = {noise}$", fontsize=11,
                              labelpad=8)
            if row == len(NOISES) - 1:
                ax.set_xlabel("Balanced accuracy", fontsize=9)
                ax.set_xticks([0.5, 0.7, 0.9])

    fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.10,
                        wspace=0.12, hspace=0.16)
    out = os.path.join(FIGDIR, "case1-quantum-geometries.png")
    fig.savefig(out, dpi=200)
    print(f"figure written to {out}")


if __name__ == "__main__":
    scores = balanced_scores()
    n = {v[2] for v in scores.values()}
    assert n == {12}, f"expected 12 seeds per cell, found {n}"
    print(latex_table(scores))
    print()
    for noise in NOISES:
        for geom in GEOMETRIES:
            best = max(QUANTUM, key=lambda m: scores[(geom, noise, m)][0])
            worst = min(QUANTUM, key=lambda m: scores[(geom, noise, m)][0])
            print(f"eta={noise} {geom:8s} best={LABEL[best]:16s} "
                  f"{scores[(geom, noise, best)][0]:.3f}   "
                  f"worst={LABEL[worst]:16s} {scores[(geom, noise, worst)][0]:.3f}")
    figure(scores)
