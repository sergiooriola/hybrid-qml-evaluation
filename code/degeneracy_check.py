"""Does the BA < 0.55 flag actually catch near-constant predictors?

The pre-registered rule flags a cell whose mean balanced accuracy falls below
0.55 as an ``essentially constant predictor''. Balanced accuracy near 0.5 is
by itself only evidence of near-chance performance: it does not establish the
failure mechanism. This script checks the mechanism directly, from the
confusion matrices recorded with every run, so that the label the thesis
attaches to the flag is verified rather than assumed.

For each flagged cell it reports the predicted-positive rate. A constant or
near-majority predictor sends that rate to 0 (always predict majority) or to 1
(always predict minority); a model that discriminates weakly but genuinely
keeps it away from both extremes.

Run from the repository root:
    python code/degeneracy_check.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRIDS = {
    "moons": os.path.join(ROOT, "code", "results", "case2_grid.json"),
    **{d: os.path.join(ROOT, "code", "annex_datasets", "results",
                       f"annex_{d}.json")
       for d in ("sinus", "corners", "spiral", "circles")},
}
FLAG = 0.55
# A predicted-positive rate outside this band is one-sided enough to call the
# predictor near-constant on a balanced test set.
EXTREME = 0.15


def main():
    flagged = degenerate = 0
    counterexamples = []
    for name, path in GRIDS.items():
        if not os.path.exists(path):
            print(f"  (skipping {name}: no grid at {path})")
            continue
        data = json.load(open(path, encoding="utf-8"))
        cfg, runs = data["config"], data["runs"]
        models = cfg["quantum_models"] + cfg["classical_models"]
        for noise in cfg["noise_levels"]:
            for ratio in cfg["ratios"]:
                for m in models:
                    cell = [r for r in runs if r["noise"] == noise
                            and r["ratio"] == ratio and r["model"] == m]
                    if not cell:
                        continue
                    ba = np.mean([r["metrics"]["balanced_accuracy"]
                                  for r in cell])
                    if ba >= FLAG:
                        continue
                    flagged += 1
                    # predicted-positive rate = (tp + fp) / n_test
                    ppr = np.mean([(r["metrics"]["tp"] + r["metrics"]["fp"]) /
                                   (r["metrics"]["tp"] + r["metrics"]["fp"] +
                                    r["metrics"]["tn"] + r["metrics"]["fn"])
                                   for r in cell])
                    if ppr < EXTREME or ppr > 1 - EXTREME:
                        degenerate += 1
                    else:
                        counterexamples.append(
                            (name, noise, ratio, m, float(ba), float(ppr)))

    print(f"cells below the {FLAG} flag          : {flagged}")
    print(f"  of which one-sided (PPR outside "
          f"[{EXTREME}, {1 - EXTREME}]): {degenerate}")
    print(f"  of which still two-sided          : "
          f"{flagged - degenerate}")
    if counterexamples:
        print("\ncells flagged as near-chance whose predictions are NOT "
              "near-constant:")
        for name, nz, r, m, ba, ppr in sorted(counterexamples,
                                              key=lambda t: -t[5]):
            print(f"  {name:8s} eta={nz} r={r:<5} {m:18s} "
                  f"BA={ba:.3f} predicted-positive rate={ppr:.3f}")


if __name__ == "__main__":
    main()
