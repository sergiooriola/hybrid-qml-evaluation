"""Compare two result grids semantically, ignoring machine-dependent fields.

A SHA-256 of the raw JSON answers a different question from the one a
reproducer actually has. The records carry `fit_time_s` and are written in
completion order, both of which depend on the machine and on the joblib worker
count, so a correct re-run legitimately produces a different hash. This
compares what the thesis actually reports: the per-run metrics, keyed on
(noise, model, ratio, seed), independent of row order.

Usage, from the code directory:
    python compare_grids.py results/case2_grid.json \
                            results/case2_grid_reproduced.json
"""
import json
import sys

METRICS = ("balanced_accuracy", "roc_auc", "tn", "fp", "fn", "tp")
# The selected threshold is part of the pre-registered decision procedure, so
# a re-run that reproduced the metrics with a different operating point has
# not reproduced the protocol. It is compared alongside the metrics.
FIELDS = METRICS + ("threshold",)
# Everything compared here is deterministic given the seed, so the tolerance
# only absorbs JSON float round-tripping, not genuine numerical drift.
TOL = 1e-12


def load(path):
    runs = json.load(open(path, encoding="utf-8"))["runs"]
    out, dupes = {}, []
    for r in runs:
        key = (r["noise"], r["model"], r["ratio"], r["seed"])
        m = r["metrics"] if "metrics" in r else r
        rec = {k: m[k] for k in FIELDS if k in m}
        if "threshold" not in rec and "threshold" in r:
            rec["threshold"] = r["threshold"]
        if key in out:
            dupes.append(key)
        out[key] = rec
    if dupes:
        raise SystemExit(
            f"{path}: {len(dupes)} duplicate (noise, model, ratio, seed) "
            f"keys, first {dupes[0]}. The file is not a clean grid; "
            f"comparing it would silently drop runs.")
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b))
    diffs = []
    for k in shared:
        for m in a[k]:
            if m not in b[k]:
                diffs.append((k, m, a[k][m], None))
            elif abs(float(a[k][m]) - float(b[k][m])) > TOL:
                diffs.append((k, m, a[k][m], b[k][m]))

    print(f"runs in {sys.argv[1]}: {len(a)}")
    print(f"runs in {sys.argv[2]}: {len(b)}")
    print(f"shared keys           : {len(shared)}")
    print(f"only in first         : {len(only_a)}")
    print(f"only in second        : {len(only_b)}")
    print(f"metric mismatches     : {len(diffs)}")

    for k in only_a[:10]:
        print(f"  missing from second: {k}")
    for k in only_b[:10]:
        print(f"  missing from first : {k}")
    for k, m, va, vb in diffs[:20]:
        print(f"  {k} {m}: {va} vs {vb}")

    ok = not diffs and not only_a and not only_b
    print("\nRESULT: identical" if ok else "\nRESULT: grids differ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
