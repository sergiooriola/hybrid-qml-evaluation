"""Write a SHA-256 manifest of the archived result artefacts.

The thesis reports numbers computed from a fixed set of JSON grids. Anyone
re-running the experiment needs a way to tell whether they regenerated the
same artefacts or different ones. This writes `results/MANIFEST.sha256` in
the standard `sha256sum` format, so verification is a one-liner:

    sha256sum -c results/MANIFEST.sha256

Run from the repository root after any grid is regenerated:
    python code/make_manifest.py
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "code", "results", "MANIFEST.sha256")

ARTEFACTS = [
    "code/results/case2_grid.json",
    "code/results/case1_validation.json",
    "code/results/case2_ablation.json",
    "code/results/case2_rescale_ablation.json",
    "code/results/case2_rescale_counterfactual.json",
    "code/results/case2_roc_scores.json",
    "code/results/ps_gradient_check.json",
    "code/annex_datasets/results/annex_sinus.json",
    "code/annex_datasets/results/annex_corners.json",
    "code/annex_datasets/results/annex_spiral.json",
    "code/annex_datasets/results/annex_circles.json",
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    lines, missing = [], []
    for rel in ARTEFACTS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        digest = sha256(path)
        # sha256sum format: digest, two spaces, path relative to repo root
        lines.append(f"{digest}  {rel}")
        print(f"{digest[:16]}...  {rel}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    if missing:
        print("\nnot found (skipped):")
        for rel in missing:
            print(f"  {rel}")
    print(f"\nwrote {len(lines)} entries to {OUT}")


if __name__ == "__main__":
    main()
