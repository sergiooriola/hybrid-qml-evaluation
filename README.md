# Hybrid Quantum-Classical Machine Learning: An Evaluation Framework

Master's Thesis (TFM) project. Designs and implements a modular evaluation
framework for Variational Quantum Circuit (VQC) classifiers and exercises
it on two case studies: a methodological validation of two gradient
estimation strategies (autodifferentiation vs parameter-shift), and a
pre-registered comparison of three VQC architectures against five
capacity-matched classical baselines under increasing class imbalance,
on two `make_moons` datasets of contrasting geometric difficulty.

The full motivation, methodology, results and discussion are documented
in the accompanying Master's Thesis manuscript. This README covers the
project structure and how to reproduce its results from source.

---

## Project Structure

```
project/
├── README.md            <- you are here
├── LICENSE              <- MIT
├── requirements.txt     <- Python dependencies
│
└── code/                <- framework + case-study notebooks
    ├── qml_core.py            quantum primitives + Adam + parameter-shift
    ├── qml_imbalance.py       data + class weighting + models + runner
    ├── toy_datasets.py        4 extra geometric datasets (make_moons-compatible)
    │
    ├── VQC_Autodiff_and_ParameterShift.ipynb   Case Study 1 (Sec. 3.1)
    ├── imbalance_experiments.ipynb             Case Study 2 (Sec. 3.2)
    │
    ├── run_experiments.py     Case Study 2 grid driver (per-task checkpointing)
    ├── analyse_results.py     dump every balanced-accuracy statistic for the thesis
    ├── analyse_auc.py         same, for the threshold-independent ROC AUC
    ├── plot_roc.py            test-set ROC curves (Sec. 3.2)
    ├── smoke_test.py          quick end-to-end verification of the framework
    │
    ├── vqc_general.py         n-qubit / L-layer VQC for the Case 1 validation
    ├── run_validation.py      Case 1 sweep vs the published reference (corners/circles)
    ├── make_case1_validation.py   build the Case-1 validation figures + LaTeX tables
    │
    ├── results/
    │   ├── runs_simplified.json   1,344 runs from the Case Study 2 grid
    │   └── case1_validation.json  accuracy sweep for the Case 1 validation
    │
    └── annex_datasets/        Annex C: the Case 2 grid re-run on 4 toy datasets
        ├── run_annex.py           full-grid runner (corners, circles, spiral, sinus)
        ├── build_notebooks.py     regenerate the 4 per-dataset notebooks
        ├── build_summary.py       regenerate summary.ipynb
        ├── annex_<dataset>.ipynb  per-dataset replica of the Case 2 notebook
        ├── summary.ipynb          the four datasets side by side
        └── results/, figures/     cached runs + generated figures
```

---

## Design at a glance

The framework is built around five principles (see Section 2.4 of the
thesis):

1. **Modularity** — six independent modules (data, circuit, training,
   class weighting, inference, runner) with explicit contracts.
2. **Reproducibility** — every random source keyed on `(seed, ratio)`;
   bit-identical results across parallel/sequential execution.
3. **Uniform interface** — `QuantumModel` and `ClassicalModel` both
   expose the same `fit / predict_score / eval_count` contract.
4. **Fixed imbalance-handling strategy** — `class_weight = "balanced"`,
   applied identically to every model. Holding the handler fixed removes
   a confound that would otherwise make the quantum-vs-classical
   comparison difficult to interpret.
5. **Dependency-light core** — pure NumPy quantum forward; Qibo+PyTorch
   only on the autodiff path.

The evaluation protocol is built around one primary metric and one
statistical pipeline:

* **Balanced accuracy** — the single primary metric of the thesis.
* **Friedman → Wilcoxon → Holm** — paired non-parametric protocol per
  `(noise, ratio)`. With 8 models and 12 seeds, the smallest attainable
  Holm-adjusted p-value is `28 × 2 / 2¹² ≈ 0.0137`, comfortably below
  `α = 0.05`.

---

## Quick start

### 1. Python interpreter

The framework was developed and tested against **CPython 3.12**.

```bash
python3 --version   # should report Python 3.12.x
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The framework's parameter-shift path (Case Study 2) only needs NumPy,
scikit-learn, scipy and joblib; Qibo and PyTorch are required only by
the autodifferentiation path of Case Study 1.

### 3. Verify the framework works

```bash
cd code/
python3 smoke_test.py
```

Expected output ends with `[ALL CHECKS PASSED]`. The smoke test exercises
imports, the cross-module circuit-eval counter, `train_ps_run` with
parameter-history capture, end-to-end `run_one` for one quantum and one
classical model, the `apply_class_weight` weighting convention, and the
`friedman_wilcoxon_holm` pipeline on synthetic data.

### 4. Reproduce Case Study 2 end-to-end

The full grid is **1,344 runs**: 8 models × 7 ratios × 12 seeds × 2 noise
levels. On a single CPU this takes roughly 90 minutes. The script
checkpoints to `results/runs_simplified.json` every 8 tasks, so you can
interrupt and resume:

```bash
cd code/
python3 run_experiments.py
```

Once the grid is complete, dump every statistical number that the thesis
tables reference:

```bash
python3 analyse_results.py
```

Or open `imbalance_experiments.ipynb` in JupyterLab and run all cells: the
notebook re-uses the same `results/runs_simplified.json` cache, regenerates
every Case-Study-2 figure into `code/figures/`, and runs the SVM-RBF and
trainable-VQC ablations of Table case2-ablation.

### 5. Reproduce Case Study 1

Case Study 1 has two parts.

**(a) Validation against a published reference.** Reproduce the test
accuracy of a data-reuploading VQC on the `corners` and `circles`
datasets of Rodriguez-Grasa, Ban & Sanz (arXiv:2401.04642) as a function
of layers and qubits, next to that paper's published numbers, and print
the LaTeX tables that justify the 2-qubit / few-layer choice carried into
Case Study 2:

```bash
cd code/
python3 run_validation.py          # accuracy sweep -> results/case1_validation.json
python3 make_case1_validation.py   # figures/case1-val-*.png + LaTeX tables
```

**(b) Autodiff vs parameter-shift, plus Bloch diagnostics.** Open the
notebook and run all cells top to bottom (~5 min). It is self-contained,
uses `make_moons(noise=0.1)` directly, and writes its figures into
`code/figures/`:

```bash
jupyter lab VQC_Autodiff_and_ParameterShift.ipynb
```

### 6. Reproduce the dataset-robustness annex (Annex C)

`code/annex_datasets/` re-runs the **identical** Case Study 2 grid on
four extra geometric datasets (`corners`, `circles`, `spiral`, `sinus`)
to test whether the conclusion transfers off `make_moons`. The folder
has its own README; in short:

```bash
cd code/annex_datasets/
N_JOBS=12 python3 run_annex.py     # ~25-40 min on 12-16 cores; checkpointed
jupyter lab summary.ipynb          # cross-dataset figures + advantage table
```

---

## Reproducibility guarantees

* Every random source in `qml_core.py` and `qml_imbalance.py` is keyed
  on an explicit seed. The same `(model, ratio, seed)` tuple at a fixed
  noise level produces a bit-identical result regardless of execution
  order or backend.
* The classical baselines wrap scikit-learn estimators with
  `random_state=seed`, so they are deterministic too.
* The pre-registered Case Study 2 grid uses **12 seeds** specifically
  so that the Friedman+Wilcoxon+Holm protocol can certify pairs rather
  than being capped by the discrete Wilcoxon distribution at the seed
  count.

---

## What the framework does, in 30 seconds

```python
from qml_imbalance import (
    QuantumModel, ClassicalModel, run_one,
    make_balanced_dataset, friedman_wilcoxon_holm,
)

# Generate two datasets of contrasting difficulty
X1, y1 = make_balanced_dataset(n_total=8000, noise=0.1, seed=42)
X3, y3 = make_balanced_dataset(n_total=8000, noise=0.3, seed=42)

# Run the same model on both, under heavy imbalance
for X, y in [(X1, y1), (X3, y3)]:
    for name in ['trainable_vqc', 'svm_rbf']:
        model = (QuantumModel(name, n_iter=200)
                 if name in QuantumModel.SPEC
                 else ClassicalModel(name))
        result = run_one(model, ratio=0.99, seed=0,
                         X_base=X, y_base=y)
        print(f'{name:<16} BA={result.metrics["balanced_accuracy"]:.3f}')

# The friedman_wilcoxon_holm helper consumes a list of RunResults and
# returns the pre-registered statistical result per imbalance ratio.
```

The point of the uniform `(QuantumModel | ClassicalModel) → run_one`
interface is that any quantum-vs-classical comparison reduces to running
the same loop over both. See Chapter 2 of the thesis for the full design.

---

## How to cite

If this framework is useful in your work, please cite the underlying
Master's Thesis (BibTeX entry after defence date):

```bibtex
@mastersthesis{oriola2026qml,
  author = {Oriola D{\'i}az, Sergio},
  title  = {Design of an Evaluation Framework for Hybrid
            Quantum-Classical Machine Learning},
  school = {Universidad Polit{\'e}cnica de Madrid,
            ETSI de Telecomunicaci{\'o}n},
  year   = {2026},
}
```

---

## License

This project is released under the MIT License — see `LICENSE` for the
full text. All upstream dependencies are under permissive licences
(BSD-3-Clause, Apache-2.0, MIT, PSF), so the project as a whole is freely
reusable and modifiable. See Annex A of the thesis for the licence
inventory and the reasoning behind the choice.
