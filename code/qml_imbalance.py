"""
qml_imbalance.py
================
QML-vs-classical imbalanced-classification experiment library. Domain-specific
layer that sits on top of `qml_core` (quantum primitives + parameter-shift
training) and adds:

    Data
    ----
    make_balanced_dataset(n_total, noise, seed)
    make_imbalanced_split(X, y, train_ratio_majority, n_train, n_val, n_test, seed)
    rescale_to_pi(X_train, X_val, X_test)
    labels_to_pm1(y)

    Imbalance handling
    ------------------
    apply_class_weight(X, y) -> (X, y, sample_weight)
        Cost-sensitive learning by per-sample weighting. Weights inversely
        proportional to class frequency (the scikit-learn `class_weight=
        "balanced"` convention). The framework intentionally supports
        only this one handler: the case studies (Chapters 4 and 5 of the thesis)
        compare quantum and classical models *as classifiers*, holding the
        imbalance-handling strategy fixed so that any observed gap is
        attributable to the model and not to the handler.

    Models (uniform interface for quantum and classical)
    ----------------------------------------------------
    QuantumModel('fixed_vqc' | 'trainable_vqc' | 'angle_vqc' |
                 'data_reuploading' | 'reuploading_2q' | 'no_reuploading',
                 n_iter=200, lr=0.015)
    ClassicalModel('logreg' | 'svm_rbf' | 'svm_linear' | 'knn' | 'rf' | 'mlp_small')

    Each model M exposes:
        M.name, M.kind ('quantum' or 'classical'), M.n_params
        M.fit(X_train, y_train, sample_weight, X_val, y_val, seed) -> state
        M.predict_score(state, X) -> ndarray
        M.eval_count(state) -> int  (circuit evals for quantum, 0 for classical)

    Evaluation
    ----------
    tune_threshold(scores_val, y_val)
        Sweep a threshold that maximises balanced accuracy on validation.
    compute_metrics(y_true, scores, threshold) -> dict
        Primary metric: balanced_accuracy. The threshold-independent ROC AUC
        and the four confusion-matrix counts (tn, fp, fn, tp) are also returned
        as diagnostic free byproducts; they feed the failure-mode and
        robustness analyses of Case Study II (Chapter 5) but do not enter the pre-registered
        conclusion.

    Experiment runner
    -----------------
    run_one(model, ratio, seed, X_base, y_base, ...) -> RunResult
    aggregate(results, by=('model', 'ratio')) -> list of dicts

    Paired statistical tests across models
    --------------------------------------
    friedman_wilcoxon_holm(results, ratio=None)
        Per imbalance level: Friedman omnibus + Wilcoxon signed-rank pairwise
        with Holm correction for multiple comparisons.

    Production grid driver
    ----------------------
    GRID_CONFIG, run_grid()
        `python qml_imbalance.py` generates (or resumes) the full Case
        Study 2 grid — 2,880 runs — with joblib-parallel workers and
        periodic checkpointing to results/case2_grid.json (Section 9).

The quantum core and the training primitives (Adam, parameter-shift gradient,
multi-seed runner) live in `qml_core`; this module imports only the four it
calls (reset_counter, get_counter, fast_predict, train_ps_run).
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)

# The quantum/training primitives live in qml_core; only the ones this
# module actually calls are imported.
from qml_core import (
    reset_counter,
    get_counter,
    fast_predict,
    train_ps_run,
)


# =============================================================================
# 1. Data utilities
# =============================================================================

def make_balanced_dataset(n_total: int = 5000, noise: float = 0.1,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic two-moons dataset. Returns (X, y) with y in {0, 1}.

    The noise parameter controls how strongly the two crescents overlap:
        noise = 0.1  -- low overlap; the dataset is geometrically simple and
                        kernel methods saturate near 1.0 BA.
        noise = 0.3  -- substantial overlap; the Bayes-optimal classifier
                        sits well below 1.0 BA and no method can recover
                        the original geometry exactly.
    Both noise levels are used in Case Study II of the thesis (Chapter 5) to test whether
    the conclusions transfer from a low-overlap (toy-like) setting to a
    higher-overlap regime where the geometry is much less benign.
    """
    X, y = make_moons(n_samples=n_total, noise=noise, random_state=seed)
    return X, y


def make_imbalanced_split(X: np.ndarray, y: np.ndarray,
                          train_ratio_majority: float,
                          n_train: int = 500,
                          n_val: int = 400,
                          n_test: int = 1200,
                          seed: int = 0
                          ) -> Tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Build an (imbalanced train, balanced val, balanced test) split.

    The minority class in TRAIN is class 1. VAL and TEST are kept balanced
    50/50 so that balanced accuracy measures generalisation to the true
    (balanced) data distribution rather than to a particular test imbalance.

    Defaults match the Case Study 2 production grid (500/400/1200).

    train_ratio_majority : fraction of train belonging to class 0
        (e.g. 0.9 for a 90/10 imbalance, 0.5 for balanced).

    Validation and test are drawn BEFORE the training subset, so that at a
    fixed seed they hold the same samples at every ``train_ratio_majority``.
    An earlier version drew the training subset first, which made the
    validation and test identities shift with the ratio: a degradation curve
    then compared different evaluation samples as well as different training
    imbalances. The two sets were always balanced draws from the same pool,
    so the effect was sampling noise rather than bias (measured at 0.003-0.004
    balanced accuracy on the twelve-seed mean, against a 0.036-0.097 real
    drop), but the ratio is now the only quantity that varies along the curve.

    Drawing the training subset last also nests it across ratios: the five
    minority samples at r = 0.99 are the first five of the two hundred and
    fifty at r = 0.50, so the curve removes samples rather than resampling.
    """
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n_train_maj = int(round(n_train * train_ratio_majority))
    n_train_min = n_train - n_train_maj
    n_val_per = n_val // 2
    n_test_per = n_test // 2

    needed_0 = n_train_maj + n_val_per + n_test_per
    needed_1 = n_train_min + n_val_per + n_test_per
    if needed_0 > len(idx0) or needed_1 > len(idx1):
        raise ValueError(
            f"Not enough samples: need {needed_0} of class 0 and "
            f"{needed_1} of class 1, have {len(idx0)} and {len(idx1)}. "
            f"Generate a larger base dataset."
        )

    take0 = iter(idx0)
    take1 = iter(idx1)

    def grab(it, n):
        return np.array([next(it) for _ in range(n)])

    val_idx = np.concatenate([grab(take0, n_val_per), grab(take1, n_val_per)])
    test_idx = np.concatenate([grab(take0, n_test_per), grab(take1, n_test_per)])
    train_idx = np.concatenate([grab(take0, n_train_maj), grab(take1, n_train_min)])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def rescale_to_pi(X_train: np.ndarray,
                  X_val: np.ndarray,
                  X_test: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine-rescale each column of X to [-pi, pi] using TRAIN statistics."""
    Xmin = X_train.min(axis=0)
    Xmax = X_train.max(axis=0)
    span = (Xmax - Xmin)
    span[span == 0] = 1.0

    def f(X):
        return (2 * X - Xmax - Xmin) * np.pi / span

    return f(X_train), f(X_val), f(X_test)


def labels_to_pm1(y: np.ndarray) -> np.ndarray:
    """Map {0,1} labels to {-1,+1}. Convention: class 0 -> -1, class 1 -> +1."""
    return 2.0 * y.astype(float) - 1.0


# =============================================================================
# 2. Imbalance handling: class_weight = "balanced"
# =============================================================================
#
# The framework intentionally supports a single, fixed imbalance-handling
# strategy: cost-sensitive learning by per-sample weights inversely
# proportional to class frequency (the convention behind scikit-learn's
# class_weight="balanced"). Two reasons:
#
#  (i) the case studies (Case Study II, Chapter 5) compare quantum and classical models
#      *as classifiers* and hold the handler fixed so the comparison is
#      not confounded by handler-specific interactions;
# (ii) class_weight is the cheapest handler in any framework that uses
#      gradient-based training: it adds a multiplicative factor to each
#      residual and otherwise leaves the training pipeline unchanged.
#      No synthetic samples, no resampling, no extra training cost.

def apply_class_weight(X: np.ndarray, y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, sample_weight) with weights inverse to class frequency.

    Equivalent to scikit-learn's class_weight="balanced":
        w_c = n / (|C| * n_c)
    so that the average weight is 1 and minority samples receive heavier
    weight proportional to the imbalance ratio. The data are not modified;
    only the weight vector is attached.
    """
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    weights = {int(c): n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sw = np.array([weights[int(c)] for c in y], dtype=float)
    return X, y, sw


# =============================================================================
# 3. Quantum model wrapper
# =============================================================================

@dataclass
class QuantumState:
    params: np.ndarray
    circuit_type: str
    history: dict
    n_circuit_evals: int


class QuantumModel:
    """Uniform interface for the six quantum architectures.

    The first three are the core thesis architectures; the last three
    are controlled contrasts, each isolating one design choice: a plain
    (non-dense) angle-encoding variant of the Fixed VQC, a two-qubit
    entangled data-reuploading circuit, and a single-qubit
    control that is identical to the Data Reuploading model except that
    the data are not re-injected in each layer.

    n_iter and lr default to the values used throughout the thesis
    (Section 3.6, the training module); override per experiment if needed.
    """

    SPEC = {
        'fixed_vqc':        dict(circuit_type='fixed',     n_params=16),
        'trainable_vqc':    dict(circuit_type='trainable', n_params=20),
        'data_reuploading': dict(circuit_type='dr',        n_params=6),
        'angle_vqc':        dict(circuit_type='angle',     n_params=16),
        'reuploading_2q':   dict(circuit_type='dr2q',      n_params=16),
        'no_reuploading':   dict(circuit_type='nodr',      n_params=6),
    }

    def __init__(self, name: str, n_iter: int = 200, lr: float = 0.015):
        if name not in self.SPEC:
            raise ValueError(f"unknown quantum model: {name}")
        self.name = name
        self.kind = 'quantum'
        spec = self.SPEC[name]
        self.circuit_type = spec['circuit_type']
        self.n_params = spec['n_params']
        self.n_iter = n_iter
        self.lr = lr

    def _init_params(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        if self.name in ('data_reuploading', 'no_reuploading'):
            # Same initialisation for the reuploading model and its
            # no-reuploading control, so the ablation is initialisation-matched.
            return rng.normal(0.0, 1.0, size=self.n_params) * np.pi
        elif self.name == 'trainable_vqc':
            # Encoding weights at 1.0, variational block ~ pi*U[0,1]
            params = np.empty(self.n_params)
            params[:4] = 1.0
            params[4:] = np.pi * rng.rand(self.n_params - 4)
            return params
        else:
            return np.pi * rng.rand(self.n_params)

    def fit(self, X_train, y_train, sample_weight, X_val, y_val,
            seed: int = 0) -> QuantumState:
        params_init = self._init_params(seed)
        # Quantum forwards expect labels in {-1, +1}.
        y_tr_pm = labels_to_pm1(y_train)
        y_va_pm = labels_to_pm1(y_val)
        reset_counter()
        h = train_ps_run(
            params_init, X_train, y_tr_pm, X_val, y_va_pm,
            circuit_type=self.circuit_type,
            n_iter=self.n_iter, lr=self.lr,
            sample_weight=sample_weight,
            select_by='val_loss',
        )
        return QuantumState(
            params=h['best_val_params'],
            circuit_type=self.circuit_type,
            history=h,
            n_circuit_evals=get_counter(),
        )

    def predict_score(self, state: QuantumState, X: np.ndarray) -> np.ndarray:
        """Continuous scores in [-1, +1] (the <Z_0> expectation)."""
        return fast_predict(state.params, np.asarray(X),
                            circuit_type=state.circuit_type)

    def eval_count(self, state: QuantumState) -> int:
        return state.n_circuit_evals


# =============================================================================
# 4. Classical model wrapper (uniform interface)
# =============================================================================

@dataclass
class ClassicalState:
    estimator: object
    fit_time_s: float
    weighted: bool = True   # whether the fit consumed the sample weights


class ClassicalModel:
    """Uniform interface around scikit-learn estimators.

    Score convention: `predict_score` returns a continuous score where larger
    means more positive class. The score is mapped to roughly [-1, 1] so that
    threshold tuning is comparable across models.

    The capacity-matched MLP baseline ('mlp_small', one hidden layer of 4 ReLU
    units, ~17 trainable parameters) sits between the parameter counts of the
    extremes among the six quantum architectures (the single-qubit models at
    6 params, Trainable VQC at 20 params), closest to the 16 params of the
    Fixed, Angle and two-qubit Reuploading VQCs.
    """

    SPEC = {
        'logreg':     dict(n_params=3,    factory=lambda: LogisticRegression(max_iter=2000)),
        'svm_rbf':    dict(n_params=None, factory=lambda: SVC(kernel='rbf')),
        'svm_linear': dict(n_params=3,    factory=lambda: SVC(kernel='linear')),
        'mlp_small':  dict(n_params=17,   factory=lambda: MLPClassifier(
                              hidden_layer_sizes=(4,), max_iter=2000,
                              random_state=0, solver='adam')),
        'knn':        dict(n_params=None, factory=lambda: KNeighborsClassifier(n_neighbors=5)),
        'rf':         dict(n_params=None, factory=lambda: RandomForestClassifier(
                              n_estimators=100, random_state=0, n_jobs=1)),
    }

    def __init__(self, name: str):
        if name not in self.SPEC:
            raise ValueError(f"unknown classical model: {name}")
        self.name = name
        self.kind = 'classical'
        self.n_params = self.SPEC[name]['n_params'] or 0

    def _make(self, seed: int):
        est = self.SPEC[self.name]['factory']()
        for attr in ('random_state',):
            if hasattr(est, attr):
                try:
                    setattr(est, attr, seed)
                except Exception:
                    pass
        return est

    @staticmethod
    def _accepts_sample_weight(est) -> bool:
        """Whether this estimator's `fit` takes a `sample_weight` argument.

        Checked explicitly rather than by catching TypeError, because a silent
        fallback hides which baselines were actually weighted. Support is not
        uniform across scikit-learn: KNeighborsClassifier has never accepted
        sample weights, and MLPClassifier only gained them in a recent release,
        which is why `requirements.txt` pins a lower bound on the version.
        """
        return 'sample_weight' in inspect.signature(est.fit).parameters

    def fit(self, X_train, y_train, sample_weight, X_val, y_val,
            seed: int = 0) -> ClassicalState:
        est = self._make(seed)
        weighted = self._accepts_sample_weight(est)
        t0 = time.time()
        if weighted:
            est.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            est.fit(X_train, y_train)
        return ClassicalState(estimator=est, fit_time_s=time.time() - t0,
                              weighted=weighted)

    def predict_score(self, state: ClassicalState, X: np.ndarray) -> np.ndarray:
        est = state.estimator
        # Prefer decision_function, then proba, then 0/1 prediction.
        if hasattr(est, 'decision_function'):
            s = est.decision_function(X)
            return np.tanh(s)
        if hasattr(est, 'predict_proba'):
            p = est.predict_proba(X)[:, 1]
            return 2.0 * p - 1.0
        return 2.0 * est.predict(X).astype(float) - 1.0

    def eval_count(self, state: ClassicalState) -> int:
        return 0


# =============================================================================
# 5. Threshold tuning + metrics
# =============================================================================

def tune_threshold(scores_val: np.ndarray, y_val: np.ndarray,
                   n_grid: int = 201) -> Tuple[float, float]:
    """Sweep a threshold over the score range; return (best_threshold, best_BA).

    The objective is balanced accuracy on the validation set (the only metric
    the framework optimises against; see Section 3.8, the inference module). y_val must be
    in {0, 1}.
    """
    s_min, s_max = float(np.min(scores_val)), float(np.max(scores_val))
    pad = 0.05 * (s_max - s_min + 1e-12)
    grid = np.linspace(s_min - pad, s_max + pad, n_grid)
    best_t = 0.0
    best_v = -np.inf
    for t in grid:
        preds = (scores_val >= t).astype(int)
        v = balanced_accuracy_score(y_val, preds)
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t, float(best_v)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float
                    ) -> Dict[str, float]:
    """Primary metric (balanced accuracy) plus confusion-matrix counts and
    a threshold-independent ranking diagnostic (ROC AUC).

    Balanced accuracy is the single primary metric of the thesis (Section 3.8).
    The confusion-matrix counts (tn, fp, fn, tp) are retained as
    free byproducts that feed the failure-mode discussion of Chapter 5;
    they do not enter the pre-registered conclusion.

    ROC AUC is computed on the *continuous* score (before thresholding) and is
    likewise a free byproduct: it measures how well the model ranks positive
    test points above negative ones, independently of any decision threshold.
    It does not enter the pre-registered conclusion either; it is reported in
    Chapter 5 as a threshold-independent robustness check on the BA-based
    conclusions (the threshold tuner is shown there to drive a large share of
    the primary BA at extreme imbalance, so a metric that removes thresholding
    from the picture is the natural control). The test set is always balanced,
    so both classes are present and the AUC is always well defined.
    """
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        'balanced_accuracy': balanced_accuracy_score(y_true, preds),
        'roc_auc':            float(roc_auc_score(y_true, scores)),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
        'threshold':          float(threshold),
    }


# =============================================================================
# 6. Experiment runner
# =============================================================================

@dataclass
class RunResult:
    model: str
    ratio: float
    seed: int
    metrics: Dict[str, float]
    threshold: float
    fit_time_s: float
    n_circuit_evals: int
    n_train_majority: int
    n_train_minority: int


def run_one(model, ratio: float, seed: int,
            X_base: np.ndarray, y_base: np.ndarray,
            n_train: int = 500, n_val: int = 400, n_test: int = 1200,
            verbose: bool = False) -> RunResult:
    """One (model, ratio, seed) experiment with class_weight handling.

    Steps:
      1. Carve an imbalanced train + balanced val/test from X_base.
      2. Attach class_weight sample weights to TRAIN.
      3. Rescale X to [-pi, pi] using TRAIN stats, for every model.
      4. Fit the model.
      5. Score val and test; tune the decision threshold on val by balanced
         accuracy.
      6. Compute the metric panel on TEST.

    Step 3 applies to quantum and classical models alike, so that every model
    in a comparison sees exactly the same inputs. An earlier version of this
    function rescaled for quantum models only, on the grounds that the step is
    part of an angle-encoded circuit's definition while for a scikit-learn
    estimator it is optional. That is defensible in isolation but it left one
    step unshared between the two arms of the comparison, and measurement
    showed it was not harmless: five of the six classical baselines are
    insensitive to it (the random forest exactly so, since the map is monotone
    per feature and tree splits are invariant under it), but the
    capacity-matched MLP gained 2.6 to 4.1 BA points from it, which is the one
    baseline where a preprocessing handicap is least acceptable. Sharing the
    step removes the asymmetry rather than arguing about it.
    """
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_imbalanced_split(
        X_base, y_base, train_ratio_majority=ratio,
        n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
    )
    X_tr_h, y_tr_h, sw_tr_h = apply_class_weight(X_tr, y_tr)
    X_tr_use, X_va_use, X_te_use = rescale_to_pi(X_tr_h, X_va, X_te)

    n_maj = int(np.sum(y_tr == 0))
    n_min = int(np.sum(y_tr == 1))

    t0 = time.time()
    state = model.fit(X_tr_use, y_tr_h, sw_tr_h, X_va_use, y_va, seed=seed)
    fit_time = time.time() - t0

    scores_val = model.predict_score(state, X_va_use)
    threshold, _ = tune_threshold(scores_val, y_va)

    scores_test = model.predict_score(state, X_te_use)
    metrics = compute_metrics(y_te, scores_test, threshold=threshold)

    if verbose:
        print(f"  [{model.name:<16} | r={ratio:.2f} | seed={seed}] "
              f"BA={metrics['balanced_accuracy']:.3f}  "
              f"t={threshold:+.2f}  fit={fit_time:.1f}s")

    return RunResult(
        model=model.name, ratio=ratio, seed=seed,
        metrics=metrics, threshold=threshold, fit_time_s=fit_time,
        n_circuit_evals=model.eval_count(state),
        n_train_majority=n_maj, n_train_minority=n_min,
    )


# =============================================================================
# 7. Aggregation helpers (no pandas; small + dependency-free)
# =============================================================================

def aggregate(results: List[RunResult],
              by: Tuple[str, ...] = ('model', 'ratio'),
              metric: str = 'balanced_accuracy',
              ) -> List[Dict]:
    """Group results by `by` and report mean/std of `metric` over the seeds.

    Balanced accuracy is the primary aggregated metric (the default). The
    `metric` argument allows the same aggregation to run on the
    threshold-independent ROC AUC byproduct ('roc_auc') for the robustness
    check of Chapter 5. The aggregated columns are always named
    ``balanced_accuracy_mean`` / ``balanced_accuracy_std`` regardless of the
    metric requested, so that the plotting and table helpers downstream do not
    need to branch on the metric name. The confusion-matrix counts
    (tn/fp/fn/tp) are intentionally not aggregated here; they live in the raw
    RunResults and are used by the failure-mode analysis on a per-seed basis.
    """
    groups: Dict[Tuple, List[RunResult]] = {}
    for r in results:
        key = tuple(getattr(r, k) for k in by)
        groups.setdefault(key, []).append(r)

    out = []
    for key, rs in sorted(groups.items()):
        row = dict(zip(by, key))
        row['n_seeds'] = len(rs)
        vals = np.array([r.metrics[metric] for r in rs], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            row['balanced_accuracy_mean'] = float('nan')
            row['balanced_accuracy_std'] = float('nan')
        else:
            row['balanced_accuracy_mean'] = float(np.mean(vals))
            row['balanced_accuracy_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row['fit_time_mean'] = float(np.mean([r.fit_time_s for r in rs]))
        row['n_circuit_evals_mean'] = float(np.mean([r.n_circuit_evals for r in rs]))
        out.append(row)
    return out


# =============================================================================
# 8. Paired statistical tests across models
# =============================================================================
#
# Rationale. The experimental design produces paired data because every model
# is trained and tested under exactly the same seed, and the seed simultaneously
# controls the data split and the parameter initialisation. Two models evaluated
# at the same (ratio, seed) therefore see *identical* training data, *identical*
# test data, and any difference between their balanced-accuracy scores reflects
# the model alone.
#
# Under this paired structure, the protocol recommended by Demsar (2006) and
# Garcia & Herrera (2008) for comparing k > 2 classifiers on the same data is:
#
#   1. Friedman omnibus test [Friedman 1937] at significance level alpha:
#        H_0 : all algorithms perform equivalently at this imbalance level.
#      Rejection (p < alpha) provides evidence that at least one pair differs.
#
#   2. If Friedman is significant, run pairwise Wilcoxon signed-rank tests
#      [Wilcoxon 1945] on every model pair, then apply Holm's step-down
#      correction [Holm 1979] to control the family-wise error rate over the
#      k(k-1)/2 comparisons. Holm is preferred over Bonferroni because it has
#      identical FWER control but uniformly greater power.
#
# Note on minimum seeds. The discrete distribution of the Wilcoxon signed-rank
# test has a smallest two-sided p-value of 2 / 2^n at n seeds. With k = 12
# models the family contains k(k-1)/2 = 66 pairs, and at n = 10 the smallest
# Holm-adjusted p is 66 * 0.00195 = 0.129, *above* alpha = 0.05. The
# thesis therefore uses n >= 12 seeds, which lowers the floor to
# 66 * 2 / 2^12 = 0.0322 < 0.05 and allows the protocol to certify pairs.
# (With the original k = 8 model set the floor was 28 * 2 / 2^12 = 0.0137.)
#
# This module implements all three tests using only scipy (no statsmodels
# dependency); Holm's step-down adjustment is short enough to implement directly.

# Decimal places at which paired metric differences are rounded before the
# signed-rank test, so that two models scoring the same value through
# different confusion matrices register as an exact tie rather than as a
# ~1e-16 difference. See the comment in `friedman_wilcoxon_holm`.
TIE_DECIMALS = 10


def _holm_adjust(pvals: List[float]) -> List[float]:
    """Holm's step-down family-wise error correction.

    Given m raw two-sided p-values, returns m Holm-adjusted p-values such
    that rejecting p_adj < alpha controls the family-wise error rate at
    level alpha. Equivalent to statsmodels.stats.multitest.multipletests
    with method='holm', re-implemented to avoid the dependency.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    factors = np.arange(m, 0, -1)
    raw = sorted_p * factors
    adj_sorted = np.maximum.accumulate(np.minimum(raw, 1.0))
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj.tolist()


def friedman_wilcoxon_holm(
    results: List[RunResult],
    *,
    ratio: Optional[float] = None,
    alpha: float = 0.05,
    metric: str = 'balanced_accuracy',
) -> Dict:
    """Paired statistical comparison of models at a single imbalance level.

    Balanced accuracy is the primary metric (the default), consistent with the
    rest of the framework. The `metric` argument allows the identical paired
    protocol (Friedman omnibus + Holm-corrected pairwise Wilcoxon) to be run on
    the threshold-independent ROC AUC byproduct ('roc_auc') as the robustness
    check of Chapter 5; the AUC run does not enter the pre-registered conclusion.

    Parameters
    ----------
    results
        Full list of RunResult records.
    ratio
        Imbalance ratio to analyse. If None, analyses every ratio in the
        data and returns a dict keyed by ratio.
    alpha
        Significance level (default 0.05).
    metric
        Per-run metric key to compare ('balanced_accuracy' or 'roc_auc').

    Returns
    -------
    dict
        For each ratio analysed:
            'ratio'         : the imbalance ratio
            'models'        : list of model names (column order)
            'seeds'         : list of seeds (row order)
            'matrix'        : (n_seeds x n_models) array of BA values
            'friedman_stat' : Friedman chi-square statistic
            'friedman_p'    : Friedman p-value
            'pairwise'      : dict {(model_i, model_j): {
                                'wilcoxon_stat', 'wilcoxon_p_raw',
                                'wilcoxon_p_holm', 'mean_diff',
                                'reject_holm' (bool)}}
    """
    from scipy.stats import friedmanchisquare, wilcoxon
    from itertools import combinations

    if ratio is None:
        all_ratios = sorted({r.ratio for r in results})
        return {
            r: friedman_wilcoxon_holm(results, ratio=r, alpha=alpha,
                                      metric=metric)
            for r in all_ratios
        }

    at_ratio = [r for r in results if r.ratio == ratio]
    models = sorted({r.model for r in at_ratio})
    seeds = sorted({r.seed for r in at_ratio})
    matrix = np.full((len(seeds), len(models)), np.nan, dtype=float)
    for j, m in enumerate(models):
        for i, s in enumerate(seeds):
            sub = [r for r in at_ratio if r.model == m and r.seed == s]
            if sub:
                matrix[i, j] = sub[0].metrics[metric]

    complete = ~np.any(np.isnan(matrix), axis=1)
    M = matrix[complete]
    used_seeds = [s for s, ok in zip(seeds, complete) if ok]

    out: Dict = {
        'ratio': ratio,
        'models': models,
        'seeds': used_seeds,
        'matrix': M,
    }

    if M.shape[0] < 2 or M.shape[1] < 3:
        out['friedman_stat'] = float('nan')
        out['friedman_p']    = float('nan')
        out['pairwise']      = {}
        return out
    try:
        chi2, p_fried = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
        out['friedman_stat'] = float(chi2)
        out['friedman_p']    = float(p_fried)
    except ValueError:
        out['friedman_stat'] = float('nan')
        out['friedman_p']    = float('nan')

    pairs    = list(combinations(range(len(models)), 2))
    raw_ps   = []
    diffs    = []
    stats_w  = []
    for i, j in pairs:
        a, b = M[:, i], M[:, j]
        # Form the paired differences explicitly and round them to
        # TIE_DECIMALS before the test. Two models can reach the same
        # balanced accuracy through different confusion matrices, in which
        # case the two float paths agree only to machine precision and the
        # difference comes out at ~1e-16 instead of exactly zero. Passing the
        # raw arrays lets scipy count such a pair as a genuine non-zero
        # observation, which changes the signed ranks and, through Holm, the
        # decision. The metric resolution here is 1/n_test, so rounding at
        # ten decimals removes the artefact with a margin of many orders of
        # magnitude and cannot merge two truly distinct values.
        d = np.round(a - b, TIE_DECIMALS)
        if np.all(d == 0):
            w_stat, w_p = float('nan'), 1.0
        else:
            try:
                w_stat, w_p = wilcoxon(d, zero_method='wilcox',
                                       alternative='two-sided')
            except ValueError:
                w_stat, w_p = float('nan'), float('nan')
        raw_ps.append(w_p if not np.isnan(w_p) else 1.0)
        diffs.append(float(np.mean(a) - np.mean(b)))
        stats_w.append(float(w_stat) if not np.isnan(w_stat) else float('nan'))

    holm = _holm_adjust(raw_ps)
    pairwise = {}
    for (i, j), w_stat, p_raw, p_h, d in zip(pairs, stats_w, raw_ps, holm, diffs):
        pairwise[(models[i], models[j])] = {
            'wilcoxon_stat':  w_stat,
            'wilcoxon_p_raw': float(p_raw),
            'wilcoxon_p_holm': float(p_h),
            'mean_diff':      d,
            'reject_holm':    bool(p_h < alpha),
        }
    out['pairwise'] = pairwise
    return out


def print_friedman_summary(stats: Dict, alpha: float = 0.05) -> None:
    """Pretty-print the output of friedman_wilcoxon_holm for one ratio."""
    print(f"=== Ratio r = {stats['ratio']:.2f}  (metric = balanced_accuracy) ===")
    print(f"Seeds used : {stats['seeds']}")
    print(f"Models (k = {len(stats['models'])}) : {', '.join(stats['models'])}")
    print(f"\nFriedman chi^2 = {stats['friedman_stat']:.3f}   "
          f"p = {stats['friedman_p']:.4g}   "
          f"({'reject H0' if stats['friedman_p'] < alpha else 'do not reject H0'})")
    if stats['friedman_p'] >= alpha or np.isnan(stats['friedman_p']):
        print("Friedman not significant -> pairwise tests are not interpreted.")
        return
    print("\nPairwise Wilcoxon signed-rank with Holm correction:")
    print(f"  {'model A':<22s} {'model B':<22s} {'diff (A-B)':>11s} "
          f"{'p_raw':>9s} {'p_holm':>9s} sig")
    rows = sorted(stats['pairwise'].items(),
                  key=lambda kv: kv[1]['wilcoxon_p_holm'])
    for (a, b), d in rows:
        flag = '*' if d['reject_holm'] else ''
        print(f"  {a:<22s} {b:<22s} "
              f"{d['mean_diff']:>+11.4f} "
              f"{d['wilcoxon_p_raw']:>9.4g} "
              f"{d['wilcoxon_p_holm']:>9.4g} {flag}")
    print("  (* = Holm-adjusted p < alpha, i.e. the median of paired "
          "differences is significantly different from zero after FWER control.)")


# =============================================================================
# 9. Production grid driver:  python qml_imbalance.py
# =============================================================================
#
# Generates the experimental data for Case Study 2 of the thesis: 2 noise
# levels x 12 models x 10 ratios x 12 seeds = 2,880 runs, with joblib-parallel
# task execution and periodic JSON checkpointing to results/case2_grid.json.
#
# Parallelism. Tasks are independent (each is a fully self-contained
# ``run_one`` call keyed on (model, ratio, seed)), so the driver fans them
# out to ``joblib.Parallel`` workers using the loky process-based backend
# (threading would be serialised by Python's GIL on the NumPy-bound
# forwards). Every random source inside ``run_one`` is keyed on
# (seed, ratio), so per-task results are bit-identical regardless of the
# worker count and of completion order. Set the workers with N_JOBS:
#
#     N_JOBS=1  python qml_imbalance.py   # sequential (default, reference)
#     N_JOBS=-1 python qml_imbalance.py   # joblib auto = all cores
#
# Checkpointing. results/case2_grid.json is rewritten atomically after every
# chunk of completed tasks and re-read on startup; already-computed
# (noise, model, ratio, seed) tuples are skipped, so the full grid resumes
# after any interruption.

GRID_CONFIG = {
    'noise_levels': [0.1, 0.3],
    'n_total_base': 8000,
    'n_train': 500, 'n_val': 400, 'n_test': 1200,
    # Ratios are log-spaced in the *minority count*, which is the variable
    # that actually drives the degradation: 250, 200, 150, 100, 50, 35, 25,
    # 20, 10, 5 minority training samples. The three points above 0.90 were
    # added after the first grid showed that ~92% of the total drop happens
    # between r=0.90 and r=0.99, so the original seven sampled the flat
    # region five times and the informative region twice.
    'ratios': [0.50, 0.60, 0.70, 0.80, 0.90, 0.93, 0.95, 0.96, 0.98, 0.99],
    'seeds': list(range(12)),  # 12 seeds keeps the Wilcoxon+Holm floor < alpha
    'quantum_iter': 200,
    'quantum_models': ['data_reuploading', 'no_reuploading', 'fixed_vqc',
                       'angle_vqc', 'reuploading_2q', 'trainable_vqc'],
    'classical_models': ['logreg', 'svm_rbf', 'svm_linear', 'knn', 'rf', 'mlp_small'],
    'handler': 'class_weight',
    'tune_metric': 'balanced_accuracy',
}


def _grid_task(noise, mname, ratio, seed):
    """One (noise, model, ratio, seed) experiment as a serialisable dict.

    Regenerates the base dataset inside the worker rather than receiving it
    as an argument: ``make_balanced_dataset`` is sub-millisecond, which
    beats repickling a 16k-float array on every job dispatch.
    """
    c = GRID_CONFIG
    X, y = make_balanced_dataset(n_total=c['n_total_base'], noise=noise, seed=42)
    model = (QuantumModel(mname, n_iter=c['quantum_iter'])
             if mname in QuantumModel.SPEC
             else ClassicalModel(mname))
    r = run_one(model, ratio=ratio, seed=seed, X_base=X, y_base=y,
                n_train=c['n_train'], n_val=c['n_val'], n_test=c['n_test'])
    return {
        'noise': noise,
        'model': r.model, 'ratio': r.ratio, 'seed': r.seed,
        'metrics': r.metrics, 'threshold': r.threshold,
        'fit_time_s': r.fit_time_s, 'n_circuit_evals': r.n_circuit_evals,
        'n_train_majority': r.n_train_majority,
        'n_train_minority': r.n_train_minority,
    }


def run_grid(out_path: Optional[str] = None,
             chunk_size: int = 8,
             n_jobs: Optional[int] = None) -> None:
    """Run (or resume) the full Case Study 2 grid with checkpointing."""
    import os
    import json
    from joblib import Parallel, delayed

    c = GRID_CONFIG
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'results', 'case2_grid.json')
    if n_jobs is None:
        n_jobs = int(os.environ.get('N_JOBS', '1'))

    all_runs: List[dict] = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_runs.extend(json.load(f).get('runs', []))
        print(f'Resumed: loaded {len(all_runs)} runs from {out_path}')
    done = {(r['noise'], r['model'], r['ratio'], r['seed']) for r in all_runs}

    # Fast models first, so an interrupted grid is still interpretable.
    fast_first = (['data_reuploading', 'no_reuploading'] + c['classical_models']
                  + ['fixed_vqc', 'angle_vqc', 'reuploading_2q',
                     'trainable_vqc'])
    tasks = [(noise, m, ratio, seed)
             for noise in c['noise_levels'] for m in fast_first
             for ratio in c['ratios'] for seed in c['seeds']
             if (noise, m, ratio, seed) not in done]

    print(f'Already done : {len(done)}')
    print(f'Remaining    : {len(tasks)}')
    print(f'Workers      : N_JOBS={n_jobs}'
          + ('  [sequential, bit-identical to reference]' if n_jobs == 1 else ''))
    if not tasks:
        print('All tasks done; nothing to do.')
        return

    def save_checkpoint(attempts: int = 5, pause: float = 1.0):
        """Write the checkpoint atomically, retrying the final rename.

        On Windows ``os.replace`` raises PermissionError if any other process
        holds the destination open (an editor, a reader, an antivirus scan).
        Losing a long grid to that would be needless, so the rename is retried
        a few times before giving up.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path + '.tmp', 'w') as f:
            json.dump({'config': GRID_CONFIG, 'runs': all_runs},
                      f, indent=1, default=float)
        for attempt in range(attempts):
            try:
                os.replace(out_path + '.tmp', out_path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(pause)

    t_start = time.time()
    n_done = 0
    pool = Parallel(n_jobs=n_jobs, backend='loky', batch_size=1)
    for start in range(0, len(tasks), chunk_size):
        chunk = tasks[start:start + chunk_size]
        results = pool(delayed(_grid_task)(*t) for t in chunk)
        all_runs.extend(results)
        save_checkpoint()
        n_done += len(results)
        elapsed = time.time() - t_start
        eta = elapsed / n_done * (len(tasks) - n_done)
        last = results[-1]
        print(f'  [{n_done:4d}/{len(tasks)}] noise={last["noise"]} '
              f'{last["model"]:<18s} r={last["ratio"]:.2f} seed={last["seed"]:>2d}  '
              f'BA={last["metrics"]["balanced_accuracy"]:.3f}  '
              f'(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)  '
              f'[checkpoint -> {out_path}]')

    print(f'\nDone. Generated {len(tasks)} new runs in '
          f'{(time.time() - t_start)/60:.1f} min.')
    print(f'Total runs (cached + new): {len(all_runs)} -> {out_path}')


if __name__ == '__main__':
    run_grid()
