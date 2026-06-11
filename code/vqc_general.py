"""
vqc_general.py
==============
Generalised Variational Quantum Classifier for the Case Study 1 *validation*
against Rodriguez-Grasa, Ban & Sanz, "Neural quantum kernels: training quantum
kernels with quantum neural networks" (arXiv:2401.04642v2).

Why this module exists
----------------------
``qml_core`` hard-codes its three architectures to a fixed number of qubits and
layers (Fixed VQC: 2 qubits / 4 layers, Data Reuploading: 1 qubit / 3 layers,
...). The Case Study 1 validation requires sweeping the classifier over a grid
of ``(n_qubits, n_layers)`` so the resulting test accuracy can be compared with
the reference paper's Tables III and IV and used to *select* the number of
qubits and layers carried forward into Case Study 2. This module provides that
generalisation as a small, self-contained, pure-NumPy state-vector simulator
plus a parameter-shift training loop, deliberately kept separate from
``qml_core`` so the byte-identical reproducibility of the Case Study 2 grid is
untouched.

Architecture (data re-uploading hardware-efficient ansatz)
----------------------------------------------------------
For ``n`` qubits and ``L`` layers, starting from |0...0>, each layer applies:

    1. Encoding  U_enc(x) = [ RY(x_0) RZ(x_1) ] on every qubit
       (applied at *every* layer if ``reupload=True``, i.e. data re-uploading,
        as in the reference; only at the first layer if ``reupload=False``,
        i.e. the single-encoding VQC used by the main models of Case Study 2).
    2. Entangling  a linear chain of CZ gates  CZ(0,1) CZ(1,2) ... CZ(n-2,n-1)
       (skipped for n = 1). The reference uses CNOT or CZ and reports the two
       to be nearly interchangeable (its Table I); CZ matches the rest of the
       framework.
    3. Trainable rotation  U(theta) on every qubit. With ``rot='su2'`` this is
       the general single-qubit rotation RZ(a) RY(b) RZ(c) (3 parameters per
       qubit per layer); with ``rot='ry_rz'`` it is RY(a) RZ(b) (2 parameters).

The binary label comes from the sign of the expectation value <Z_0> of the
Pauli-Z observable on the first qubit, exactly as in ``qml_core``; the squared
loss compares <Z_0> in [-1, 1] against labels in {-1, +1}.

Conventions (identical to qml_core)
-----------------------------------
    * Big-endian state ordering: basis index = sum_q bit_q * 2^(n-1-q), so the
      reshape (B,) + (2,)*n places qubit q on axis (1 + q) and qubit 0 is the
      most significant bit. This makes ``qml_core.fast_z0`` (sum of the first
      half minus the second half of the probabilities) compute <Z_0> for any n.
    * RY(t) = exp(-i t Y / 2),  RZ(t) = exp(-i t Z / 2).
    * Parameter-shift rule with shift pi/2 for every trainable rotation angle
      (encoding angles carry data, not trainable parameters, and are not
      shifted).

Pure NumPy + scikit-learn (only for the balanced-accuracy / split helpers).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the gate factories, the <Z_0> reader, the optimiser and the counter
# from the framework core so the only new thing here is the n-qubit plumbing.
from qml_core import (
    _ry_batch, _rz_batch, fast_z0,
    AdamOptimizer, reset_counter, get_counter,
)


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass(frozen=True)
class VQCConfig:
    """Architecture knobs for the generalised VQC.

    n_qubits : number of qubits (>= 1)
    n_layers : number of (encode, entangle, rotate) layers L
    reupload : True  -> re-upload the data at every layer (paper's QNN);
               False -> encode once at the first layer (single-encoding VQC).
    rot      : 'su2'   -> RZ-RY-RZ trainable rotation (3 params/qubit/layer);
               'ry_rz' -> RY-RZ trainable rotation (2 params/qubit/layer).
    entangle : 'cz'  -> linear CZ chain (only entangler implemented here).
    """
    n_qubits: int
    n_layers: int
    reupload: bool = True
    rot: str = "su2"
    entangle: str = "cz"

    @property
    def rot_per_qubit(self) -> int:
        return 3 if self.rot == "su2" else 2

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_qubits * self.rot_per_qubit


def n_params(cfg: VQCConfig) -> int:
    """Number of trainable parameters for a configuration."""
    return cfg.n_params


# =============================================================================
# 2. Gate application on batched n-qubit states (B, 2**n)
# =============================================================================

def _apply_1q(state: np.ndarray, g: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a single-qubit gate to qubit ``q`` of a batch of states.

    state : (B, 2**n) complex
    g     : (2, 2)        same gate for all samples, or
            (B, 2, 2)     a per-sample gate (used for data-dependent encoding)
    """
    B = state.shape[0]
    s = state.reshape((B,) + (2,) * n)
    s = np.moveaxis(s, 1 + q, -1)              # bring qubit q to the last axis
    if g.ndim == 2:
        s = s @ g.T                            # out_i = sum_j s_j g_ij
    else:
        s = np.einsum('b...j,bij->b...i', s, g)
    s = np.moveaxis(s, -1, 1 + q)
    return s.reshape(B, 2 ** n)


def _cz_chain_diag(n: int) -> np.ndarray:
    """Diagonal of the product of CZ gates on the linear chain (0,1),...,(n-2,n-1).

    All CZ gates are diagonal in the computational basis and commute, so the
    whole entangling layer is a single elementwise phase multiply.
    """
    if n < 2:
        return np.ones(2 ** n)
    idx = np.arange(2 ** n)
    phase = np.ones(2 ** n)
    for c in range(n - 1):
        t = c + 1
        bit_c = (idx >> (n - 1 - c)) & 1
        bit_t = (idx >> (n - 1 - t)) & 1
        phase = phase * np.where((bit_c == 1) & (bit_t == 1), -1.0, 1.0)
    return phase


# =============================================================================
# 3. Circuit forward
# =============================================================================

def _apply_encoding(state: np.ndarray, X: np.ndarray, n: int) -> np.ndarray:
    """U_enc(x) = RY(x_0) RZ(x_1) on every qubit (data-dependent, batched)."""
    gy = _ry_batch(X[:, 0])                    # (B, 2, 2)
    gz = _rz_batch(X[:, 1])
    for q in range(n):
        state = _apply_1q(state, gy, q, n)
        state = _apply_1q(state, gz, q, n)
    return state


def _apply_rotations(state: np.ndarray, theta_layer: np.ndarray,
                     n: int, rot: str) -> np.ndarray:
    """Trainable single-qubit rotation on every qubit for one layer.

    theta_layer : (n * rot_per_qubit,) parameters for this layer, qubit-major.
    """
    k = 3 if rot == "su2" else 2
    for q in range(n):
        base = q * k
        if rot == "su2":
            a, b, c = theta_layer[base], theta_layer[base + 1], theta_layer[base + 2]
            state = _apply_1q(state, _rz_batch(a), q, n)
            state = _apply_1q(state, _ry_batch(b), q, n)
            state = _apply_1q(state, _rz_batch(c), q, n)
        else:
            a, b = theta_layer[base], theta_layer[base + 1]
            state = _apply_1q(state, _ry_batch(a), q, n)
            state = _apply_1q(state, _rz_batch(b), q, n)
    return state


def general_vqc_states(theta: np.ndarray, X: np.ndarray,
                       cfg: VQCConfig) -> np.ndarray:
    """Forward pass. Returns the (B, 2**n) state batch before measurement."""
    n, L = cfg.n_qubits, cfg.n_layers
    B = X.shape[0]
    state = np.zeros((B, 2 ** n), dtype=np.complex128)
    state[:, 0] = 1.0

    cz = _cz_chain_diag(n)
    per_layer = n * cfg.rot_per_qubit

    for l in range(L):
        if cfg.reupload or l == 0:
            state = _apply_encoding(state, X, n)
        if n >= 2:
            state = state * cz
        theta_layer = theta[l * per_layer:(l + 1) * per_layer]
        state = _apply_rotations(state, theta_layer, n, cfg.rot)
    return state


def general_vqc_predict(theta: np.ndarray, X: np.ndarray,
                        cfg: VQCConfig) -> np.ndarray:
    """Continuous prediction <Z_0> in [-1, 1] for a batch of inputs."""
    return fast_z0(general_vqc_states(np.asarray(theta), np.asarray(X), cfg)).real


# =============================================================================
# 4. Cost and parameter-shift gradient
# =============================================================================

def cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
         cfg: VQCConfig) -> Tuple[float, float, np.ndarray]:
    """Forward-only MSE cost. Labels in {-1, +1}. Returns (loss, acc, scores)."""
    scores = general_vqc_predict(theta, X, cfg)
    loss = float(np.mean((scores - y) ** 2))
    acc = float(np.mean(np.sign(scores) == np.sign(y)))
    return loss, acc, scores


def parameter_shift_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                             cfg: VQCConfig) -> Tuple[np.ndarray, float, float]:
    """Vectorised parameter-shift gradient of the MSE loss. Labels in {-1, +1}.

    Every trainable parameter is a rotation angle, so the analytic shift is
    pi/2 for all of them. Each parameter costs two batched forwards.
    Returns (gradients, train_loss, train_acc).
    """
    theta = np.asarray(theta, dtype=float)
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    P = len(theta)
    N = X.shape[0]

    f = general_vqc_predict(theta, X, cfg)
    residuals = f - y
    train_loss = float(np.mean(residuals ** 2))
    train_acc = float(np.mean(np.sign(f) == np.sign(y)))

    grads = np.zeros(P)
    shift = np.pi / 2.0
    for k in range(P):
        tp = theta.copy(); tp[k] += shift
        tm = theta.copy(); tm[k] -= shift
        f_plus = general_vqc_predict(tp, X, cfg)
        f_minus = general_vqc_predict(tm, X, cfg)
        df = (f_plus - f_minus) / 2.0
        grads[k] = float(np.mean(2.0 * residuals * df))
    return grads, train_loss, train_acc


# =============================================================================
# 5. Training
# =============================================================================

def _init_params(cfg: VQCConfig, seed: int,
                 init_mode: str = "pi_uniform", init_scale: float = 1.0
                 ) -> np.ndarray:
    """Initial parameter vector.

    init_mode : 'pi_uniform' -> init_scale * pi * Uniform[0,1]^P (framework default)
                'normal'      -> init_scale * Normal(0,1)^P (small-angle init)
    """
    rng = np.random.RandomState(seed)
    if init_mode == "normal":
        return init_scale * rng.randn(cfg.n_params)
    return init_scale * np.pi * rng.rand(cfg.n_params)


def train_run(cfg: VQCConfig,
              X_tr: np.ndarray, y_tr: np.ndarray,
              X_va: np.ndarray, y_va: np.ndarray,
              seed: int = 0, n_iter: int = 150, lr: float = 0.05,
              select_by: str = "val_loss",
              init_mode: str = "pi_uniform", init_scale: float = 1.0,
              batch_size: Optional[int] = None) -> Dict:
    """Single Adam + parameter-shift training run.

    Labels y_tr / y_va in {-1, +1}. Selects the best-by-validation checkpoint.
    ``batch_size=None`` is full-batch; an integer enables mini-batch SGD-Adam
    (the gradient is computed on a fresh random mini-batch each iteration,
    matching the reference paper's training protocol).
    Returns a history dict including 'best_params' and 'evals'.
    """
    params = _init_params(cfg, seed, init_mode=init_mode, init_scale=init_scale)
    opt = AdamOptimizer(learning_rate=lr)
    h: Dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_vl, best_va, best_it = np.inf, -np.inf, -1
    best_params = params.copy()
    reset_counter()
    rng = np.random.RandomState(1000 + seed)
    N = X_tr.shape[0]

    for it in range(n_iter):
        if batch_size is not None and batch_size < N:
            bi = rng.choice(N, size=batch_size, replace=False)
            Xb, yb = X_tr[bi], y_tr[bi]
        else:
            Xb, yb = X_tr, y_tr
        grads, tl, ta = parameter_shift_gradient(params, Xb, yb, cfg)
        vl, va, _ = cost(params, X_va, y_va, cfg)
        h["train_loss"].append(tl); h["train_acc"].append(ta)
        h["val_loss"].append(vl); h["val_acc"].append(va)

        better = ((select_by == "val_loss" and vl < best_vl) or
                  (select_by == "val_acc" and (va > best_va or
                                               (va == best_va and vl < best_vl))))
        if better:
            best_vl, best_va, best_it = vl, va, it
            best_params = params.copy()

        params = opt.step(params, grads)

    h["best_val_loss"] = best_vl
    h["best_val_acc"] = best_va
    h["best_val_iter"] = best_it
    h["best_params"] = best_params
    h["final_params"] = params.copy()
    h["evals"] = get_counter()
    return h


# =============================================================================
# 6. Data helpers (balanced split + [-pi, pi] rescale)
# =============================================================================

def split_balanced(X: np.ndarray, y: np.ndarray,
                   n_train: int, n_val: int, n_test: int,
                   seed: int = 0
                   ) -> Tuple[np.ndarray, ...]:
    """Stratified (balanced) train/val/test split from a balanced (X, y).

    Each split holds an equal number of class-0 and class-1 samples, so plain
    accuracy and balanced accuracy coincide on val/test (the paper reports
    plain test accuracy on its balanced toy datasets).
    """
    rng = np.random.RandomState(seed)
    idx0 = rng.permutation(np.where(y == 0)[0])
    idx1 = rng.permutation(np.where(y == 1)[0])

    def take(per_split):
        out = []
        c0 = c1 = 0
        for n_tr in per_split:
            half = n_tr // 2
            out.append((idx0[c0:c0 + half], idx1[c1:c1 + half]))
            c0 += half; c1 += half
        return out

    halves = take([n_train, n_val, n_test])
    (tr0, tr1), (va0, va1), (te0, te1) = halves

    def join(a, b):
        ii = np.concatenate([a, b]); rng.shuffle(ii)
        return X[ii], y[ii]

    Xtr, ytr = join(tr0, tr1)
    Xva, yva = join(va0, va1)
    Xte, yte = join(te0, te1)
    return Xtr, ytr, Xva, yva, Xte, yte


def rescale_to_pi(X_train: np.ndarray, *others: np.ndarray
                  ) -> Tuple[np.ndarray, ...]:
    """Affine-rescale each feature column to [-pi, pi] using TRAIN statistics."""
    Xmin = X_train.min(axis=0)
    Xmax = X_train.max(axis=0)
    span = (Xmax - Xmin); span[span == 0] = 1.0

    def f(X):
        return (2 * X - Xmax - Xmin) * np.pi / span

    return (f(X_train),) + tuple(f(X) for X in others)


def labels_to_pm1(y: np.ndarray) -> np.ndarray:
    return 2.0 * y.astype(float) - 1.0


# =============================================================================
# 7. One configuration, averaged over seeds
# =============================================================================

@dataclass
class ConfigResult:
    cfg: VQCConfig
    dataset: str
    test_acc_mean: float
    test_acc_std: float
    test_acc_best: float
    val_acc_mean: float
    per_seed_test_acc: List[float]
    n_params: int
    mean_evals: float
    mean_time_s: float


def run_config(X: np.ndarray, y: np.ndarray, cfg: VQCConfig,
               dataset: str = "",
               seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
               n_train: int = 1000, n_val: int = 200, n_test: int = 300,
               n_iter: int = 150, lr: float = 0.05,
               init_mode: str = "pi_uniform", init_scale: float = 1.0,
               batch_size: Optional[int] = None,
               verbose: bool = True) -> ConfigResult:
    """Train ``cfg`` over several seeds on a balanced (X, y); report test acc.

    For each seed the data split AND the parameter initialisation are reseeded,
    so the seed spread reflects both initialisation and split variability. The
    decision rule is plain ``sign(<Z_0>)`` (threshold 0); the test set is
    balanced so this matches the paper's reported plain test accuracy.
    """
    test_accs, val_accs, evals, times = [], [], [], []
    for s in seeds:
        Xtr, ytr, Xva, yva, Xte, yte = split_balanced(
            X, y, n_train, n_val, n_test, seed=s)
        Xtr, Xva, Xte = rescale_to_pi(Xtr, Xva, Xte)
        ytr_pm = labels_to_pm1(ytr)
        yva_pm = labels_to_pm1(yva)
        yte_pm = labels_to_pm1(yte)

        t0 = time.time()
        h = train_run(cfg, Xtr, ytr_pm, Xva, yva_pm, seed=s,
                      n_iter=n_iter, lr=lr, init_mode=init_mode,
                      init_scale=init_scale, batch_size=batch_size)
        dt = time.time() - t0

        _, te_acc, _ = cost(h["best_params"], Xte, yte_pm, cfg)
        test_accs.append(te_acc)
        val_accs.append(h["best_val_acc"])
        evals.append(h["evals"])
        times.append(dt)
        if verbose:
            print(f"  [{dataset:>7} n={cfg.n_qubits} L={cfg.n_layers} "
                  f"seed={s}] test_acc={te_acc:.3f} "
                  f"val_acc={h['best_val_acc']:.3f} ({dt:.1f}s)")

    res = ConfigResult(
        cfg=cfg, dataset=dataset,
        test_acc_mean=float(np.mean(test_accs)),
        test_acc_std=float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
        test_acc_best=float(np.max(test_accs)),
        val_acc_mean=float(np.mean(val_accs)),
        per_seed_test_acc=[float(a) for a in test_accs],
        n_params=cfg.n_params,
        mean_evals=float(np.mean(evals)),
        mean_time_s=float(np.mean(times)),
    )
    if verbose:
        print(f"  -> {dataset} n={cfg.n_qubits} L={cfg.n_layers}: "
              f"test_acc {res.test_acc_mean:.3f} +/- {res.test_acc_std:.3f} "
              f"(best {res.test_acc_best:.3f}), params={res.n_params}")
    return res
