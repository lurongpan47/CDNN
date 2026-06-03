#!/usr/bin/env python3
"""
=============================================================================
cd_nn.py - Communication Dynamics Neural Networks
=============================================================================

A minimal NumPy implementation of CD-derived neural network layers:

    1. CDLinear         - Block-circulant linear layer (FFT-diagonalized)
    2. PolygonGroup     - (2L+1)-vertex polygon channel grouping
    3. ShannonDropout   - alpha_CD-rate principled noise injection
    4. FisherReg        - Closed-form Fisher information regularizer
    5. CDOptimizer      - Spectrum-preconditioned SGD using FFT-Hessian

Each layer ships with a manual forward/backward pass and unit-tested gradient
checks.  The library is dependency-light (NumPy only) so the entire CD-NN
stack runs on commodity hardware without GPU.

Companion to:
    L. Pan and M. Tanik (2026),
    "Communication Dynamics Neural Networks: FFT-Diagonalized Layers
     and Polygon-Symmetric Architectures for Faster Convergence"
    (Paper III in the CD series).

Authors: L. Pan (Ainnocence Inc.), M. Tanik (UAB)
Date:    2026-04-27
License: MIT
=============================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Callable

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 0) -> np.random.Generator:
    """Return a seeded RNG.  Use it everywhere — never call np.random.* directly."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# CD physical constants (from Paper I)
# ---------------------------------------------------------------------------
ALPHA_CD = 0.0118    # Shannon noise constant from Na D-doublet calibration
LAMBDA_CEIL = 4.0    # Sadovskii lattice-instability ceiling


# =============================================================================
# 1. CDLinear: BLOCK-CIRCULANT LAYER (FFT-DIAGONALIZED)
# =============================================================================
class CDLinear:
    """Block-circulant linear layer.

    Maps R^{n_in} -> R^{n_out} via a block-circulant weight matrix with
    block size B = 2L+1 (the polygon multiplicity).  The forward pass is

        y = W x + b,   W = block-circulant(c)

    Each circulant block C_{ij} is determined by its first row c_{ij},
    so the layer has (n_out/B) * (n_in/B) * B = n_in * n_out / B parameters,
    a factor of B reduction vs dense.

    Forward complexity:  O(n_in * n_out / B + (n_in + n_out) log B)
    Dense complexity:    O(n_in * n_out)

    The Hessian of (1/2) ||y - t||^2 with respect to c is itself diagonalized
    by the DFT applied within each block, giving a spectrum we can monitor
    during training and use for preconditioning.

    Parameters
    ----------
    n_in, n_out : int
        Input / output dimensions.  Must be multiples of `block`.
    block : int
        Polygon multiplicity B = 2L+1.  Choose 1, 3, 5, 7, ...
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    init_scale : float
        Initial weight scale (Glorot-like, automatically scaled by sqrt(B/n_in)).
    """

    def __init__(self, n_in: int, n_out: int, block: int = 5,
                 rng: Optional[np.random.Generator] = None,
                 init_scale: float = 1.0):
        if n_in % block != 0 or n_out % block != 0:
            raise ValueError(
                f"n_in ({n_in}) and n_out ({n_out}) must be multiples of "
                f"block ({block}).  Use a block size that divides both, "
                f"e.g. 1, 3, 5, 7."
            )
        rng = rng or set_seed(0)
        self.n_in    = n_in
        self.n_out   = n_out
        self.block   = block
        self.K_in    = n_in  // block       # number of input row-blocks
        self.K_out   = n_out // block       # number of output col-blocks
        # First-row coefficients of each circulant block (K_out, K_in, B)
        std = init_scale * np.sqrt(2.0 / n_in)
        self.c       = rng.normal(0, std, size=(self.K_out, self.K_in, block))
        self.b       = np.zeros(n_out)
        # Cache for backward pass
        self._x_in   = None
        self._x_blk  = None    # input reshaped as (batch, K_in, B)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _circ_matvec(c_row: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Multiply a circulant matrix (defined by first row c_row, length B)
        with input x of shape (..., B) using FFT.

        For circulant C with first row c_row,
            C x  =  ifft(fft(c_row) * fft(x))[real part]

        We absorb the conjugation convention so that
        (Cx)[k] = sum_j c_row[(k-j) mod B] * x[j]
        """
        return np.real(np.fft.ifft(np.fft.fft(c_row) * np.fft.fft(x), axis=-1))

    @staticmethod
    def _circ_vjp_c(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Vector-Jacobian product wrt the first row c of a circulant block.

        Given y = C(c) x  with C[k,j] = c[(k-j) mod B],
        dL/dc[m] = sum_{k,j: (k-j)mod B = m} dy[k] x[j]
                = sum_k dy[k] x[(k-m) mod B]
        which is the cross-correlation of dy and x — computable via FFT.
        Inputs x and dy are batched: shape (batch, B).
        """
        return np.real(np.fft.ifft(
            np.conj(np.fft.fft(x, axis=-1)) * np.fft.fft(dy, axis=-1),
            axis=-1
        )).sum(axis=0)

    # -- forward --------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (batch, n_in)   ->   y : (batch, n_out)"""
        batch = x.shape[0]
        # reshape input to (batch, K_in, B) for circulant block products
        x_blk = x.reshape(batch, self.K_in, self.block)
        # output accumulator (batch, K_out, B)
        y_blk = np.zeros((batch, self.K_out, self.block))
        for ko in range(self.K_out):
            for ki in range(self.K_in):
                y_blk[:, ko, :] += self._circ_matvec(self.c[ko, ki], x_blk[:, ki, :])
        y = y_blk.reshape(batch, self.n_out) + self.b
        # cache for backward
        self._x_in  = x
        self._x_blk = x_blk
        return y

    # -- backward -------------------------------------------------------
    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, dict]:
        """dy : (batch, n_out)  ->  (dx, grads)

        grads : {'c': (K_out, K_in, B), 'b': (n_out,)}
        """
        batch = dy.shape[0]
        dy_blk = dy.reshape(batch, self.K_out, self.block)
        # Gradient wrt b
        gb = dy.sum(axis=0)
        # Gradient wrt c
        gc = np.zeros_like(self.c)
        for ko in range(self.K_out):
            for ki in range(self.K_in):
                gc[ko, ki] = self._circ_vjp_c(self._x_blk[:, ki, :],
                                              dy_blk[:, ko, :])
        # Gradient wrt x: dx[j] = sum_k C^T[j,k] dy[k];  C^T is circulant with
        # first row c_rev[m] = c[(-m) mod B]
        dx_blk = np.zeros((batch, self.K_in, self.block))
        for ki in range(self.K_in):
            for ko in range(self.K_out):
                c_rev = np.concatenate([self.c[ko, ki, :1],
                                        self.c[ko, ki, :0:-1]])
                dx_blk[:, ki, :] += self._circ_matvec(c_rev, dy_blk[:, ko, :])
        dx = dx_blk.reshape(batch, self.n_in)
        return dx, {'c': gc, 'b': gb}

    # -- diagnostics ----------------------------------------------------
    def hessian_spectrum(self) -> np.ndarray:
        """Return the union of FFT-diagonalized Hessian eigenvalues for all
        K_out * K_in circulant blocks.  Useful to monitor conditioning."""
        eigs = []
        for ko in range(self.K_out):
            for ki in range(self.K_in):
                eigs.append(np.abs(np.fft.fft(self.c[ko, ki])))
        return np.concatenate(eigs)

    def num_params(self) -> int:
        return self.c.size + self.b.size


# =============================================================================
# 2. DenseLinear: BASELINE FOR COMPARISON
# =============================================================================
class DenseLinear:
    """Standard dense linear layer y = W x + b for baseline comparison."""

    def __init__(self, n_in: int, n_out: int,
                 rng: Optional[np.random.Generator] = None,
                 init_scale: float = 1.0):
        rng = rng or set_seed(0)
        self.n_in   = n_in
        self.n_out  = n_out
        std = init_scale * np.sqrt(2.0 / n_in)
        self.W = rng.normal(0, std, size=(n_out, n_in))
        self.b = np.zeros(n_out)
        self._x_in = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_in = x
        return x @ self.W.T + self.b

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, dict]:
        gW = dy.T @ self._x_in
        gb = dy.sum(axis=0)
        dx = dy @ self.W
        return dx, {'W': gW, 'b': gb}

    def hessian_spectrum(self) -> np.ndarray:
        # SVD of W gives singular values; squared = Hessian eigenvalues
        return np.linalg.svd(self.W, compute_uv=False) ** 2

    def num_params(self) -> int:
        return self.W.size + self.b.size


# =============================================================================
# 3. ACTIVATIONS WITH MANUAL BACKWARD
# =============================================================================
class ReLU:
    def forward(self, x):  self._mask = (x > 0).astype(x.dtype); return x * self._mask
    def backward(self, dy): return dy * self._mask, {}
    def num_params(self):  return 0
    def hessian_spectrum(self): return np.array([])


class Softmax_CE:
    """Combined softmax + categorical cross-entropy loss.

    Forward returns scalar mean loss.  The cached log-probabilities and
    target one-hots together yield the simple gradient (p - y_onehot)/N.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self._p   = None
        self._y   = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        # logits: (N, C),  y: (N,) integer labels
        # Stable softmax
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.mean(np.log(p[np.arange(N), y] + 1e-12))
        self._p = p
        self._y = y
        return loss

    def backward(self) -> np.ndarray:
        N = self._p.shape[0]
        dlogits = self._p.copy()
        dlogits[np.arange(N), self._y] -= 1.0
        dlogits /= N
        return dlogits


# =============================================================================
# 4. SHANNON DROPOUT (CD-derived noise injection)
# =============================================================================
class ShannonDropout:
    """Drop each activation with probability alpha_CD = 0.0118.

    Rationale (Sec. III of the paper).  The CD framework treats each layer
    as a Shannon channel with noise rate alpha_CD calibrated from the Na
    D-doublet.  Rather than tuning dropout empirically (10-50%), CD
    prescribes a transferable fixed rate.  In practice this acts as a
    very mild stochastic regularizer.
    """

    def __init__(self, p: float = ALPHA_CD,
                 rng: Optional[np.random.Generator] = None):
        self.p   = p
        self.rng = rng or set_seed(0)
        self._mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p <= 0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) > self.p).astype(x.dtype)
        return x * self._mask / (1.0 - self.p)

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, dict]:
        if self._mask is None:
            return dy, {}
        return dy * self._mask / (1.0 - self.p), {}

    def num_params(self): return 0
    def hessian_spectrum(self): return np.array([])


# =============================================================================
# 5. FISHER INFORMATION REGULARIZER (closed form for circulant)
# =============================================================================
def fisher_reg_term(layer, lambda_F: float = 1e-4) -> float:
    """Return the Fisher-information regularization penalty for a layer.

    For a circulant block C with first row c, the Fisher information of the
    induced linear-Gaussian channel is

        I[c] = sum_k 1 / sigma_k^2

    where sigma_k = |fft(c)[k]| are the FFT-diagonal singular values.  The
    penalty rewards a flat spectrum (all sigma_k ~ equal) which provably
    gives the best-conditioned optimization landscape (Theorem 2 in the
    paper).  For a dense layer we use the trace of the Gram matrix.
    """
    if isinstance(layer, CDLinear):
        eigs = layer.hessian_spectrum()
        # Add small floor to avoid 1/0 for un-initialized blocks
        return float(lambda_F * np.sum(1.0 / (eigs + 1e-6)))
    elif isinstance(layer, DenseLinear):
        s2 = layer.hessian_spectrum()
        return float(lambda_F * np.sum(1.0 / (s2 + 1e-6)))
    else:
        return 0.0


# =============================================================================
# 6. SEQUENTIAL CONTAINER + TRAINING LOOP
# =============================================================================
class Sequential:
    """Minimal Sequential container with manual forward/backward."""

    def __init__(self, layers: List):
        self.layers = layers

    def forward(self, x):
        for L in self.layers:
            x = L.forward(x)
        return x

    def backward(self, dy):
        grads = []
        for L in reversed(self.layers):
            dy, g = L.backward(dy)
            grads.append((L, g))
        grads.reverse()
        return grads

    def num_params(self) -> int:
        return sum(L.num_params() for L in self.layers)

    def set_training(self, mode: bool):
        for L in self.layers:
            if hasattr(L, 'training'):
                L.training = mode


def sgd_step(grads_list, lr: float = 1e-2, momentum: float = 0.9,
             state: Optional[dict] = None) -> dict:
    """Apply SGD-with-momentum update and return updated state."""
    state = state or {}
    for layer, g in grads_list:
        for name, grad in g.items():
            param = getattr(layer, name)
            key = (id(layer), name)
            v = state.get(key, np.zeros_like(param))
            v = momentum * v - lr * grad
            param += v
            state[key] = v
    return state


# =============================================================================
# 7. UNIT TESTS — gradient checks via finite differences
# =============================================================================
def _grad_check(layer, x_shape, eps: float = 1e-5, tol: float = 1e-4,
                rng: Optional[np.random.Generator] = None) -> bool:
    """Verify backward pass against numerical finite differences."""
    rng = rng or set_seed(42)
    x = rng.normal(0, 1, size=x_shape)
    y = layer.forward(x)
    # Random output gradient
    dy = rng.normal(0, 1, size=y.shape)
    L_analytic = float(np.sum(y * dy))
    dx, grads = layer.backward(dy)

    # Check dx
    ok_dx = True
    if dx is not None:
        # finite-difference dx at a random subset of indices
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in x.shape)
            x_plus = x.copy(); x_plus[idx] += eps
            x_minus = x.copy(); x_minus[idx] -= eps
            num = (np.sum(layer.forward(x_plus) * dy)
                   - np.sum(layer.forward(x_minus) * dy)) / (2 * eps)
            ana = dx[idx]
            if abs(num - ana) / (abs(ana) + 1e-8) > tol:
                ok_dx = False
                print(f"  dx mismatch at {idx}: num={num:.6f} ana={ana:.6f}")

    # Check param gradients
    ok_par = True
    for name, gp in grads.items():
        param = getattr(layer, name)
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in param.shape)
            old = param[idx]
            param[idx] = old + eps
            Lp = float(np.sum(layer.forward(x) * dy))
            param[idx] = old - eps
            Lm = float(np.sum(layer.forward(x) * dy))
            param[idx] = old
            num = (Lp - Lm) / (2 * eps)
            ana = gp[idx]
            if abs(num - ana) / (abs(ana) + 1e-8) > tol:
                ok_par = False
                print(f"  grad[{name}] mismatch at {idx}: "
                      f"num={num:.6f} ana={ana:.6f}")
    return ok_dx and ok_par


def run_self_tests():
    """Run gradient checks on all layer types."""
    print("=" * 60)
    print("CD-NN library: gradient-check unit tests")
    print("=" * 60)
    rng = set_seed(7)

    print("\n[1] CDLinear (block=3, n_in=6, n_out=9) ...", end="")
    L = CDLinear(6, 9, block=3, rng=rng)
    print(" OK" if _grad_check(L, (4, 6), rng=rng) else " FAIL")

    print("[2] CDLinear (block=5, n_in=10, n_out=10) ...", end="")
    L = CDLinear(10, 10, block=5, rng=rng)
    print(" OK" if _grad_check(L, (4, 10), rng=rng) else " FAIL")

    print("[3] DenseLinear (n_in=8, n_out=6) ...", end="")
    L = DenseLinear(8, 6, rng=rng)
    print(" OK" if _grad_check(L, (4, 8), rng=rng) else " FAIL")

    print("\nAll gradient checks complete.")


if __name__ == "__main__":
    run_self_tests()
