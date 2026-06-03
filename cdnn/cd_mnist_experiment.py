#!/usr/bin/env python3
"""
=============================================================================
cd_mnist_experiment.py - Empirical test of CD-Net on MNIST
=============================================================================

This script compares three classifiers on a downsampled MNIST task
(8x8 images, 10 classes, 1797 samples - sklearn.datasets.load_digits)
under MATCHED PARAMETER BUDGETS:

    1. Dense MLP baseline   (~50k parameters)
    2. CD-Net (block=4)     (~13k parameters)   [4x compression]
    3. CD-Net (block=8)     (~6.5k parameters)  [8x compression]

For each model we train for the same number of epochs with the same
optimizer (SGD + momentum) and report:

    - Convergence curve (training loss vs epoch)
    - Final test accuracy
    - Hessian condition number (max/min eigenvalue of weight Gram)
    - Wall-clock time per epoch

The hypothesis from CD theory (Paper III, Sec. IV) is that CDLinear
layers, having FFT-diagonal Hessians by construction, converge faster
PER PARAMETER than dense layers and generalize at least as well at
matched parameter count.

Outputs:
    cd_mnist_results.json    - full numeric results
    cd_mnist_curves.png      - training-loss curves
    cd_mnist_spectrum.png    - Hessian spectrum at end of training

Usage:
    python3 cd_mnist_experiment.py [--epochs N] [--seed S]

Author: L. Pan (Ainnocence Inc.), M. Tanik (UAB)
=============================================================================
"""
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cd_nn import (CDLinear, DenseLinear, ReLU, Softmax_CE, ShannonDropout,
                   Sequential, sgd_step, set_seed, fisher_reg_term)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(seed: int = 0):
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0       # normalize to [0,1]
    y = digits.target.astype(np.int64)
    # 80/20 train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def build_dense_mlp(n_in: int, n_hid: int, n_out: int, rng):
    """Standard 3-layer MLP with ReLU activations."""
    return Sequential([
        DenseLinear(n_in, n_hid, rng=rng),
        ReLU(),
        ShannonDropout(p=0.0, rng=rng),     # disabled for fair comparison
        DenseLinear(n_hid, n_hid, rng=rng),
        ReLU(),
        ShannonDropout(p=0.0, rng=rng),
        DenseLinear(n_hid, n_out, rng=rng),
    ])


def build_cd_mlp(n_in: int, n_hid: int, n_out: int, block: int, rng):
    """CD-Net 3-layer MLP with circulant linear layers.

    n_in, n_hid, n_out must all be multiples of `block`.  Last layer maps
    n_hid -> n_classes_padded, where n_classes_padded is the smallest
    multiple of `block` >= n_out; we then slice off the extra logits.
    """
    n_in_pad  = ((n_in  + block - 1) // block) * block
    n_hid_pad = ((n_hid + block - 1) // block) * block
    n_out_pad = ((n_out + block - 1) // block) * block
    return Sequential([
        InputPad(n_in, n_in_pad),
        CDLinear(n_in_pad, n_hid_pad, block=block, rng=rng),
        ReLU(),
        ShannonDropout(p=0.0, rng=rng),
        CDLinear(n_hid_pad, n_hid_pad, block=block, rng=rng),
        ReLU(),
        ShannonDropout(p=0.0, rng=rng),
        CDLinear(n_hid_pad, n_out_pad, block=block, rng=rng),
        OutputSlice(n_out),
    ])


# ---------------------------------------------------------------------------
# Helpers: zero-pad inputs / slice outputs so dims are multiples of block
# ---------------------------------------------------------------------------
class InputPad:
    """Right-pad the input vector to a target dimension with zeros."""

    def __init__(self, n_in: int, n_pad: int):
        self.n_in = n_in
        self.n_pad = n_pad

    def forward(self, x):
        if self.n_pad == self.n_in:
            return x
        return np.concatenate([x, np.zeros((x.shape[0], self.n_pad - self.n_in))],
                              axis=1)

    def backward(self, dy):
        return dy[:, :self.n_in], {}

    def num_params(self): return 0
    def hessian_spectrum(self): return np.array([])


class OutputSlice:
    """Slice off the first n_out logits, discarding padding."""

    def __init__(self, n_out: int):
        self.n_out = n_out
        self._full_dim = None

    def forward(self, x):
        self._full_dim = x.shape[1]
        return x[:, :self.n_out]

    def backward(self, dy):
        out = np.zeros((dy.shape[0], self._full_dim))
        out[:, :self.n_out] = dy
        return out, {}

    def num_params(self): return 0
    def hessian_spectrum(self): return np.array([])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, loss_fn, X_tr, y_tr, X_te, y_te,
          epochs: int = 30, batch: int = 64, lr: float = 0.1,
          momentum: float = 0.9, fisher_lambda: float = 0.0,
          seed: int = 0, verbose: bool = True):
    rng = set_seed(seed)
    n = X_tr.shape[0]
    history = {"train_loss": [], "test_acc": [], "epoch_time": []}
    state = {}

    for epoch in range(epochs):
        t0 = time.time()
        # Shuffle
        idx = rng.permutation(n)
        X_sh, y_sh = X_tr[idx], y_tr[idx]
        epoch_losses = []

        for i in range(0, n, batch):
            xb = X_sh[i:i + batch]
            yb = y_sh[i:i + batch]
            model.set_training(True)
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            # Optional Fisher regularization
            if fisher_lambda > 0:
                for L in model.layers:
                    loss += fisher_reg_term(L, fisher_lambda)
            epoch_losses.append(loss)

            dlogits = loss_fn.backward()
            grads = model.backward(dlogits)
            state = sgd_step(grads, lr=lr, momentum=momentum, state=state)

        # Evaluate
        model.set_training(False)
        logits_te = model.forward(X_te)
        pred = np.argmax(logits_te, axis=1)
        acc = float(np.mean(pred == y_te))
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["test_acc"].append(acc)
        history["epoch_time"].append(time.time() - t0)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"loss = {history['train_loss'][-1]:.4f}  "
                  f"test_acc = {acc:.4f}  "
                  f"({history['epoch_time'][-1]:.2f} s)")

    return history


# ---------------------------------------------------------------------------
# Hessian condition-number diagnostic
# ---------------------------------------------------------------------------
def hessian_condition_number(model) -> float:
    """Report the condition number max(eig)/min(eig) over weight layers,
    averaged across all layers that expose a Hessian spectrum."""
    cn = []
    for L in model.layers:
        if hasattr(L, 'hessian_spectrum'):
            s = L.hessian_spectrum()
            if s.size > 0:
                s = s[s > 1e-12]
                if s.size > 0:
                    cn.append(s.max() / s.min())
    return float(np.mean(cn)) if cn else float("nan")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main(epochs: int = 30, seed: int = 0):
    print("=" * 70)
    print("CD-NN MNIST Experiment - Pan & Tanik (2026), Paper III")
    print("=" * 70)

    X_tr, X_te, y_tr, y_te = load_data(seed=seed)
    n_in = X_tr.shape[1]                # 64 (8x8 digits)
    n_out = 10
    n_hid = 64                          # hidden layer width
    print(f"\nDataset: sklearn.load_digits  ({X_tr.shape[0]} train + "
          f"{X_te.shape[0]} test, {n_in} features, {n_out} classes)")

    rng_dense = set_seed(seed)
    rng_cd4   = set_seed(seed)
    rng_cd8   = set_seed(seed)

    print(f"\nBuilding models (n_hid = {n_hid}):")
    dense = build_dense_mlp(n_in, n_hid, n_out, rng=rng_dense)
    cd4   = build_cd_mlp(n_in, n_hid, n_out, block=4, rng=rng_cd4)
    cd8   = build_cd_mlp(n_in, n_hid, n_out, block=8, rng=rng_cd8)

    print(f"  Dense MLP    : {dense.num_params():>6d} params")
    print(f"  CD-MLP B=4   : {cd4.num_params():>6d} params  "
          f"({dense.num_params()/cd4.num_params():.1f}x compression)")
    print(f"  CD-MLP B=8   : {cd8.num_params():>6d} params  "
          f"({dense.num_params()/cd8.num_params():.1f}x compression)")

    loss_dense = Softmax_CE(n_classes=n_out)
    loss_cd4   = Softmax_CE(n_classes=n_out)
    loss_cd8   = Softmax_CE(n_classes=n_out)

    print(f"\nTraining (epochs={epochs}, batch=64, lr=0.1, momentum=0.9):")

    print("\n--- Dense MLP ---")
    h_dense = train(dense, loss_dense, X_tr, y_tr, X_te, y_te,
                    epochs=epochs, lr=0.1, seed=seed)

    print("\n--- CD-MLP (block=4) ---")
    h_cd4 = train(cd4, loss_cd4, X_tr, y_tr, X_te, y_te,
                  epochs=epochs, lr=0.1, seed=seed)

    print("\n--- CD-MLP (block=8) ---")
    h_cd8 = train(cd8, loss_cd8, X_tr, y_tr, X_te, y_te,
                  epochs=epochs, lr=0.1, seed=seed)

    # Compute condition numbers post-training
    cn_dense = hessian_condition_number(dense)
    cn_cd4   = hessian_condition_number(cd4)
    cn_cd8   = hessian_condition_number(cd8)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18}{'#params':>10}{'final loss':>14}"
          f"{'final acc':>12}{'cond #':>14}{'avg sec/ep':>14}")
    print("-" * 82)
    for name, h, cn, m in [
        ("Dense MLP",      h_dense, cn_dense, dense),
        ("CD-MLP (B=4)",   h_cd4,   cn_cd4,   cd4),
        ("CD-MLP (B=8)",   h_cd8,   cn_cd8,   cd8),
    ]:
        print(f"{name:<18}{m.num_params():>10d}"
              f"{h['train_loss'][-1]:>14.4f}"
              f"{h['test_acc'][-1]:>12.4f}"
              f"{cn:>14.2f}"
              f"{np.mean(h['epoch_time']):>14.3f}")

    # ------- Plots ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].semilogy(h_dense['train_loss'], label='Dense MLP', linewidth=2)
    ax[0].semilogy(h_cd4['train_loss'],   label='CD-MLP (B=4)', linewidth=2)
    ax[0].semilogy(h_cd8['train_loss'],   label='CD-MLP (B=8)', linewidth=2)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Training loss (log scale)')
    ax[0].set_title('Convergence: matched LR and seed')
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(h_dense['test_acc'], label='Dense MLP', linewidth=2)
    ax[1].plot(h_cd4['test_acc'],   label='CD-MLP (B=4)', linewidth=2)
    ax[1].plot(h_cd8['test_acc'],   label='CD-MLP (B=8)', linewidth=2)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Test accuracy')
    ax[1].set_title('Test accuracy vs epoch')
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('cd_mnist_curves.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("\nSaved: cd_mnist_curves.png")

    # Hessian spectrum plot (last weight layer of each model)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    # Dense: last DenseLinear layer
    sp_d = [L.hessian_spectrum() for L in dense.layers if isinstance(L, DenseLinear)][-1]
    sp_4 = [L.hessian_spectrum() for L in cd4.layers   if isinstance(L, CDLinear)][-1]
    sp_8 = [L.hessian_spectrum() for L in cd8.layers   if isinstance(L, CDLinear)][-1]
    ax.semilogy(np.sort(sp_d)[::-1], label=f'Dense (cond {cn_dense:.1f})',
                linewidth=2)
    ax.semilogy(np.sort(sp_4)[::-1], label=f'CD B=4 (cond {cn_cd4:.1f})',
                linewidth=2)
    ax.semilogy(np.sort(sp_8)[::-1], label=f'CD B=8 (cond {cn_cd8:.1f})',
                linewidth=2)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('|eigenvalue| (log scale)')
    ax.set_title('Hessian spectrum at end of training')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('cd_mnist_spectrum.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: cd_mnist_spectrum.png")

    # JSON summary
    out = {
        "epochs": epochs, "seed": seed,
        "dataset": "sklearn.load_digits (8x8 MNIST)",
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
        "models": {
            "Dense":     {"params": dense.num_params(),
                          "history": h_dense, "condition_number": cn_dense},
            "CD_block4": {"params": cd4.num_params(),
                          "history": h_cd4,   "condition_number": cn_cd4},
            "CD_block8": {"params": cd8.num_params(),
                          "history": h_cd8,   "condition_number": cn_cd8},
        },
    }
    with open('cd_mnist_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved: cd_mnist_results.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()
    main(epochs=args.epochs, seed=args.seed)
