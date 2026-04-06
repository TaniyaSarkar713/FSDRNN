#!/usr/bin/env python3
"""
Teacher-Student Simulation for FSDRNN Evaluation (SPD-Only)

This script implements a teacher-student simulation that mirrors the three stages of FSDRNN:
1. Predictor reduction to Z = f0(X)
2. Response-specific weight generation H(X)
3. Low-rank refinement W(X) = H(X) M0
4. Response generation from weighted Fréchet means

The teacher (FSDRNN oracle) generates SPD matrix-valued data, and students (various methods) 
try to recover the latent structure and predict responses.

SPD-only implementation: Uses Fréchet geometry on the manifold of symmetric positive definite matrices.
"""

import sys
import os
import argparse
import time
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.spd_frechet_adaptive import (
    FrechetDRNN,
    train_frechet_model,
    evaluate_frechet_model,
    differentiable_frechet_mean,
    get_distance_fn,
    GlobalFrechetRegression,
    DeepFrechetRegression,
)
import scipy.optimize


def matrix_log(M):
    """Matrix logarithm for SPD matrices with numerical stability."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals = eigvals.clamp(min=1e-8)  # Avoid log of tiny numbers
    return eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T


def matrix_exp(M):
    """Matrix exponential for SPD matrices with numerical stability and symmetrization."""
    eigvals, eigvecs = torch.linalg.eigh((M + M.T) / 2)  # Symmetrize input
    out = eigvecs @ torch.diag(torch.exp(eigvals)) @ eigvecs.T
    return 0.5 * (out + out.T)  # Symmetrize result


def generate_teacher_params_spd(p, m, d0, V, r0, n_ref, seed=None):
    """Generate teacher parameters for SPD case with shared Y_ref."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    B_star = torch.randn(p, d0)
    B_star, _ = torch.linalg.qr(B_star)

    # Generate more spread out anchors
    centers = []
    for v in range(V):
        Q = torch.randn(m, m)
        C = (Q + Q.T) / 2
        C = 1.5 * C / torch.norm(C, 'fro')
        centers.append(C)

    Y_ref = torch.zeros(n_ref, V, m, m)
    for i in range(n_ref):
        for v in range(V):
            E = torch.randn(m, m)
            E = (E + E.T) / 2
            E = 0.25 * E / torch.norm(E, 'fro')
            Y_ref[i, v] = matrix_exp(centers[v] + E)

    hidden_dim = 64
    head_params = []
    for v in range(V):
        W1 = torch.randn(d0, hidden_dim) * 0.4
        b1 = torch.zeros(hidden_dim)
        W2 = torch.randn(hidden_dim, n_ref) * 0.4
        b2 = torch.zeros(n_ref)
        head_params.append({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

    lambda_coupling = 1.5  # Strengthened cross-response coupling
    A0 = torch.randn(V, r0) * 0.1
    B0 = torch.randn(V, r0) * 0.1
    M0 = torch.eye(V) + lambda_coupling * (A0 @ B0.T)

    return {
        'B_star': B_star,
        'Y_ref': Y_ref,
        'head_params': head_params,
        'M0': M0,
        'd0': d0,
        'r0': r0,
    }


def local_frechet_predict_1d(t_train, Y_train, t_query, dist_name, frechet_mean_fn, h):
    """Local Fréchet prediction in 1D index space."""
    # Convert inputs to tensors
    t_train = torch.tensor(t_train, dtype=torch.float32) if isinstance(t_train, np.ndarray) else t_train
    t_query = torch.tensor(t_query, dtype=torch.float32) if isinstance(t_query, np.ndarray) else t_query
    h = torch.tensor(h, dtype=torch.float32) if isinstance(h, (int, float, np.ndarray)) else h

    # Gaussian kernel weights
    w = torch.exp(-0.5 * ((t_train - t_query) / h) ** 2)
    w = w / (torch.sum(w) + 1e-12)

    # Check if scalar case (first element is a number)
    is_scalar = isinstance(Y_train[0], (int, float, np.number))
    
    if is_scalar:
        # Scalar case: Y_train is list of scalars
        Y_train_torch = torch.tensor(Y_train, dtype=torch.float32)  # [n]
        # For scalars, weighted mean is just dot product
        pred = torch.sum(w * Y_train_torch)  # scalar
        return pred.item()
    else:
        # Matrix case: Y_train is list of tensors/matrices
        w_torch = w.unsqueeze(0)  # [1, n]
        Y_train_torch = torch.stack([y.detach().clone() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32) for y in Y_train])  # [n, p, p]

        # Compute weighted Fréchet mean
        pred = frechet_mean_fn(w_torch, Y_train_torch, dist_name)
        pred = pred.squeeze(0)  # [1, p, p] -> [p, p]
        return pred.detach().numpy()


def predict_ifr(X_train, Y_train, theta, X_new, dist_name, frechet_mean_fn, h):
    """Predict using fitted IFR model."""
    t_train = X_train @ theta
    preds = []
    
    for x in X_new:
        t = x @ theta
        yhat = local_frechet_predict_1d(t_train, Y_train, t, dist_name, frechet_mean_fn, h)
        preds.append(yhat)
    
    return preds


def ifr_loss(beta, X, Y, dist_name, frechet_mean_fn, h):
    """Loss function for IFR optimization."""
    from src.spd_frechet_adaptive import get_distance_fn
    dist_fn = get_distance_fn(dist_name)

    theta = beta / (np.linalg.norm(beta) + 1e-12)
    t = X @ theta
    loss = 0.0

    for i in range(len(Y)):
        yhat_i = local_frechet_predict_1d(t, Y, t[i], dist_name, frechet_mean_fn, h)
        
        # Handle scalar vs matrix distance
        is_scalar_i = isinstance(Y[i], (int, float, np.number))
        if is_scalar_i:
            # Scalar case
            loss += (yhat_i - Y[i]) ** 2
        else:
            # Matrix case - ensure both are tensors
            if isinstance(yhat_i, np.ndarray):
                yhat_i = torch.tensor(yhat_i, dtype=torch.float32)
            if isinstance(Y[i], np.ndarray):
                Y_i = torch.tensor(Y[i], dtype=torch.float32)
            else:
                Y_i = Y[i]
            loss += (dist_fn(yhat_i, Y_i) ** 2).item()

    return loss / len(Y)


def fit_ifr(X, Y, dist_name, frechet_mean_fn, h, beta0=None, max_iter=50):
    """Fit single-index Fréchet regression."""
    p = X.shape[1]
    if beta0 is None:
        # Initialize with first principal component
        X_centered = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        beta0 = Vt[0]  # First PC
    
    # Optimize beta
    res = scipy.optimize.minimize(
        ifr_loss,
        beta0,
        args=(X, Y, dist_name, frechet_mean_fn, h),
        method="L-BFGS-B",
        options={'maxiter': max_iter}
    )
    
    beta = res.x
    theta = beta / (np.linalg.norm(beta) + 1e-12)
    return theta


def sir_candidate_matrix(X, u, H=10):
    """Compute SIR candidate matrix for scalar response u."""
    # Standardize X
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # Slice u into H bins
    u_sorted_idx = np.argsort(u)
    n_per_slice = len(u) // H
    remainder = len(u) % H

    M = np.zeros((X.shape[1], X.shape[1]))

    start_idx = 0
    for h in range(H):
        # Determine slice size
        slice_size = n_per_slice + (1 if h < remainder else 0)
        end_idx = start_idx + slice_size

        # Get indices for this slice
        slice_idx = u_sorted_idx[start_idx:end_idx]
        start_idx = end_idx

        if len(slice_idx) == 0:
            continue

        # Slice mean
        X_slice = X_std[slice_idx]
        x_bar_h = X_slice.mean(axis=0)

        # Weight by slice proportion
        p_h = len(slice_idx) / len(u)
        M += p_h * np.outer(x_bar_h, x_bar_h)

    return M


def frechet_sdr(X, Y, dist_fn, d, n_anchors=30, method="sir", gamma=None):
    """Fréchet SDR using kernel ensemble."""
    n = len(Y)

    # 1. Pairwise distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dij = dist_fn(Y[i], Y[j])
            D[i, j] = D[j, i] = dij

    # 2. Kernel bandwidth
    if gamma is None:
        sigma2 = np.mean(D[np.triu_indices(n, 1)] ** 2)
        gamma = 1.0 / (2.0 * sigma2 + 1e-8)

    # 3. Choose anchors
    anchor_idx = np.random.choice(n, size=min(n_anchors, n), replace=False)

    # 4. Ensemble candidate matrix
    M_sum = np.zeros((X.shape[1], X.shape[1]))

    for a in anchor_idx:
        # Kernel pseudo-response
        u = np.exp(-gamma * D[:, a] ** 2)

        if method == "sir":
            M_a = sir_candidate_matrix(X, u)
        else:
            raise ValueError(f"SDR method {method} not implemented")

        M_sum += M_a

    M_hat = M_sum / len(anchor_idx)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M_hat)
    order = np.argsort(eigvals)[::-1]
    B_hat = eigvecs[:, order[:d]]

    return B_hat, M_hat


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_weight_recovery(W_pred, W_true):
    """Evaluate weight recovery quality."""
    # W_pred, W_true: [n_test, n_ref, V]
    # Compute L2 distance
    l2_dist = np.linalg.norm(W_pred - W_true, axis=(1, 2)).mean()
    
    # Compute KL divergence (assuming normalized weights)
    # Clip for numerical stability
    W_pred_clipped = np.clip(W_pred, 1e-12, 1.0)
    W_true_clipped = np.clip(W_true, 1e-12, 1.0)
    kl_div = (W_true_clipped * np.log(W_true_clipped / W_pred_clipped)).sum(axis=(1, 2)).mean()
    
    return l2_dist, kl_div


class NeuralTeacherSPDDataset(torch.utils.data.Dataset):
    """
    Teacher-Student Dataset for FSDRNN evaluation (SPD case only).

    Mirrors the three stages of FSDRNN:
    1. Predictor reduction to Z = f0(X)
    2. Response-specific weight generation H(X)
    3. Low-rank refinement W(X) = H(X) M0
    4. Response generation from weighted Fréchet means
    """

    def __init__(self, n: int, p: int = 20, m: int = 4, V: int = 10,
                 d0: int = 2, r0: int = 2, n_ref: int = 100,
                 teacher_params: Optional[Dict] = None, seed: int = 42,
                 noise_std: float = 0.02):
        super().__init__()
        self.n = n
        self.p = p
        self.m = m
        self.V = V
        self.d0 = d0
        self.r0 = r0
        self.n_ref = n_ref
        self.noise_std = noise_std

        # Set seed for X and noise generation
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Use provided teacher params or generate new ones
        if teacher_params is None:
            teacher_params = generate_teacher_params_spd(p, m, d0, V, r0, n_ref, seed)
        self.teacher_params = teacher_params

        # Extract teacher parameters (Y_ref is shared between train/test)
        self.B_star = teacher_params['B_star']
        self.Y_ref = teacher_params['Y_ref']
        self.head_params = teacher_params['head_params']
        self.M0 = teacher_params['M0']

        # 1. Generate predictors X ~ N(0, Sigma_X)
        rho_X = 0.3
        Sigma_X = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma_X[i,j] = rho_X ** abs(i - j)

        self.X = torch.from_numpy(
            np.random.multivariate_normal(np.zeros(p), Sigma_X, n)
        ).float()

        # 2. True sufficient reduction
        # U = X @ B_star [n, d0]
        U = self.X @ self.B_star

        # Nonlinear map Z = f0(U)
        if d0 >= 1:
            Z1 = U[:, 0]  # linear
        if d0 >= 2:
            Z2 = U[:, 1]**2 - 1  # quadratic
        if d0 >= 3:
            Z3 = torch.sin(np.pi * U[:, 0] * U[:, 1])  # interaction

        Z_components = []
        if d0 >= 1: Z_components.append(Z1.unsqueeze(1))
        if d0 >= 2: Z_components.append(Z2.unsqueeze(1))
        if d0 >= 3: Z_components.append(Z3.unsqueeze(1))
        self.Z = torch.cat(Z_components, dim=1)  # [n, d0]

        # 3. Teacher head networks H(X) [n, n_ref, V]
        self.H = self._generate_head_logits()

        # 4. Low-rank response coupling
        # W_true [n, n_ref, V]
        self.W_true = self._compute_true_weights()

        # 5. True conditional means m_true [n, V, ...]
        self.m_true = self._compute_true_means()

        # 6. Observed responses Y [n, V, ...]
        self.Y = self._generate_observed_responses()

    def _generate_head_logits(self):
        """Generate teacher head logits H [n, n_ref, V] using stored parameters"""
        H = torch.zeros(self.n, self.n_ref, self.V)
        for v in range(self.V):
            params = self.head_params[v]
            W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

            # Forward pass
            hidden = torch.tanh(self.Z @ W1 + b1)  # [n, hidden_dim]
            logits = hidden @ W2 + b2  # [n, n_ref]
            H[:, :, v] = logits

        return H

    def _compute_true_weights(self):
        """Compute true weights W_true [n, n_ref, V] with temperature sharpening"""
        # H_tilde = H @ M0  (einsum)
        H_tilde = torch.einsum('bnv,vw->bnw', self.H, self.M0)  # [n, n_ref, V]

        # Softmax with temperature = 0.2 for much sharper weights (strengthened structure)
        temperature = 0.2
        W_true = torch.softmax(H_tilde / temperature, dim=1)
        return W_true

    def _compute_true_means(self):
        """Compute true conditional means m_true [n, V, m, m]"""
        # Weighted Fréchet mean
        m_true = torch.zeros(self.n, self.V, self.m, self.m)
        for i in range(self.n):
            for v in range(self.V):
                weights = self.W_true[i, :, v]  # [n_ref]
                Y_ref_v = self.Y_ref[:, v]  # [n_ref, m, m]
                # Use differentiable Frechet mean
                mean_v = differentiable_frechet_mean(
                    weights.unsqueeze(0),   # [1, n_ref]
                    Y_ref_v,                # [n_ref, m, m]
                    'frobenius'
                )
                m_true[i, v] = mean_v.squeeze(0)
        return m_true

    def _generate_observed_responses(self):
        """Generate observed SPD responses with tangent-space noise (noise_std=0.03)"""
        Y = torch.zeros(self.n, self.V, self.m, self.m)
        for i in range(self.n):
            for v in range(self.V):
                # Log-space noise using matrix logarithm
                log_m = matrix_log(self.m_true[i, v])
                noise_matrix = torch.randn(self.m, self.m) * self.noise_std
                noise_matrix = (noise_matrix + noise_matrix.T) / 2  # symmetric
                log_Y = log_m + noise_matrix
                Y[i, v] = matrix_exp(log_Y)
        return Y

    def get_true_mean(self, idx):
        """Get the true conditional mean for sample idx."""
        return self.m_true[idx]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def evaluate_predictions(preds, Y_true, m_true, dist_fn):
    """
    SPD case only:
    preds:  [n, V, m, m]
    Y_true: same shape as preds
    m_true: same shape as preds (true conditional means)
    """
    d_list = []
    d2_list = []
    mspe_list = []

    n = len(Y_true)
    V = Y_true.shape[1]

    for i in range(n):
        d_per_v = []
        d2_per_v = []
        mspe_per_v = []
        for v in range(V):
            pred_v = preds[i, v]
            if not isinstance(pred_v, torch.Tensor):
                pred_v = torch.tensor(pred_v, dtype=torch.float32)

            y_v = Y_true[i, v]
            m_v = m_true[i, v]

            dv = dist_fn(pred_v, y_v)
            d_per_v.append(dv.item() if isinstance(dv, torch.Tensor) else float(dv))
            d2_per_v.append((dv ** 2).item() if isinstance(dv, torch.Tensor) else float(dv ** 2))

            dm = dist_fn(pred_v, m_v)
            mspe_per_v.append((dm ** 2).item() if isinstance(dm, torch.Tensor) else float(dm ** 2))

        d = np.mean(d_per_v)
        d2 = np.mean(d2_per_v)
        mspe = np.mean(mspe_per_v)

        d_list.append(d)
        d2_list.append(d2)
        mspe_list.append(mspe)

    return {
        "avg_dist": float(np.mean(d_list)),
        "avg_dist_sq": float(np.mean(d2_list)),
        "amspe": float(np.mean(mspe_list)),
    }


def _evaluate_amspe_on_loader(model, loader, dataset, Y_ref, dist_name, device):
    """Evaluate AMSPE (Average Mean Squared Prediction Error) against true conditional means."""
    from src.spd_frechet_adaptive import get_distance_fn
    dist_fn = get_distance_fn(dist_name)
    model.eval()
    d2_vals = []
    start_idx = 0

    with torch.no_grad():
        for X_b, Y_b in loader:
            B = X_b.size(0)
            X_b = X_b.to(device)

            if hasattr(model, 'get_weights'):
                W = model.get_weights(X_b)  # [B, n_ref, V] - use get_weights for proper normalization
            else:
                logits = model(X_b)  # [B, n_ref] or [B, n_ref, V]
                if hasattr(model, 'n_responses') and model.n_responses == 1:
                    W = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, n_ref, 1]
                else:
                    W = torch.softmax(logits, dim=1)  # [B, n_ref, V]

            # collect true means for this batch from dataset
            true_means = []
            for i in range(start_idx, start_idx + B):
                true_means.append(dataset.get_true_mean(i))
            true_means = torch.stack(true_means).to(device)   # [B, V, m, m] or [B, V]
            start_idx += B

            # Determine number of responses
            n_responses = getattr(model, 'n_responses', W.shape[-1] if W.dim() == 3 else 1)

            if n_responses == 1:
                Y_ref_use = Y_ref[:, 0].to(device) if Y_ref.dim() == 4 else Y_ref.to(device)
                Y_hat = differentiable_frechet_mean(W.squeeze(-1), Y_ref_use, dist_name)
                d2 = dist_fn(Y_hat, true_means) ** 2
                d2_vals.append(d2.cpu())
            else:
                V = W.shape[2]
                d2_batch = []
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v].to(device)
                    Y_hat_v = differentiable_frechet_mean(w_v, Y_ref_v, dist_name)
                    d2_v = dist_fn(Y_hat_v, true_means[:, v]) ** 2
                    d2_batch.append(d2_v)
                d2_avg = torch.stack(d2_batch).mean(dim=0)   # [B]
                d2_vals.append(d2_avg.cpu())

    cat = torch.cat(d2_vals)
    return cat.mean().item()


def run_teacher_student_simulation(
    n_train: int = 250,
    n_test: int = 250,
    p: int = 20,
    m: int = 4,
    V: int = 8,
    d0: int = 3,
    r0: int = 2,
    n_ref: int = 80,
    epochs: int = 100,
    lr: float = 5e-4,
    entropy_reg: float = 0.0,
    dropout: float = 0.0,
    batch_size: int = 32,
    device: str = "cpu",
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run teacher-student simulation for FSDRNN evaluation (SPD case only).
    """
    set_seed(seed)

    if verbose:
        print(f"\n{'='*80}")
        print("  Teacher-Student FSDRNN Simulation")
        print(f"  case=spd, n_train={n_train}, n_test={n_test}")
        print(f"  p={p}, V={V}, d0={d0}, r0={r0}, n_ref={n_ref}")
        print(f"  epochs={epochs}, device={device}, seed={seed}")
        print(f"{'='*80}")

    # Generate shared teacher parameters with improved design
    teacher_params = generate_teacher_params_spd(p, m, d0, V, r0, n_ref, seed)

    # Generate datasets with shared teacher (Y_ref is now reliably shared)
    ds_train = NeuralTeacherSPDDataset(n_train, p, m, V, d0, r0, n_ref, teacher_params, seed=seed)
    ds_test = NeuralTeacherSPDDataset(n_test, p, m, V, d0, r0, n_ref, teacher_params, seed=seed+1000)

    # Convert to numpy for some methods
    X_train_np = ds_train.X.numpy()
    X_test_np = ds_test.X.numpy()
    Y_train = ds_train.Y
    Y_test = ds_test.Y

    # True values for evaluation
    W_true_train = ds_train.W_true.numpy()
    W_true_test = ds_test.W_true.numpy()
    m_true_train = ds_train.m_true
    m_true_test = ds_test.m_true
    B_star = teacher_params['B_star'].numpy()

    results = {
        'true_d0': d0,
        'true_r0': r0,
        'true_B_star': B_star,
        'n_train': n_train,
        'n_test': n_test,
        'V': V,
        'm': m,
        'n_ref': n_ref,
    }

    # Distance function (SPD case only)
    dist_name = 'frobenius'
    dist_fn = get_distance_fn(dist_name)

    # ==============================================================
    # Method Evaluations
    # ==============================================================

    # 1. Global Mean (baseline)
    if verbose:
        print(f"\n  --- Global Mean (Baseline) ---")
    t_baseline = time.time()

    # For SPD, compute one Frechet mean per response
    global_mean = torch.zeros(V, m, m)
    for v in range(V):
        Y_train_v = Y_train[:, v]  # [n, m, m]
        weights = torch.ones(len(Y_train_v)) / len(Y_train_v)
        mean_v = differentiable_frechet_mean(
            weights.unsqueeze(0), Y_train_v, 'frobenius'
        ).squeeze(0)
        global_mean[v] = mean_v
    baseline_preds = global_mean.numpy().reshape(1, V, m, m).repeat(len(ds_test), axis=0)

    baseline_preds_stacked = np.array(baseline_preds)

    # Evaluate baseline
    baseline_metrics = evaluate_predictions(
        baseline_preds_stacked, Y_test, m_true_test, dist_fn
    )

    baseline_time = time.time() - t_baseline

    results.update({
        'baseline_avg_dist': baseline_metrics["avg_dist"],
        'baseline_avg_dist_sq': baseline_metrics["avg_dist_sq"],
        'baseline_amspe': baseline_metrics["amspe"],
        'baseline_time_sec': baseline_time,
    })

    # 2. IFR (Single-index Fréchet Regression)
    if verbose:
        print(f"\n  --- IFR (Single-index Fréchet Regression) ---")
    t_ifr = time.time()

    ifr_preds = []
    ifr_weight_errors = []

    for v in range(V):
        if verbose and v == 0:
            print(f"    Fitting IFR for response {v+1}/{V}...")

        # Extract response v (SPD case)
        Y_train_v = [Y_train[i, v] for i in range(len(Y_train))]
        Y_test_v = [Y_test[i, v] for i in range(len(Y_test))]

        # Fit IFR
        theta_v = fit_ifr(X_train_np, Y_train_v, dist_name, differentiable_frechet_mean, h=1.0)

        # Predict
        preds_v = predict_ifr(X_train_np, Y_train_v, theta_v, X_test_np, dist_name, differentiable_frechet_mean, h=1.0)
        ifr_preds.append(preds_v)

        # Weight recovery (dummy - IFR doesn't estimate weights)
        ifr_weight_errors.append(np.nan)

    # Stack predictions (SPD case)
    ifr_preds_stacked = np.array([[ifr_preds[v][i] for v in range(V)] for i in range(len(ds_test))])

    # Evaluate IFR
    ifr_metrics = evaluate_predictions(
        ifr_preds_stacked, Y_test, m_true_test, dist_fn
    )

    ifr_time = time.time() - t_ifr

    results.update({
        'ifr_avg_dist': ifr_metrics["avg_dist"],
        'ifr_avg_dist_sq': ifr_metrics["avg_dist_sq"],
        'ifr_amspe': ifr_metrics["amspe"],
        'ifr_time_sec': ifr_time,
    })

    # 3. Fréchet SDR
    if verbose:
        print(f"\n  --- Fréchet SDR (SIR ensemble) ---")
    t_fsdr = time.time()

    fsdr_preds = []
    fsdr_weight_errors = []

    for v in range(V):
        if verbose and v == 0:
            print(f"    Fitting Fréchet SDR for response {v+1}/{V}...")

        # Extract response v (SPD case)
        Y_train_v = [Y_train[i, v] for i in range(len(Y_train))]
        Y_test_v = [Y_test[i, v] for i in range(len(Y_test))]

        # Fit Fréchet SDR (SPD case)
        d_sdr = min(d0, X_train_np.shape[1])
        B_hat, _ = frechet_sdr(X_train_np, Y_train_v, dist_fn, d=d_sdr)

        theta_v = B_hat[:, 0]

        # Predict using local Fréchet regression
        preds_v = predict_ifr(X_train_np, Y_train_v, theta_v, X_test_np, dist_name, differentiable_frechet_mean, h=1.0)
        fsdr_preds.append(preds_v)

        # Weight recovery (dummy)
        fsdr_weight_errors.append(np.nan)

    # Stack predictions (SPD case)
    fsdr_preds_stacked = np.array([[fsdr_preds[v][i] for v in range(V)] for i in range(len(ds_test))])

    # Evaluate FSDR
    fsdr_metrics = evaluate_predictions(
        fsdr_preds_stacked, Y_test, m_true_test, dist_fn
    )

    fsdr_time = time.time() - t_fsdr

    results.update({
        'fsdr_avg_dist': fsdr_metrics["avg_dist"],
        'fsdr_avg_dist_sq': fsdr_metrics["avg_dist_sq"],
        'fsdr_amspe': fsdr_metrics["amspe"],
        'fsdr_time_sec': fsdr_time,
    })

    # 4. GFR (Global Fréchet Regression) - SPD case
    if verbose:
        print(f"\n  --- GFR (Global Fréchet Regression) ---")
    t_gfr = time.time()

    gfr_model = GlobalFrechetRegression(
        dist_name=dist_name
    )

    # Fit GFR
    gfr_model.fit(torch.tensor(X_train_np), Y_train)

    # Predict
    gfr_preds = gfr_model.predict(torch.tensor(X_test_np))

    # Evaluate GFR
    gfr_metrics = evaluate_predictions(
        gfr_preds, Y_test, m_true_test, dist_fn
    )

    gfr_time = time.time() - t_gfr

    results.update({
        'gfr_avg_dist': gfr_metrics["avg_dist"],
        'gfr_avg_dist_sq': gfr_metrics["avg_dist_sq"],
        'gfr_amspe': gfr_metrics["amspe"],
        'gfr_time_sec': gfr_time,
    })

    # 5. DFR (Deep Fréchet Regression) - SPD case
    if verbose:
        print(f"\n  --- DFR (Deep Fréchet Regression) ---")
    t_dfr = time.time()

    dfr_model = DeepFrechetRegression(
        dist_name=dist_name,
        hidden=50,
        layer=2,
        num_epochs=epochs,
        lr=lr,
        device=device
    )

    # Fit DFR
    dfr_model.fit(torch.tensor(X_train_np), Y_train)

    # Predict
    dfr_preds = dfr_model.predict(torch.tensor(X_test_np))

    # Evaluate DFR
    dfr_metrics = evaluate_predictions(
        dfr_preds, Y_test, m_true_test, dist_fn
    )

    dfr_time = time.time() - t_dfr

    results.update({
        'dfr_avg_dist': dfr_metrics["avg_dist"],
        'dfr_avg_dist_sq': dfr_metrics["avg_dist_sq"],
        'dfr_amspe': dfr_metrics["amspe"],
        'dfr_time_sec': dfr_time,
    })

    # 6. E2M (Embedding to Manifold) - FrechetDRNN without reduction or LoRA
    if verbose:
        print(f"\n  --- E2M (Embedding to Manifold) ---")
    t_e2m = time.time()

    # E2M: FrechetDRNN with no reduction (reduction_dim=p) and no response LoRA (response_rank=None)
    e2m_model = FrechetDRNN(
        input_dim=p,
        n_ref=n_ref,
        reduction_dim=p,  # no reduction
        n_responses=V,
        reduction_type='nonlinear',
        response_rank=None,  # no LoRA
        dropout=dropout
    ).to(device)

    # Create train loader for E2M
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Train E2M
    train_frechet_model(
        model=e2m_model,
        Y_ref=ds_train.Y_ref,
        train_loader=train_loader,
        dist_name=dist_name,
        epochs=epochs,
        lr=lr,
        entropy_reg=entropy_reg,
        device=device,
        verbose=verbose,
    )

    # Evaluate E2M using evaluate_frechet_model
    e2m_results = evaluate_frechet_model(
        model=e2m_model,
        Y_ref=ds_train.Y_ref,
        test_loader=test_loader,
        dist_name=dist_name,
        device=device,
        true_means=m_true_test
    )

    e2m_avg_dist = e2m_results['avg_dist']
    e2m_avg_dist_sq = e2m_results['avg_dist_sq']
    e2m_amspe = e2m_results['avg_dist_sq_to_mean']

    e2m_time = time.time() - t_e2m

    # Weight recovery for E2M
    with torch.no_grad():
        W_pred_e2m = e2m_model.get_weights(torch.tensor(X_test_np, dtype=torch.float32).to(device))
        W_pred_e2m = W_pred_e2m.cpu().numpy()
    e2m_weight_l2, e2m_weight_kl = evaluate_weight_recovery(W_pred_e2m, W_true_test)

    e2m_time = time.time() - t_e2m

    results.update({
        'e2m_avg_dist': e2m_avg_dist,
        'e2m_avg_dist_sq': e2m_avg_dist_sq,
        'e2m_amspe': e2m_amspe,
        'e2m_time_sec': e2m_time,
        'e2m_weight_l2': e2m_weight_l2,
        'e2m_weight_kl': e2m_weight_kl,
    })

    # 7. FSDRNN - with extended grid search over d, lr, entropy_reg, dropout
    if verbose:
        print(f"\n  --- FSDRNN (grid search: d∈[2,3,4], lr∈[3e-4,5e-4,1e-3], entropy_reg∈[0,1e-4,1e-3], dropout∈[0.0,0.1]) ---")
    t_fsdrnn = time.time()

    # Grid search parameters
    d_candidates = [2, 3, 4]
    lr_candidates = [3e-4, 5e-4, 1e-3]
    entropy_reg_candidates = [0.0, 1e-4, 1e-3]
    dropout_candidates = [0.0, 0.1]
    r_max = min(V, 6)  # adaptive LoRA max rank
    
    best_d = None
    best_lr = None
    best_entropy_reg = None
    best_dropout = None
    best_val_loss = float('inf')
    best_fsdrnn_model = None
    
    # Create validation set (20% of training data)
    val_size = int(0.2 * len(ds_train))
    train_size = len(ds_train) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        ds_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    
    # Convert to tensors for validation
    X_val = torch.stack([ds_train.X[i] for i in val_subset.indices])
    Y_val = torch.stack([ds_train.Y[i] for i in val_subset.indices])
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Grid search
    grid_size = len(d_candidates) * len(lr_candidates) * len(entropy_reg_candidates) * len(dropout_candidates)
    grid_count = 0
    
    for d in d_candidates:
        for test_lr in lr_candidates:
            for test_entropy_reg in entropy_reg_candidates:
                for test_dropout in dropout_candidates:
                    grid_count += 1
                    if verbose and grid_count % max(1, grid_size // 5) == 0:
                        print(f"    [{grid_count}/{grid_size}] d={d}, lr={test_lr:.1e}, ent_reg={test_entropy_reg:.1e}, drop={test_dropout}")
                    
                    # Create FSDRNN model with current hyperparameters
                    fsdrnn_model = FrechetDRNN(
                        input_dim=p,
                        n_responses=V,
                        n_ref=n_ref,
                        reduction_dim=d,
                        response_rank=r_max,
                        reduction_type="nonlinear",
                        dropout=test_dropout
                    ).to(device)
                    
                    # Create training data loader from train_subset
                    X_train_sub = torch.stack([ds_train.X[i] for i in train_subset.indices])
                    Y_train_sub = torch.stack([ds_train.Y[i] for i in train_subset.indices])
                    train_dataset_sub = TensorDataset(X_train_sub, Y_train_sub)
                    train_loader_sub = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
                    
                    # Quick training for validation
                    train_frechet_model(
                        model=fsdrnn_model,
                        Y_ref=ds_train.Y_ref,
                        train_loader=train_loader_sub,
                        dist_name=dist_name,
                        epochs=min(40, epochs // 10),  # Quick validation training
                        lr=test_lr,
                        entropy_reg=test_entropy_reg,
                        device=device,
                        verbose=False
                    )
                    
                    # Evaluate on validation set
                    val_results = evaluate_frechet_model(
                        model=fsdrnn_model,
                        Y_ref=ds_train.Y_ref,
                        test_loader=val_loader,
                        dist_name=dist_name,
                        device=device
                    )
                    
                    val_loss = val_results['avg_dist_sq']
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_d = d
                        best_lr = test_lr
                        best_entropy_reg = test_entropy_reg
                        best_dropout = test_dropout
                        best_fsdrnn_model = fsdrnn_model
    
    if verbose:
        print(f"    Best: d={best_d}, lr={best_lr:.1e}, ent_reg={best_entropy_reg:.1e}, dropout={best_dropout} (val_loss={best_val_loss:.6f})")
    
    # Reinitialize a fresh FSDRNN model with best hyperparameters and train on full training set
    fsdrnn_model = FrechetDRNN(
        input_dim=p,
        n_responses=V,
        n_ref=n_ref,
        reduction_dim=best_d,
        response_rank=r_max,
        reduction_type="nonlinear",
        dropout=best_dropout
    ).to(device)
    
    # Create full training data loader
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = Y_train.detach().clone() if isinstance(Y_train, torch.Tensor) else torch.tensor(Y_train)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Full training on fresh model with best hyperparameters
    train_frechet_model(
        model=fsdrnn_model,
        Y_ref=ds_train.Y_ref,
        train_loader=train_loader,
        dist_name=dist_name,
        epochs=epochs,
        lr=best_lr,
        entropy_reg=best_entropy_reg,
        device=device,
        verbose=verbose
    )

    # Create test data loader for evaluation
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test_tensor = Y_test.detach().clone() if isinstance(Y_test, torch.Tensor) else torch.tensor(Y_test)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate using evaluate_frechet_model
    fsdrnn_results = evaluate_frechet_model(
        model=fsdrnn_model,
        Y_ref=ds_train.Y_ref,
        test_loader=test_loader,
        dist_name=dist_name,
        device=device
    )

    fsdrnn_avg_dist = fsdrnn_results['avg_dist']
    fsdrnn_time = time.time() - t_fsdrnn

    # Weight recovery for FSDRNN
    with torch.no_grad():
        logits = fsdrnn_model(torch.tensor(X_test_np, dtype=torch.float32).to(device))
        W_pred_fsdrnn = torch.softmax(logits, dim=1)  # normalize logits to weights
        W_pred_fsdrnn = W_pred_fsdrnn.cpu().numpy()
    fsdrnn_weight_l2, fsdrnn_weight_kl = evaluate_weight_recovery(W_pred_fsdrnn, W_true_test)

    # Compute AMSPE (against true conditional means)
    fsdrnn_amspe = _evaluate_amspe_on_loader(
        fsdrnn_model, test_loader, ds_test, ds_train.Y_ref, dist_name, device
    )

    results.update({
        'fsdrnn_avg_dist': fsdrnn_avg_dist,
        'fsdrnn_avg_dist_sq': fsdrnn_results['avg_dist_sq'],
        'fsdrnn_amspe': fsdrnn_amspe,
        'fsdrnn_time_sec': fsdrnn_time,
        'fsdrnn_weight_l2': fsdrnn_weight_l2,
        'fsdrnn_weight_kl': fsdrnn_weight_kl,
        'fsdrnn_best_d': best_d,
        'fsdrnn_best_lr': best_lr,
        'fsdrnn_best_entropy_reg': best_entropy_reg,
        'fsdrnn_best_dropout': best_dropout,
        'fsdrnn_best_val_loss': best_val_loss,
    })

    if verbose:
        print(f"\n  Results Summary:")
        print(f"  True structural dimension: {d0}")
        print(f"  True response rank: {r0}")
        print(f"  Baseline avg dist: {results['baseline_avg_dist']:.6f}")
        print(f"  IFR avg dist:      {results['ifr_avg_dist']:.6f}")
        print(f"  FSDR avg dist:     {results['fsdr_avg_dist']:.6f}")
        print(f"  GFR avg dist:      {results['gfr_avg_dist']:.6f}")
        print(f"  DFR avg dist:      {results['dfr_avg_dist']:.6f}")
        print(f"  E2M avg dist:      {results['e2m_avg_dist']:.6f}")
        print(f"  FSDRNN avg dist:   {results['fsdrnn_avg_dist']:.6f}")

    return results


def convert_to_serializable(obj):
    """Convert numpy arrays and torch tensors to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def print_detailed_results(result, result_num=1, total_results=1):
    """Print detailed results with all metrics and tuned hyperparameters."""
    print(f"\n{'='*90}")
    print(f"  DETAILED RESULTS - Repetition {result_num}/{total_results}")
    print(f"{'='*90}")
    
    # Problem setup
    print(f"\n  Problem Setup:")
    print(f"    True structural dimension (d0):  {result.get('true_d0', 'N/A')}")
    print(f"    True response rank (r0):         {result.get('true_r0', 'N/A')}")
    print(f"    Number of responses (V):         {result.get('V', 'N/A')}")
    print(f"    SPD matrix dimension (m):        {result.get('m', 'N/A')}")
    print(f"    Train samples:                   {result.get('n_train', 'N/A')}")
    print(f"    Test samples:                    {result.get('n_test', 'N/A')}")
    print(f"    Reference anchors (n_ref):       {result.get('n_ref', 'N/A')}")
    
    # FSDRNN tuning
    if 'fsdrnn_best_d' in result:
        print(f"\n  FSDRNN Hyperparameter Tuning (via grid search):")
        print(f"    Selected reduction dimension (d):    {result['fsdrnn_best_d']}")
        print(f"    Selected learning rate (lr):         {result['fsdrnn_best_lr']:.1e}")
        print(f"    Selected entropy regularization:     {result['fsdrnn_best_entropy_reg']:.1e}")
        print(f"    Selected dropout rate:               {result['fsdrnn_best_dropout']}")
        print(f"    Validation loss (at selected config):{result['fsdrnn_best_val_loss']:.8f}")
    
    # Weight recovery comparison
    print(f"\n  Weight Recovery Metrics:")
    print(f"    {'Method':<15} {'L2 Distance':<20} {'KL Divergence':<20}")
    print(f"    {'-'*55}")
    
    for method in ['e2m', 'fsdrnn']:
        l2_key = f"{method}_weight_l2"
        kl_key = f"{method}_weight_kl"
        if l2_key in result and kl_key in result:
            l2_val = result[l2_key]
            kl_val = result[kl_key]
            # Handle NaN values
            if isinstance(l2_val, float) and np.isnan(l2_val):
                l2_str = "N/A"
            else:
                l2_str = f"{float(l2_val):.8f}"
            if isinstance(kl_val, float) and np.isnan(kl_val):
                kl_str = "N/A"
            else:
                kl_str = f"{float(kl_val):.8f}"
            print(f"    {method.upper():<15} {l2_str:<20} {kl_str:<20}")
    
    # Prediction performance: all methods
    print(f"\n  Prediction Performance (Average Fréchet Distance on Test Set):")
    print(f"    {'Method':<15} {'Avg Dist':<18} {'Avg Dist²':<18} {'AMSPE':<18} {'Time (sec)':<15}")
    print(f"    {'-'*85}")
    
    methods = [
        ('baseline', 'Baseline'),
        ('ifr', 'IFR'),
        ('fsdr', 'FSDR'),
        ('gfr', 'GFR'),
        ('dfr', 'DFR'),
        ('e2m', 'E2M'),
        ('fsdrnn', 'FSDRNN'),
    ]
    
    for key, label in methods:
        dist_key = f"{key}_avg_dist"
        dist_sq_key = f"{key}_avg_dist_sq"
        amspe_key = f"{key}_amspe"
        time_key = f"{key}_time_sec"
        
        if dist_key in result:
            dist_val = float(result[dist_key])
            dist_sq_val = float(result.get(dist_sq_key, 0.0))
            amspe_val = result.get(amspe_key, np.nan)
            time_val = result.get(time_key, 0.0)
            
            dist_str = f"{dist_val:.8f}"
            dist_sq_str = f"{dist_sq_val:.8f}"
            amspe_str = f"{float(amspe_val):.8f}" if isinstance(amspe_val, (int, float)) and not np.isnan(float(amspe_val)) else "N/A"
            time_str = f"{float(time_val):.4f}" if time_val else "N/A"
            
            print(f"    {label:<15} {dist_str:<18} {dist_sq_str:<18} {amspe_str:<18} {time_str:<15}")
    
    # Ranking by avg dist (lower is better)
    print(f"\n  Method Ranking (by Avg Fréchet Distance, lower is better):")
    rankings = []
    for key, label in methods:
        dist_key = f"{key}_avg_dist"
        if dist_key in result:
            rankings.append((label, float(result[dist_key])))
    
    rankings.sort(key=lambda x: x[1])
    for i, (label, dist) in enumerate(rankings, 1):
        print(f"    {i}. {label:<15} {dist:.8f}")
    
    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser(
        description="Teacher-Student FSDRNN Simulation"
    )
    parser.add_argument("--n_train", type=int, default=250)
    parser.add_argument("--n_test", type=int, default=250)
    parser.add_argument("--p", type=int, default=20, help="Predictor dimension")
    parser.add_argument("--m", type=int, default=4, help="SPD matrix dimension")
    parser.add_argument("--V", type=int, default=8, help="Number of responses")
    parser.add_argument("--d0", type=int, default=3, help="True structural dimension")
    parser.add_argument("--r0", type=int, default=2, help="True response rank")
    parser.add_argument("--n_ref", type=int, default=80, help="Number of reference anchors")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy_reg", type=float, default=0.0, help="Entropy regularization coefficient")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--output_file", type=str, default="teacher_student_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run multiple repetitions
    all_results = []
    for rep in range(args.n_reps):
        rep_seed = args.seed + rep
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"  REPETITION {rep+1}/{args.n_reps} (seed={rep_seed})")
            print(f"{'='*60}")

        result = run_teacher_student_simulation(
            n_train=args.n_train,
            n_test=args.n_test,
            p=args.p,
            m=args.m,
            V=args.V,
            d0=args.d0,
            r0=args.r0,
            n_ref=args.n_ref,
            epochs=args.epochs,
            lr=args.lr,
            entropy_reg=args.entropy_reg,
            dropout=args.dropout,
            batch_size=args.batch_size,
            device=args.device,
            verbose=args.verbose,
            seed=rep_seed,
        )
        result["rep"] = rep
        all_results.append(result)

        # Print detailed results
        print_detailed_results(result, result_num=rep+1, total_results=args.n_reps)

        print(
            f"rep {rep+1}/{args.n_reps} | "
            f"baseline d={result['baseline_avg_dist']:.4f}, d2={result['baseline_avg_dist_sq']:.4f} | "
            f"IFR d={result['ifr_avg_dist']:.4f}, d2={result['ifr_avg_dist_sq']:.4f} | "
            f"FSDR d={result['fsdr_avg_dist']:.4f}, d2={result['fsdr_avg_dist_sq']:.4f}"
        )

        print(
            f"             "
            f"GFR d={result['gfr_avg_dist']:.4f}, d2={result['gfr_avg_dist_sq']:.4f} | "
            f"DFR d={result['dfr_avg_dist']:.4f}, d2={result['dfr_avg_dist_sq']:.4f} | "
            f"E2M d={result['e2m_avg_dist']:.4f}, d2={result['e2m_avg_dist_sq']:.4f} | "
            f"FSDRNN d={result['fsdrnn_avg_dist']:.4f}, d2={result['fsdrnn_avg_dist_sq']:.4f}"
        )

    # Compute summary statistics
    if len(all_results) > 1:
        keys = [k for k in all_results[0].keys() if k != "rep" and isinstance(all_results[0][k], (int, float))]
        summary = {}
        for key in keys:
            values = [r[key] for r in all_results]
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_se"] = np.std(values) / np.sqrt(len(values))

        print(f"\n{'='*80}")
        print("  SUMMARY ACROSS REPETITIONS")
        print(f"{'='*80}")
        print(f"  Number of repetitions: {args.n_reps}")
        print(f"  True structural dimension: {args.d0}")
        print(f"  True response rank: {args.r0}")
        print(f"  FSDRNN grid search: d in [2,3,5], adaptive LoRA r_max = min(V,6) = {min(args.V, 6)}")
        print(f"  ")
        print(f"  Response prediction errors (mean ± SE):")
        methods = ['baseline', 'ifr', 'fsdr', 'gfr', 'dfr', 'e2m', 'fsdrnn']
        for method in methods:
            key = f"{method}_avg_dist"
            if key in summary:
                mean_val = summary[f"{key}_mean"]
                se_val = summary[f"{key}_se"]
                print(f"    {method.upper():8}: {mean_val:.6f} ± {se_val:.6f}")

        print(f"  ")
        print(f"  Weight recovery (L2 distance, mean ± SE):")
        weight_methods = ['e2m', 'fsdrnn']
        for method in weight_methods:
            key = f"{method}_weight_l2"
            if key in summary:
                mean_val = summary[f"{key}_mean"]
                se_val = summary[f"{key}_se"]
                print(f"    {method.upper():8}: {mean_val:.6f} ± {se_val:.6f}")

    # Save results if requested
    if args.output_file:
        try:
            import pandas as pd
            df = pd.DataFrame(all_results)
            df.to_csv(args.output_file, index=False)
            print(f"\nResults saved to {args.output_file}")
        except ImportError:
            print(f"\nWarning: pandas not available. Saving results as JSON to {args.output_file}")
            import json
            with open(args.output_file, 'w') as f:
                json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\nAll results: {all_results}")


if __name__ == "__main__":
    main()