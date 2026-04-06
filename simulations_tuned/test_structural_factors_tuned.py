#!/usr/bin/env python3
"""
Structural Factors Simulation for Fréchet Regression on SPD Manifolds

This script implements the structural factors regime with:
- n_train=400, n_test=1000, p=30 predictors
- d0=2 structural dimension
- m=4 shared factors
- V=15 responses
- Three cases: A (easier), B (main), C (stronger heterogeneity)
"""

import sys
import os
import argparse
import time
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from src.spd_frechet_adaptive import (
    FrechetDRNN,
    train_frechet_model,
    evaluate_frechet_model,
    differentiable_frechet_mean,
    entropy,
    get_distance_fn,
    GlobalFrechetRegression,
    DeepFrechetRegression,
    grid_search_frechet,
    grid_search_dfr,
)
import scipy.optimize


def local_frechet_predict_1d(t_train, Y_train, t_query, dist_name, frechet_mean_fn, h):
    """Local Fréchet prediction in 1D index space."""
    # Convert inputs to tensors
    t_train = torch.tensor(t_train, dtype=torch.float32) if not isinstance(t_train, torch.Tensor) else t_train.float()
    t_query = torch.tensor(t_query, dtype=torch.float32) if not isinstance(t_query, torch.Tensor) else t_query.float()
    h = torch.tensor(h, dtype=torch.float32) if not isinstance(h, torch.Tensor) else h.float()
    
    # Gaussian kernel weights
    w = torch.exp(-0.5 * ((t_train - t_query) / h) ** 2)
    w = w / (torch.sum(w) + 1e-12)
    
    # Convert to torch tensors
    w_torch = w.unsqueeze(0).float()  # [1, n] ensure float32
    Y_train_torch = torch.stack(Y_train).float()  # [n, p, p]
    
    # Compute weighted Fréchet mean
    pred = frechet_mean_fn(w_torch, Y_train_torch, dist_name)
    pred = pred.squeeze(0)  # [1, p, p] -> [p, p]
    return pred


def ifr_loss(beta, X, Y, dist_name, frechet_mean_fn, h):
    """Loss function for IFR optimization."""
    from src.spd_frechet_adaptive import get_distance_fn
    dist_fn = get_distance_fn(dist_name)
    
    theta = beta / (np.linalg.norm(beta) + 1e-12)
    t = X @ theta
    loss = 0.0
    
    for i in range(len(Y)):
        yhat_i = local_frechet_predict_1d(t, Y, t[i], dist_name, frechet_mean_fn, h)
        loss += (dist_fn(yhat_i, Y[i]) ** 2).item()
    
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


def predict_ifr(X_train, Y_train, theta, X_new, dist_name, frechet_mean_fn, h):
    """Predict using fitted IFR model."""
    t_train = X_train @ theta
    preds = []
    
    for x in X_new:
        t = x @ theta
        yhat = local_frechet_predict_1d(t_train, Y_train, t, dist_name, frechet_mean_fn, h)
        preds.append(yhat)
    
    return preds


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
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StructuralFactorsDataset(torch.utils.data.Dataset):
    """
    Dataset for structural factors regime.

    Parameters:
        n: number of samples
        case: 'A', 'B', or 'C' (difficulty level)
        seed: random seed
        shared_params: dict of shared structural parameters (optional)
    """

    def __init__(self, n: int, case: str = 'B', seed: int = 42, shared_params: Optional[Dict] = None):
        super().__init__()
        self.n = n
        self.case = case
        self.p = 30  # predictors
        self.d0 = 2  # structural dimension
        self.m = 4   # shared factors
        self.V = 15  # responses
        self.nu = 20.0  # Wishart degrees of freedom

        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

        # Generate predictors X ~ N(0, Σ_X) with AR(1) covariance
        rho = 0.4
        Sigma_X = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                Sigma_X[i,j] = rho ** abs(i - j)

        self.X = torch.from_numpy(
            np.random.multivariate_normal(np.zeros(self.p), Sigma_X, self.n)
        ).float()

        # Fixed orthonormal B0 (30x2)
        b1 = torch.zeros(self.p)
        b1[:5] = 1.0 / np.sqrt(5)
        b2 = torch.zeros(self.p)
        b2[5:10] = 1.0 / np.sqrt(5)
        self.B0 = torch.stack([b1, b2], dim=1)  # [30, 2]

        # Z = B0.T @ X  [2, n]
        self.Z = self.B0.T @ self.X.T  # [2, n]

        # Shared factors g(Z) [4, n]
        Z1, Z2 = self.Z[0], self.Z[1]
        g1 = Z1
        g2 = Z2**2 - 1
        g3 = torch.sin(np.pi * Z1 * Z2)
        g4_raw = torch.exp(-0.5 * (Z1**2 + Z2**2))
        g4 = g4_raw - g4_raw.mean()  # subtract mean
        self.G = torch.stack([g1, g2, g3, g4], dim=0)  # [4, n]

        # SPD basis matrices U1-U4 [4, 4, 4]
        self.U = torch.zeros(4, 4, 4)
        # U1
        self.U[0] = torch.tensor([
            [1.0, 0.30, 0.00, 0.00],
            [0.30, 0.00, 0.20, 0.00],
            [0.00, 0.20, -1.0, 0.10],
            [0.00, 0.00, 0.10, 0.00]
        ])
        # U2
        self.U[1] = torch.tensor([
            [0.00, 0.20, 0.10, 0.00],
            [0.20, 1.00, 0.00, 0.20],
            [0.10, 0.00, 0.00, 0.30],
            [0.00, 0.20, 0.30, -1.0]
        ])
        # U3
        self.U[2] = torch.tensor([
            [0.50, 0.00, 0.25, 0.10],
            [0.00, -0.50, 0.15, 0.00],
            [0.25, 0.15, 0.50, 0.20],
            [0.10, 0.00, 0.20, -0.50]
        ])
        # U4
        self.U[3] = torch.tensor([
            [0.20, 0.10, 0.00, 0.15],
            [0.10, 0.20, 0.10, 0.00],
            [0.00, 0.10, -0.20, 0.10],
            [0.15, 0.00, 0.10, -0.20]
        ])

        # Baseline S0
        self.S0 = 0.6 * torch.eye(4)

        # Use shared parameters if provided, otherwise generate new ones
        if shared_params is not None:
            self.a = shared_params['a']
            self.R = shared_params['R']
            self.c_v = shared_params['c_v']
            self.omega_v = shared_params['omega_v']
            self.d_v = shared_params['d_v']
        else:
            # Response loading structure
            # Clusters
            centers = torch.tensor([
                [1.20, 0.80, 0.20],  # cluster 1
                [0.80, -0.40, 1.00], # cluster 2
                [1.00, 0.20, -0.80]  # cluster 3
            ])

            self.a = torch.zeros(self.V, 3)
            for v in range(self.V):
                cluster = v // 5  # 0,1,2 for clusters 1,2,3
                noise = torch.randn(3) * 0.05
                self.a[v] = centers[cluster] + noise

            # Response-specific deviation matrices R_v [V, 4, 4]
            self.R = torch.zeros(self.V, 4, 4)
            for v in range(self.V):
                Q = torch.randn(4, 4)
                R_v = (Q + Q.T) / 2
                R_v = R_v / torch.norm(R_v, 'fro')
                self.R[v] = R_v

            # Response-specific deviation functions delta_v
            self.c_v = torch.rand(self.V) * 0.1 + 0.05  # Unif(0.05, 0.15)
            self.omega_v = torch.rand(self.V) * 0.7 + 0.8  # Unif(0.8, 1.5)
            self.d_v = torch.zeros(self.V)
            for v in range(self.V):
                if v % 3 == 0:
                    self.d_v[v] = -0.05
                elif v % 3 == 1:
                    self.d_v[v] = 0.0
                else:
                    self.d_v[v] = 0.05

        # Factor loadings b_v [V, 4] (computed from a)
        self.b = torch.zeros(self.V, 4)
        self.b[:, 0] = self.a[:, 0]  # b_v1 = a_v1
        self.b[:, 1] = self.a[:, 1]  # b_v2 = a_v2
        self.b[:, 2] = self.a[:, 2]  # b_v3 = a_v3
        self.b[:, 3] = 0.5 * (self.a[:, 0] + self.a[:, 1])  # b_v4 = 0.5*(a_v1 + a_v2)

        # Adjust for case
        if case == 'A':
            self.delta_coeff = 0.0
        elif case == 'B':
            self.delta_coeff = 0.25
        elif case == 'C':
            self.delta_coeff = 0.35
        else:
            raise ValueError(f"Unknown case: {case}")

        # Generate responses Y [n, V, 4, 4]
        self.Y = torch.zeros(self.n, self.V, 4, 4)
        self.true_mean = torch.zeros(self.n, self.V, 4, 4)  # Store nu * Sigma_v for AMSPE

        for i in range(self.n):
            for v in range(self.V):
                # Compute S_v
                S_v = self.S0.clone()
                for k in range(4):
                    coeff = [0.6, 0.5, 0.5, 0.3][k]
                    S_v += coeff * self.b[v, k] * self.G[k, i] * self.U[k]

                # Add deviation
                if self.delta_coeff > 0:
                    delta_v = self.c_v[v] * torch.sin(self.omega_v[v] * self.Z[0, i]) + self.d_v[v] * self.Z[1, i]
                    S_v += self.delta_coeff * delta_v * self.R[v]

                # Sigma_v = exp(S_v)
                Sigma_v = torch.matrix_exp(S_v)

                # Store true mean: nu * Sigma_v
                self.true_mean[i, v] = self.nu * Sigma_v

                # Sample Y_v ~ Wishart(nu, Sigma_v)
                Y_v = self.sample_wishart(Sigma_v, self.nu)
                self.Y[i, v] = Y_v

    def sample_wishart(self, Sigma: torch.Tensor, df: int) -> torch.Tensor:
        """Sample from Wishart distribution W(Sigma, df) using stable Bartlett decomposition."""
        # Ensure Sigma is SPD before sampling
        Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        eigvals = eigvals.clamp(min=1e-4)  # floor eigenvalues
        Sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        
        # Use Bartlett decomposition for stability
        p = Sigma.shape[-1]
        assert df >= p, f"df={df} must be >= p={p} for Wishart"
        L = torch.linalg.cholesky(Sigma)  # [p, p]

        # Bartlett decomposition
        A = torch.zeros(p, p, dtype=Sigma.dtype, device=Sigma.device)
        for i in range(p):
            # Diagonal: sqrt of chi-squared with (df - i) degrees of freedom
            chi2 = torch.distributions.Chi2(df=df - i)
            A[i, i] = chi2.sample().sqrt()
            # Below diagonal: standard normal
            for j in range(i):
                A[i, j] = torch.randn(1, dtype=Sigma.dtype, device=Sigma.device).squeeze()
        # Y = L A A^T L^T
        LA = L @ A
        Y = LA @ LA.t()
        return Y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def run_single_case(
    case: str,
    n_train: int = 400,
    n_test: int = 1000,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
    dfr_tuned_params: Optional[Dict] = None,
    fdrnn_tuned_params: Optional[Dict] = None,
    seed: int = 42,
    metric: str = "frobenius",
    skip_ifr: bool = False,
) -> Dict:
    """
    Run single case of structural factors simulation.
    """
    # Set seed immediately at the start
    set_seed(seed)
    
    p = 30
    V = 15
    d0 = 2

    if verbose:
        print(f"\n{'='*80}")
        print(f"  Structural Factors Case {case}")
        print(f"  n_train={n_train}, n_test={n_test}, p={p}, V={V}, d0={d0}")
        print(f"  epochs={epochs}, device={device}, seed={seed}")
        print(f"{'='*80}")

    # Generate shared structural parameters once for this repetition
    # (seed already set above)
    
    # Response loading structure
    centers = torch.tensor([
        [1.20, 0.80, 0.20],  # cluster 1
        [0.80, -0.40, 1.00], # cluster 2
        [1.00, 0.20, -0.80]  # cluster 3
    ])

    a = torch.zeros(V, 3)
    for v in range(V):
        cluster = v // 5  # 0,1,2 for clusters 1,2,3
        noise = torch.randn(3) * 0.05
        a[v] = centers[cluster] + noise

    # Response-specific deviation matrices R_v [V, 4, 4]
    R = torch.zeros(V, 4, 4)
    for v in range(V):
        Q = torch.randn(4, 4)
        R_v = (Q + Q.T) / 2
        R_v = R_v / torch.norm(R_v, 'fro')
        R[v] = R_v

    # Response-specific deviation functions delta_v
    c_v = torch.rand(V) * 0.1 + 0.05  # Unif(0.05, 0.15)
    omega_v = torch.rand(V) * 0.7 + 0.8  # Unif(0.8, 1.5)
    d_v = torch.zeros(V)
    for v in range(V):
        if v % 3 == 0:
            d_v[v] = -0.05
        elif v % 3 == 1:
            d_v[v] = 0.0
        else:
            d_v[v] = 0.05

    shared_params = {
        'a': a,
        'R': R,
        'c_v': c_v,
        'omega_v': omega_v,
        'd_v': d_v,
    }

    # Generate data with shared parameters
    ds_train = StructuralFactorsDataset(n_train, case=case, seed=seed, shared_params=shared_params)
    ds_test = StructuralFactorsDataset(n_test, case=case, seed=seed + 1000, shared_params=shared_params)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Reference set and true means for AMSPE
    Y_ref = ds_train.Y
    true_mean_train = ds_train.true_mean
    true_mean_test = ds_test.true_mean

    # For this setup, use the specified distance for matrices
    dist_name = metric
    dist_fn = get_distance_fn(dist_name)

    # Global mean baseline
    global_mean = ds_train.Y.mean(dim=0)  # [V, 4, 4]

    baseline_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_batch = Y_batch.to(device)
            B = X_batch.size(0)
            gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1).to(device)  # [B, V, 4, 4]
            # Distance per sample, averaged over responses
            d_per_response = []
            for v in range(V):
                d_v = dist_fn(gm[:, v, :, :], Y_batch[:, v, :, :])  # [B]
                d_per_response.append(d_v)
            d = torch.stack(d_per_response).mean(dim=0)  # [B]
            baseline_dists.append(d.cpu())

    baseline_dists_cat = torch.cat(baseline_dists)
    base_avg = baseline_dists_cat.mean().item()
    base_avg_sq = (baseline_dists_cat ** 2).mean().item()

    # AMSPE for baseline: squared distance between global_mean and true_mean
    # Only valid for Frobenius metric where true mean calculation is correct
    if dist_name == "frobenius":
        baseline_amspe_dists = []
        true_mean_test = ds_test.true_mean
        with torch.no_grad():
            for i in range(len(ds_test)):
                true_mean_i = true_mean_test[i].to(device)  # [V, 4, 4]
                gm = global_mean.unsqueeze(0).expand(1, -1, -1, -1).to(device)  # [1, V, 4, 4]
                d_per_response = []
                for v in range(V):
                    d_v = dist_fn(gm[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                baseline_amspe_dists.append(d.cpu())

        baseline_amspe_dists_cat = torch.cat(baseline_amspe_dists)
        base_amspe = baseline_amspe_dists_cat.mean().item()
    else:
        base_amspe = None

    results = {
        "case": case,
        "baseline_avg_dist": base_avg,
        "baseline_avg_dist_sq": base_avg_sq,
        "baseline_amspe": base_amspe,
        "baseline_time_sec": 0.0
    }

    if verbose:
        print(f"\n  --- Competitors ---")
        print(f"  Global Mean avg Dist:   {base_avg:.6f}")

    # GFR: Global Frechet Regression
    if verbose:
        print(f"\n  --- Global Frechet Regression (GFR) ---")
    t_gfr = time.time()
    gfr = GlobalFrechetRegression(dist_name=dist_name)
    gfr.fit(ds_train.X, ds_train.Y)

    gfr_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            Y_pred_gfr = gfr.predict(X_batch)  # [B, V, 4, 4]
            # Distance per sample, averaged over responses
            d_per_response = []
            for v in range(V):
                d_v = dist_fn(Y_pred_gfr[:, v, :, :], Y_batch[:, v, :, :])  # [B]
                d_per_response.append(d_v)
            d = torch.stack(d_per_response).mean(dim=0)  # [B]
            gfr_dists.append(d.cpu())

    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_avg_sq = (gfr_dists_cat ** 2).mean().item()
    gfr_time = time.time() - t_gfr

    # AMSPE for GFR
    # Only valid for Frobenius metric where true mean calculation is correct
    if dist_name == "frobenius":
        gfr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                X_i = ds_test.X[i:i+1].to(device)  # [1, p]
                true_mean_i = true_mean_test[i].to(device)  # [V, 4, 4]
                Y_pred_gfr_i = gfr.predict(X_i)  # [1, V, 4, 4]
                d_per_response = []
                for v in range(V):
                    d_v = dist_fn(Y_pred_gfr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                gfr_amspe_dists.append(d.cpu())

        gfr_amspe_dists_cat = torch.cat(gfr_amspe_dists)
        gfr_amspe = gfr_amspe_dists_cat.mean().item()
    else:
        gfr_amspe = None

    results["gfr_avg_dist"] = gfr_avg
    results["gfr_avg_dist_sq"] = gfr_avg_sq
    results["gfr_amspe"] = gfr_amspe
    results["gfr_time_sec"] = gfr_time

    if verbose:
        print(f"  GFR avg Dist:   {gfr_avg:.6f} (time: {gfr_time:.1f}s)")

    # DFR: Deep Frechet Regression
    if verbose:
        print(f"\n  --- Deep Frechet Regression (DFR) ---")
    t_dfr = time.time()
    
    # Use tuned parameters if provided, otherwise defaults
    dfr_params = {
        'dist_name': dist_name,
        'manifold_method': "isomap",
        'manifold_dim': 2,
        'manifold_k': 10,
        'hidden': 32,
        'layer': 4,
        'num_epochs': epochs,
        'lr': lr,
        'dropout': 0.3,
        'seed': seed,
    }
    if dfr_tuned_params is not None:
        dfr_params.update(dfr_tuned_params)
    
    dfr = DeepFrechetRegression(**dfr_params)
    dfr.fit(ds_train.X, ds_train.Y, verbose=verbose)

    dfr_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            Y_pred_dfr = dfr.predict(X_batch)  # [B, V, 4, 4]
            # Distance per sample, averaged over responses
            d_per_response = []
            for v in range(V):
                d_v = dist_fn(Y_pred_dfr[:, v, :, :], Y_batch[:, v, :, :])  # [B]
                d_per_response.append(d_v)
            d = torch.stack(d_per_response).mean(dim=0)  # [B]
            dfr_dists.append(d.cpu())

    dfr_dists_cat = torch.cat(dfr_dists)
    dfr_avg = dfr_dists_cat.mean().item()
    dfr_avg_sq = (dfr_dists_cat ** 2).mean().item()
    dfr_time = time.time() - t_dfr

    # AMSPE for DFR
    # Only valid for Frobenius metric where true mean calculation is correct
    if dist_name == "frobenius":
        dfr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                X_i = ds_test.X[i:i+1].to(device)  # [1, p]
                true_mean_i = true_mean_test[i].to(device)  # [V, 4, 4]
                Y_pred_dfr_i = dfr.predict(X_i.to(device))  # [1, V, 4, 4]
                d_per_response = []
                for v in range(V):
                    d_v = dist_fn(Y_pred_dfr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                dfr_amspe_dists.append(d.cpu())

        dfr_amspe_dists_cat = torch.cat(dfr_amspe_dists)
        dfr_amspe = dfr_amspe_dists_cat.mean().item()
    else:
        dfr_amspe = None

    results["dfr_avg_dist"] = dfr_avg
    results["dfr_avg_dist_sq"] = dfr_avg_sq
    results["dfr_amspe"] = dfr_amspe
    results["dfr_time_sec"] = dfr_time

    if verbose:
        print(f"  DFR avg Dist:   {dfr_avg:.6f} (time: {dfr_time:.1f}s)")

    # E2M: Extrinsic 2M baseline (FSDRNN with d=p, no response coupling, linear reduction)
    if verbose:
        print(f"\n  --- E2M baseline (FSDRNN d=p, no coupling) ---")
    t_e2m = time.time()

    e2m_model = FrechetDRNN(
        input_dim=p,
        n_ref=len(ds_train),
        n_responses=V,
        reduction_dim=p,  # no reduction
        response_rank=None,  # no response coupling
        response_alpha=1.0,
        reduction_type="linear",  # simpler linear reduction for baseline
        encoder_sizes=[128, 64],
        head_sizes=[64, 128],
        activation="relu",
        dropout=0.3,
    )

    # Train the model
    e2m_history = train_frechet_model(
        model=e2m_model,
        Y_ref=Y_ref,
        train_loader=train_loader,
        dist_name=dist_name,
        epochs=epochs,
        lr=lr,
        entropy_reg=0.01,
        nuclear_reg=0.0,  # no reduction regularization for baseline
        device=device,
        verbose=False,
    )

    e2m_results = evaluate_frechet_model(
        model=e2m_model,
        Y_ref=Y_ref,
        test_loader=test_loader,
        dist_name=dist_name,
        device=device,
    )
    e2m_avg = e2m_results["avg_dist"]
    e2m_avg_sq = e2m_results["avg_dist_sq"]

    e2m_time = time.time() - t_e2m

    # AMSPE for E2M
    # Only valid for Frobenius metric where true mean calculation is correct
    if dist_name == "frobenius":
        e2m_amspe_results = evaluate_frechet_model(
            model=e2m_model,
            Y_ref=Y_ref,
            test_loader=test_loader,
            dist_name=dist_name,
            device=device,
            true_means=true_mean_test,
        )
        e2m_amspe = e2m_amspe_results["avg_dist_sq_to_mean"]
    else:
        e2m_amspe = None

    results["e2m_avg_dist"] = e2m_avg
    results["e2m_avg_dist_sq"] = e2m_avg_sq
    results["e2m_amspe"] = e2m_amspe
    results["e2m_time_sec"] = e2m_time

    if verbose:
        print(f"  E2M avg Dist:         {e2m_avg:.6f} (time: {e2m_time:.1f}s)")

    # FDRNN: Frechet Deep RNN with adaptive LoRA
    if fdrnn_tuned_params is not None:
        if verbose:
            print(f"\n  --- FDRNN (d={fdrnn_tuned_params['reduction_dim']}, adaptive LoRA) ---")
        t_fdrnn = time.time()

        fdrnn_model = FrechetDRNN(
            input_dim=p,
            n_ref=len(ds_train),
            n_responses=V,
            reduction_dim=fdrnn_tuned_params["reduction_dim"],
            response_rank=fdrnn_tuned_params["response_rank"],
            response_alpha=1.0,
            reduction_type=fdrnn_tuned_params["reduction_type"],
            encoder_sizes=fdrnn_tuned_params["encoder_sizes"],
            head_sizes=fdrnn_tuned_params["head_sizes"],
            activation="relu",
            dropout=fdrnn_tuned_params["dropout"],
        )

        # Train the model (modifies in place) and get training history
        fdrnn_history = train_frechet_model(
            model=fdrnn_model,
            Y_ref=Y_ref,
            train_loader=train_loader,
            dist_name=dist_name,
            epochs=epochs,
            lr=fdrnn_tuned_params["lr"],
            entropy_reg=fdrnn_tuned_params["entropy_reg"],
            nuclear_reg=fdrnn_tuned_params["nuclear_reg"],
            device=device,
            verbose=False,  # less verbose for individual runs
        )

        fdrnn_results = evaluate_frechet_model(
            model=fdrnn_model,
            Y_ref=Y_ref,
            test_loader=test_loader,
            dist_name=dist_name,
            device=device,
        )
        fdrnn_avg = fdrnn_results["avg_dist"]
        fdrnn_avg_sq = fdrnn_results["avg_dist_sq"]

        fdrnn_time = time.time() - t_fdrnn

        # AMSPE for FDRNN
        # Only valid for Frobenius metric where true mean calculation is correct
        if dist_name == "frobenius":
            fdrnn_amspe_results = evaluate_frechet_model(
                model=fdrnn_model,
                Y_ref=Y_ref,
                test_loader=test_loader,
                dist_name=dist_name,
                device=device,
                true_means=true_mean_test,
            )
            fdrnn_amspe = fdrnn_amspe_results["avg_dist_sq_to_mean"]
        else:
            fdrnn_amspe = None

        results["fdrnn_avg_dist"] = fdrnn_avg
        results["fdrnn_avg_dist_sq"] = fdrnn_avg_sq
        results["fdrnn_amspe"] = fdrnn_amspe
        results["fdrnn_time_sec"] = fdrnn_time

        if verbose:
            print(f"  FDRNN avg Dist:         {fdrnn_avg:.6f}")
            print(f"  FDRNN Improvement:   {(base_avg - fdrnn_avg) / base_avg * 100:.1f}%")
            print(f"  FDRNN time:          {fdrnn_time:.1f}s")

    # Prepare numpy arrays for IFR and Fréchet SDR
    X_train_np = ds_train.X.numpy()
    X_test_np = ds_test.X.numpy()
    h = 1.06 * X_train_np.std() * (len(X_train_np)) ** (-1/5)  # Bandwidth for kernel

    # IFR: Single-index Fréchet Regression (response-by-response)
    if skip_ifr:
        ifr_avg = None
        ifr_avg_sq = None
        ifr_amspe = None
        ifr_time = 0.0
    else:
        if verbose:
            print(f"\n  --- IFR (Single-index Fréchet Regression) ---")
        t_ifr = time.time()
        
        # Convert data to numpy for IFR (already done above)
        
        ifr_preds = []
        ifr_failed = False
        for v in range(V):
            if verbose and v == 0:
                print(f"    Fitting IFR for response {v+1}/{V}...")
            
            try:
                # Extract response v
                Y_train_v = [ds_train.Y[i, v] for i in range(len(ds_train))]
                Y_test_v = [ds_test.Y[i, v] for i in range(len(ds_test))]
                
                # Fit IFR for this response
                theta_v = fit_ifr(X_train_np, Y_train_v, dist_name, differentiable_frechet_mean, h)
                
                # Predict
                preds_v = predict_ifr(X_train_np, Y_train_v, theta_v, X_test_np, dist_name, differentiable_frechet_mean, h)
                ifr_preds.append(preds_v)
            except Exception as e:
                if verbose and v == 0:
                    print(f"    IFR failed for response {v+1}: {e}")
                ifr_failed = True
                break
        
        if ifr_failed:
            ifr_avg = None
            ifr_avg_sq = None
            ifr_amspe = None
            ifr_time = 0.0
        else:
            # Stack predictions: [n_test, V, 4, 4]
            ifr_preds_stacked = torch.stack([torch.stack(ifr_preds[v]) for v in range(V)], dim=1)
            
            ifr_dists = []
            with torch.no_grad():
                for i in range(len(ds_test)):
                    Y_true_i = ds_test.Y[i].unsqueeze(0).to(device)  # [1, V, 4, 4]
                    Y_pred_ifr_i = ifr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 4, 4]
                    
                    d_per_response = []
                    for v in range(V):
                        d_v = dist_fn(Y_pred_ifr_i[:, v, :, :], Y_true_i[:, v, :, :])  # [1]
                        d_per_response.append(d_v)
                    d = torch.stack(d_per_response).mean(dim=0)  # [1]
                    ifr_dists.append(d.cpu())
            
            ifr_dists_cat = torch.cat(ifr_dists)
            ifr_avg = ifr_dists_cat.mean().item()
            ifr_avg_sq = (ifr_dists_cat ** 2).mean().item()
            ifr_time = time.time() - t_ifr
            
            # AMSPE for IFR (only for Frobenius)
            if dist_name == "frobenius":
                ifr_amspe_dists = []
                with torch.no_grad():
                    for i in range(len(ds_test)):
                        true_mean_i = true_mean_test[i].to(device)  # [V, 4, 4]
                        Y_pred_ifr_i = ifr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 4, 4]
                        
                        d_per_response = []
                        for v in range(V):
                            d_v = dist_fn(Y_pred_ifr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                            d_per_response.append(d_v ** 2)  # Square the distance
                        d = torch.stack(d_per_response).mean(dim=0)  # [1]
                        ifr_amspe_dists.append(d.cpu())
                
                ifr_amspe_dists_cat = torch.cat(ifr_amspe_dists)
                ifr_amspe = ifr_amspe_dists_cat.mean().item()
            else:
                ifr_amspe = None
    
    results["ifr_avg_dist"] = ifr_avg
    results["ifr_avg_dist_sq"] = ifr_avg_sq if ifr_avg_sq is not None else None
    results["ifr_amspe"] = ifr_amspe
    results["ifr_time_sec"] = ifr_time
    
    if verbose and ifr_avg is not None:
        print(f"  IFR avg Dist:          {ifr_avg:.6f} (time: {ifr_time:.1f}s)")

    # Fréchet SDR (response-by-response)
    if verbose:
        print(f"\n  --- Fréchet SDR (SIR ensemble) ---")
    t_fsdr = time.time()
    
    fsdr_preds = []
    for v in range(V):
        if verbose and v == 0:
            print(f"    Fitting Fréchet SDR for response {v+1}/{V}...")
        
        # Extract response v
        Y_train_v = [ds_train.Y[i, v] for i in range(len(ds_train))]
        Y_test_v = [ds_test.Y[i, v] for i in range(len(ds_test))]
        
        # Fit Fréchet SDR for this response
        B_hat, _ = frechet_sdr(X_train_np, Y_train_v, get_distance_fn(dist_name), 1)
        theta_v = B_hat[:, 0]
        
        # Predict using local Fréchet regression along the SDR direction
        preds_v = predict_ifr(X_train_np, Y_train_v, theta_v, X_test_np, dist_name, differentiable_frechet_mean, h)
        fsdr_preds.append(preds_v)
    
    # Stack predictions: [n_test, V, 4, 4]
    fsdr_preds_stacked = torch.stack([torch.stack(fsdr_preds[v]) for v in range(V)], dim=1)
    
    fsdr_dists = []
    with torch.no_grad():
        for i in range(len(ds_test)):
            Y_true_i = ds_test.Y[i].unsqueeze(0).to(device)  # [1, V, 4, 4]
            Y_pred_fsdr_i = fsdr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 4, 4]
            
            d_per_response = []
            for v in range(V):
                d_v = dist_fn(Y_pred_fsdr_i[:, v, :, :], Y_true_i[:, v, :, :])  # [1]
                d_per_response.append(d_v)
            d = torch.stack(d_per_response).mean(dim=0)  # [1]
            fsdr_dists.append(d.cpu())
    
    fsdr_dists_cat = torch.cat(fsdr_dists)
    fsdr_avg = fsdr_dists_cat.mean().item()
    fsdr_avg_sq = (fsdr_dists_cat ** 2).mean().item()
    fsdr_time = time.time() - t_fsdr
    
    # AMSPE for Fréchet SDR (only for Frobenius)
    if dist_name == "frobenius":
        fsdr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                true_mean_i = true_mean_test[i].to(device)  # [V, 4, 4]
                Y_pred_fsdr_i = fsdr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 4, 4]
                
                d_per_response = []
                for v in range(V):
                    d_v = dist_fn(Y_pred_fsdr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                fsdr_amspe_dists.append(d.cpu())
        
        fsdr_amspe_dists_cat = torch.cat(fsdr_amspe_dists)
        fsdr_amspe = fsdr_amspe_dists_cat.mean().item()
    else:
        fsdr_amspe = None
    
    results["fsdr_avg_dist"] = fsdr_avg
    results["fsdr_avg_dist_sq"] = fsdr_avg_sq
    results["fsdr_amspe"] = fsdr_amspe
    results["fsdr_time_sec"] = fsdr_time
    
    if verbose:
        print(f"  Fréchet SDR avg Dist:   {fsdr_avg:.6f} (time: {fsdr_time:.1f}s)")

    # Summary table
    if verbose:
        print(f"\n{'='*80}")
        print(f"  COMPARISON TABLE  (Case {case}, d0={d0}, metric={dist_name})")
        print(f"{'='*80}")
        print(f"  {'Method':<30} {'avg Dist':>14} {'AMSPE':>14}")
        print(f"  {'-'*61}")

        def _imp(val):
            return (base_avg - val) / base_avg * 100 if base_avg > 0 else 0

        def _format_amspe(amspe_val):
            return f"{amspe_val:>14.6f}" if amspe_val is not None else f"{'N/A':>14}"

        print(f"  {'Global Mean':<30} {base_avg:>14.6f} {_format_amspe(base_amspe)}")
        print(f"  {'GFR':<30} {gfr_avg:>14.6f} {_format_amspe(gfr_amspe)}")
        print(f"  {'DFR':<30} {dfr_avg:>14.6f} {_format_amspe(dfr_amspe)}")
        print(f"  {'E2M':<30} {e2m_avg:>14.6f} {_format_amspe(e2m_amspe)}")
        print(f"  {'IFR':<30} {ifr_avg if ifr_avg is not None else 'N/A':>14} {_format_amspe(ifr_amspe)}")
        print(f"  {'Fréchet SDR':<30} {fsdr_avg:>14.6f} {_format_amspe(fsdr_amspe)}")
        if fdrnn_tuned_params is not None:
            print(f"  FDRNN (d={fdrnn_tuned_params['reduction_dim']}, adaptive)".ljust(30) + f" {fdrnn_avg:>14.6f} {_format_amspe(fdrnn_amspe)}")

        # Print AMSPE interpretation note
        if dist_name == "frobenius":
            print(f"\n  Note: AMSPE (Average Mean Squared Prediction Error) measures prediction accuracy")
            print(f"        against true conditional means m(X), providing a fair comparison of methods.")
        else:
            print(f"\n  Note: AMSPE only computed for Frobenius metric (true mean calculation valid).")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Structural Factors Simulation (Pre-tuned)"
    )
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=["A", "B", "C", "all"],
        help="Case: A (easier), B (main), C (stronger FDRNN advantage), or 'all' to run all cases"
    )
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=20, help="Number of repetitions")
    parser.add_argument("--output_file", type=str, default="structural_factors_tuned_results.csv")
    parser.add_argument("--metric", type=str, default="frobenius", choices=["frobenius", "power"], help="Distance metric")
    parser.add_argument("--skip_ifr", action="store_true", help="Skip IFR method (useful for faster testing)")

    args = parser.parse_args()

    # Determine which cases to run
    if args.case == "all":
        cases_to_run = ["A", "B", "C"]
    else:
        cases_to_run = [args.case]

    # ==============================================================
    # Pre-tuned hyperparameters (no hyperparameter tuning)
    # ==============================================================
    print(f"Using pre-tuned hyperparameters for cases {cases_to_run}, metric {args.metric}...")
    print(f"  METRIC: {args.metric}  |  {args.n_reps} repetitions per case")
    print(f"  reduction dim d=2 (pre-tuned)")
    print(f"  DFR pre-tuned:       {{'hidden': 16, 'layer': 3, 'lr': 0.0005, 'manifold_dim': 2}}")
    print(f"  FSDRNN (d=2, adaptive r) pre-tuned:  {{'dropout': 0.3, 'entropy_reg': 0.0, 'lr': 0.001, 'nuclear_reg': 0.0001, 'reduction_dim': 2, 'reduction_type': 'nonlinear', 'response_rank': 5}}")
    print(f"######################################################################")

    # Pre-tuned parameters
    dfr_tuned_params = {'hidden': 16, 'layer': 3, 'lr': 0.0005, 'manifold_dim': 2}
    fdrnn_tuned_params = {
        'dropout': 0.3,
        'entropy_reg': 0.0,
        'lr': 0.001,
        'nuclear_reg': 0.0001,
        'reduction_dim': 2,
        'reduction_type': 'nonlinear',
        'response_rank': 5,
        'encoder_sizes': [128, 64],  # fixed architecture
        'head_sizes': [64, 128]      # fixed architecture
    }

    # Store results for all cases
    all_case_results = {}

    for case in cases_to_run:
        print(f"\n======================================================================")
        print(f"  Case {case}: {args.metric}")
        print(f"  n_train={args.n_train}, n_test={args.n_test}, df=6, seed=42")
        print(f"  epochs={args.epochs}, lr={args.lr}, d0=5")
        print(f"  [TUNED DFR params: {dfr_tuned_params}]")
        print(f"  [TUNED FSDRNN params: {fdrnn_tuned_params}]")
        print(f"======================================================================")

        # ==============================================================
        # Main experiment repetitions for this case (no hyperparameter tuning)
        # ==============================================================
        case_results = []

        for rep in range(args.n_reps):
            rep_seed = 42 + rep
            print(f"\nRepetition {rep + 1}/{args.n_reps} (seed={rep_seed})")

            result = run_single_case(
                case=case,
                n_train=args.n_train,
                n_test=args.n_test,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=args.device,
                verbose=True,
                dfr_tuned_params=dfr_tuned_params,
                fdrnn_tuned_params=fdrnn_tuned_params,
                seed=rep_seed,
                metric=args.metric,
                skip_ifr=args.skip_ifr,
            )
            result["rep"] = rep
            case_results.append(result)

        all_case_results[case] = case_results

        # Save results for this case
        case_output_file = args.output_file.replace('.csv', f'_case_{case}.csv')
        import csv
        with open(case_output_file, 'w', newline='') as csvfile:
            fieldnames = ["case", "rep", "baseline_avg_dist", "baseline_avg_dist_sq", "baseline_amspe", "baseline_time_sec",
                          "gfr_avg_dist", "gfr_avg_dist_sq", "gfr_amspe", "gfr_time_sec",
                          "dfr_avg_dist", "dfr_avg_dist_sq", "dfr_amspe", "dfr_time_sec",
                          "e2m_avg_dist", "e2m_avg_dist_sq", "e2m_amspe", "e2m_time_sec",
                          "ifr_avg_dist", "ifr_avg_dist_sq", "ifr_amspe", "ifr_time_sec",
                          "fsdr_avg_dist", "fsdr_avg_dist_sq", "fsdr_amspe", "fsdr_time_sec"]
            if fdrnn_tuned_params is not None:
                fieldnames.extend(["fdrnn_avg_dist", "fdrnn_avg_dist_sq", "fdrnn_amspe", "fdrnn_time_sec"])
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in case_results:
                writer.writerow(result)

        print(f"\nResults for case {case} saved to {case_output_file}")

        # Print averages with SE for this case
        print(f"\nAverage results for case {case} with SE:")
        for key in sorted(case_results[0].keys()):
            if key in ["case", "rep"]:
                continue
            values = [r[key] for r in case_results if key in r and isinstance(r[key], (int, float))]
            if values:
                mean = np.mean(values)
                se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                print(f"{key}: {mean:.6f} ± {se:.6f}")

    # ==============================================================
    # Create comparative AMSPE plots for all cases
    # ==============================================================
    print("\nGenerating comparative AMSPE plots...")
    try:
        import matplotlib.pyplot as plt

        # Define methods and their keys
        amspe_keys = ["baseline_amspe", "gfr_amspe", "dfr_amspe", "e2m_amspe", "ifr_amspe", "fsdr_amspe"]
        method_names = ["Global Mean", "GFR", "DFR", "E2M", "IFR", "Fréchet SDR"]

        if fdrnn_tuned_params is not None:
            amspe_keys.append("fdrnn_amspe")
            method_names.append("FDRNN")

        # Filter out methods that have no valid AMSPE values across all cases
        valid_methods = []
        valid_keys = []
        for key, name in zip(amspe_keys, method_names):
            has_valid_values = False
            for case in cases_to_run:
                case_results = all_case_results[case]
                values = [r[key] for r in case_results if key in r and r[key] is not None]
                if values:
                    has_valid_values = True
                    break
            if has_valid_values:
                valid_keys.append(key)
                valid_methods.append(name)

        # Colors for different cases
        case_colors = {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral'}

        # Create figure with subplots for each method
        n_methods = len(valid_methods)
        if n_methods > 0:
            fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 6))
            if n_methods == 1:
                axes = [axes]

            for i, (key, method_name) in enumerate(zip(valid_keys, valid_methods)):
                ax = axes[i]

                # Collect data for each case
                case_names = []
                means = []
                ses = []

                for case in cases_to_run:
                    case_results = all_case_results[case]
                    values = [r[key] for r in case_results if key in r and r[key] is not None]
                    if values:
                        mean = np.mean(values)
                        se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                        case_names.append(f"Case {case}")
                        means.append(mean)
                        ses.append(se)

                if case_names:
                    # Create bar chart
                    bars = ax.bar(case_names, means, yerr=ses, capsize=3,
                                 color=[case_colors[case.split()[-1]] for case in case_names])

                    ax.set_title(f'{method_name}')
                    ax.set_ylabel('AMSPE' if i == 0 else '')
                    ax.grid(axis='y', alpha=0.3)

                    # Add value labels on bars
                    for bar, mean, se in zip(bars, means, ses):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + se + max(ses)*0.05,
                               '.4f', ha='center', va='bottom', fontsize=8)

            plt.suptitle(f'AMSPE Comparison Across Cases ({args.metric})', fontsize=14)
            plt.tight_layout()
            plot_file = args.output_file.replace('.csv', '_amspe_comparison.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Comparative AMSPE plot saved to {plot_file}")
            plt.close()

            # Also create a single combined plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Set up the bar positions
            n_cases = len(cases_to_run)
            bar_width = 0.8 / n_methods
            case_positions = np.arange(n_cases)

            for i, (key, method_name) in enumerate(zip(valid_keys, valid_methods)):
                means = []
                ses = []

                for case in cases_to_run:
                    case_results = all_case_results[case]
                    values = [r[key] for r in case_results if key in r and r[key] is not None]
                    if values:
                        mean = np.mean(values)
                        se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                        means.append(mean)
                        ses.append(se)
                    else:
                        means.append(0)
                        ses.append(0)

                # Position for this method within each case
                positions = case_positions + i * bar_width

                ax.bar(positions, means, bar_width, yerr=ses, capsize=3,
                       label=method_name, alpha=0.8)

            ax.set_xlabel('Case')
            ax.set_ylabel('AMSPE (Mean Squared Prediction Error)')
            ax.set_title(f'AMSPE Comparison Across Cases ({args.metric})')
            ax.set_xticks(case_positions + bar_width * (n_methods - 1) / 2)
            ax.set_xticklabels([f'Case {case}' for case in cases_to_run])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            combined_plot_file = args.output_file.replace('.csv', '_amspe_combined.png')
            plt.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
            print(f"Combined AMSPE plot saved to {combined_plot_file}")
            plt.close()
        else:
            print("No AMSPE data available for plotting")

    except ImportError:
        print("matplotlib not available, skipping AMSPE plots")
    except Exception as e:
        print(f"Error creating AMSPE plots: {e}")


# ============================================================================
# IFR and Fréchet SDR Functions
# ============================================================================

def ifr_loss(theta, X, Y, dist_name, differentiable_frechet_mean, h):
    """
    Loss function for IFR: sum of squared distances from local predictions to true responses.
    
    Args:
        theta: [p] parameter vector
        X: [n, p] predictors
        Y: list of [4, 4] SPD matrices
        dist_name: distance metric
        differentiable_frechet_mean: Fréchet mean function
        h: kernel bandwidth
    
    Returns:
        scalar loss
    """
    n = len(X)
    loss = torch.tensor(0.0, requires_grad=True)  # Initialize as tensor requiring grad
    
    # Convert X to tensor if it's numpy
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    
    # Compute projected values for all training points
    t_train = X @ theta  # [n] projected 1D values
    
    # Convert to numpy for the prediction function
    t_train_np = t_train.detach().numpy()
    
    for i in range(n):
        # Get the projected value for this test point
        t_query = t_train[i]  # tensor
        
        # Compute kernel weights
        w = torch.exp(-0.5 * ((t_train - t_query) / h) ** 2)
        w = w / (torch.sum(w) + 1e-12)
        w_torch = w.unsqueeze(0)
        
        # Convert Y to tensor
        Y_train_torch = torch.stack(Y).float()
        
        # Compute weighted Fréchet mean
        pred = differentiable_frechet_mean(w_torch, Y_train_torch, dist_name)
        pred = pred.squeeze(0)
        
        # Distance between prediction and true response
        true_y_tensor = Y[i].detach().clone().float()
        diff = pred - true_y_tensor
        dist_sq = torch.sum(diff * diff)
        
        loss = loss + dist_sq
    
    return loss / n


def fit_ifr(X_train, Y_train, dist_name, differentiable_frechet_mean, h, max_iter=100):
    """
    Fit IFR model using L-BFGS optimization.
    
    Args:
        X_train: [n, p] training predictors
        Y_train: list of [4, 4] SPD matrices
        dist_name: distance metric
        differentiable_frechet_mean: Fréchet mean function
        h: kernel bandwidth
        max_iter: maximum iterations for optimization
    
    Returns:
        theta: [p] fitted parameter vector
    """
    import scipy.optimize
    
    p = X_train.shape[1]
    
    # Initialize theta randomly
    theta_init = np.random.randn(p)
    theta_init = theta_init / np.linalg.norm(theta_init)  # normalize
    
    # Objective function for scipy
    def objective(theta):
        theta_tensor = torch.tensor(theta, dtype=torch.float32)
        loss = ifr_loss(theta_tensor, X_train, Y_train, dist_name, differentiable_frechet_mean, h)
        return loss.item()
    
    # Optimize
    bounds = [(-1, 1) for _ in range(p)]  # constrain to unit sphere
    result = scipy.optimize.minimize(
        objective,
        theta_init,
        method='L-BFGS-B',
        jac='3-point',  # Use finite differences for gradient
        bounds=bounds,
        options={'maxiter': max_iter}
    )
    
    theta_opt = result.x
    theta_opt = theta_opt / np.linalg.norm(theta_opt)  # ensure unit norm
    
    return torch.tensor(theta_opt, dtype=torch.float32)


def predict_ifr(X_train, Y_train, theta, X_test, dist_name, differentiable_frechet_mean, h):
    """
    Make predictions for IFR model.
    
    Args:
        X_train: [n, p] training predictors
        Y_train: list of [4, 4] SPD matrices
        theta: [p] fitted parameter vector
        X_test: [n_test, p] test predictors
        dist_name: distance metric
        differentiable_frechet_mean: Fréchet mean function
        h: kernel bandwidth
    
    Returns:
        list of [4, 4] predicted SPD matrices
    """
    n_test = len(X_test)
    preds = []
    
    # Convert inputs to tensors
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if isinstance(theta, np.ndarray):
        theta = torch.tensor(theta, dtype=torch.float32)
    
    # Project training data onto theta
    t_train = X_train @ theta
    
    for i in range(n_test):
        # Project test point onto theta direction
        x_proj = X_test[i] @ theta  # scalar
        
        # Local prediction at projected point
        pred = local_frechet_predict_1d(t_train, Y_train, x_proj, dist_name, differentiable_frechet_mean, h)
        preds.append(pred)
    
    return preds


def direct_sir_candidate_matrix(X, Y, dist_name, n_slices=5):
    """
    Compute SIR candidate matrix for Fréchet SDR.
    
    Args:
        X: [n, p] predictors
        Y: [n, 4, 4] responses
        dist_name: distance metric
        n_slices: number of slices for discretization
    
    Returns:
        M: [p, p] candidate matrix
    """
    n, p = X.shape
    
    # Discretize responses for SIR
    # For SPD matrices, we need a way to discretize
    # Use trace as a scalar summary for slicing
    traces = torch.diagonal(Y, dim1=-2, dim2=-1).sum(dim=-1)  # [n]
    traces_np = traces.numpy()
    
    # Create slices based on quantiles
    quantiles = np.linspace(0, 1, n_slices + 1)
    slice_bounds = np.quantile(traces_np, quantiles)
    
    # Assign each observation to a slice
    slice_indices = np.digitize(traces_np, slice_bounds[1:-1])  # [n]
    slice_indices = np.clip(slice_indices, 0, n_slices - 1)
    
    # Compute slice means
    slice_means = []
    for s in range(n_slices):
        mask = slice_indices == s
        if np.sum(mask) > 0:
            # Mean of X in this slice
            slice_mean = np.mean(X[mask], axis=0)  # [p]
            slice_means.append(slice_mean)
        else:
            slice_means.append(np.zeros(p))
    
    slice_means = np.array(slice_means)  # [n_slices, p]
    
    # Compute covariance between X and slice means
    X_centered = X - np.mean(X, axis=0)  # [n, p]
    
    # SIR candidate matrix: Cov(X, E[X|Y_slice])
    M = np.zeros((p, p))
    for i in range(n):
        s = slice_indices[i]
        diff = X_centered[i] - (slice_means[s] - np.mean(X, axis=0))
        M += np.outer(diff, diff)
    M /= n
    
    return M


def direct_frechet_sdr(X_train, Y_train, dist_name, differentiable_frechet_mean, h, n_slices=5):
    """
    Fit Fréchet SDR model.
    
    Args:
        X_train: [n, p] training predictors
        Y_train: [n, 4, 4] responses as tensor
        dist_name: distance metric
        differentiable_frechet_mean: Fréchet mean function
        h: kernel bandwidth
        n_slices: number of slices for SIR
    
    Returns:
        theta: [p] fitted parameter vector
    """
    # Compute SIR candidate matrix
    M = direct_sir_candidate_matrix(X_train, Y_train, dist_name, n_slices)
    
    # Find eigenvector corresponding to largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(M)
    theta = eigvecs[:, -1]  # largest eigenvalue
    
    # Normalize
    theta = theta / np.linalg.norm(theta)
    
    return torch.tensor(theta, dtype=torch.float32)


if __name__ == "__main__":
    main()