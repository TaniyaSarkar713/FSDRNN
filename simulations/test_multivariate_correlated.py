#!/usr/bin/env python3
"""
Test case: Multivariate regression with correlated responses.

Setup:
  - X in R^p, p=20, X ~ N(0, Σ_X), Σ_X_{jk} = ρ_X^{|j-k|}, ρ_X=0.3
  - True structural dim d0=2, Z = B0^T X = (Z1, Z2)^T
  - Y = (Y1, ..., YV)^T in R^V, V=6
  - g(Z) = (Z1, Z2^2, sin(π Z1 Z2))^T for NL, or (Z1, Z2, 0)^T for L
  - μ_v(X) = a_v^T g(Z) + δ_v(Z)
  - δ_v(Z) = c_v sin(ω_v Z1) + b_v Z2
  - Y = μ(X) + ε, ε ~ N_V(0, Σ_Y), Σ_Y_{uv} = ρ_Y^{|u-v|}, ρ_Y=0.5

Methods compared:
  1. Global Mean           -- ignores X, constant prediction
  2. GFR                   -- Global linear regression
  3. DFR                   -- Deep regression

Usage:
  python simulations/test_multivariate_correlated.py              # default: NL
  python simulations/test_multivariate_correlated.py --case L     # linear case
  python simulations/test_multivariate_correlated.py --case NL    # nonlinear case
"""

import sys
import os
import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster/CI
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
    DISTANCE_FUNCTIONS,
    GlobalFrechetRegression,
    DeepFrechetRegression,
    grid_search_frechet,
    grid_search_dfr,
)


class MultivariateCorrelatedDataset(torch.utils.data.Dataset):
    """Dataset for multivariate correlated regression."""

    def __init__(self, n: int = None, case: str = "NL", seed: int = 42, n_responses: int = 6,
                 B0=None, A=None, c_v=None, b_v=None, omega_v=None, X=None, Y=None):
        """
        Generate data for multivariate correlated regression or create from existing data.

        Args:
            n: number of samples (if generating)
            case: "L" for linear, "NL" for nonlinear
            seed: random seed
            n_responses: number of responses V
            B0, A, c_v, b_v, omega_v: optional pre-generated parameters
            X, Y: optional existing data [n, p] and [n, V, 1, 1]
        """
        if X is not None and Y is not None:
            # Create from existing data
            self.X = X
            self.Y = Y
            self.n = X.shape[0]
            self.n_responses = Y.shape[1]
            return

        self.n = n
        self.case = case
        self.seed = seed
        self.n_responses = n_responses

        # Parameters
        p = 20  # input dimension
        d0 = 2  # true structural dimension
        V = n_responses
        rho_X = 0.3
        rho_Y = 0.5

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate Σ_X
        Sigma_X = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma_X[i, j] = rho_X ** abs(i - j)
        Sigma_X = torch.tensor(Sigma_X, dtype=torch.float32)

        # Generate X ~ N(0, Σ_X)
        L = torch.linalg.cholesky(Sigma_X)
        X_normal = torch.randn(n, p)
        self.X = (L @ X_normal.T).T  # [n, p]

        # Generate B0, orthonormal
        if B0 is None:
            B0 = torch.randn(p, d0)
            B0, _ = torch.linalg.qr(B0)  # [p, d0]
        self.B0 = B0

        # Z = B0^T X
        self.Z = self.X @ B0  # [n, d0]

        # g(Z)
        Z1 = self.Z[:, 0]
        Z2 = self.Z[:, 1]
        if case == "L":
            g1 = Z1
            g2 = Z2
            g3 = torch.zeros_like(Z1)
        else:  # NL
            g1 = Z1
            g2 = Z2 ** 2
            g3 = torch.sin(math.pi * Z1 * Z2)
        self.g = torch.stack([g1, g2, g3], dim=1)  # [n, 3]

        # A: V x 3
        if A is None:
            A = torch.tensor([
                [1.2, 0.8, 0.0],
                [1.0, 0.6, 0.2],
                [0.8, 0.0, 0.9],
                [0.7, 0.2, 1.0],
                [1.1, -0.5, 0.4],
                [0.9, -0.3, 0.6],
            ], dtype=torch.float32)[:V]  # [V, 3]
        self.A = A

        # δ_v(Z)
        if c_v is None:
            c_v = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.12, 0.18])[:V]
        if b_v is None:
            b_v = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.0, 0.1])[:V]
        if omega_v is None:
            omega_v = torch.tensor([1.0, 1.5, 2.0, 0.5, 1.2, 0.8])[:V]
        delta = torch.zeros(n, V)
        for v in range(V):
            delta[:, v] = c_v[v] * torch.sin(omega_v[v] * Z1) + b_v[v] * Z2

        # μ = g @ A^T + delta
        self.mu = self.g @ A.T + delta  # [n, V]

        # Σ_Y
        Sigma_Y = torch.zeros(V, V)
        for u in range(V):
            for v in range(V):
                Sigma_Y[u, v] = rho_Y ** abs(u - v)
        L_Y = torch.linalg.cholesky(Sigma_Y)

        # ε ~ N(0, Σ_Y)
        eps_normal = torch.randn(n, V)
        eps = (L_Y @ eps_normal.T).T

        # Y = μ + ε
        self.Y = self.mu + eps  # [n, V]

        # Reshape Y to [n, V, 1, 1] for compatibility with manifold code
        self.Y = self.Y.view(self.n, V, 1, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_true_mean(self, idx):
        """Get true mean for sample idx."""
        return self.mu[idx].view(self.n_responses, 1, 1)


def _grid_search_fdrnn(
    case, n_train, n_test, epochs, batch_size, device, structural_dim, n_responses, 
    shared_params, param_grid, tune_seed, verbose=True
):
    """Grid search for FDRNN hyperparameters in multivariate correlated setting."""
    from itertools import product
    
    # Generate tuning data
    set_seed(tune_seed)
    tune_n_train = min(200, n_train)  # smaller dataset for tuning
    tune_n_test = min(500, n_test)
    
    # Create dataset for tuning
    tune_ds = MultivariateCorrelatedDataset(
        n=tune_n_train, case=case, seed=tune_seed,
        B0=shared_params['B0'] if shared_params else None,
        A=shared_params['A'] if shared_params else None,
        c_v=shared_params['c_v'] if shared_params else None,
        b_v=shared_params['b_v'] if shared_params else None,
        omega_v=shared_params['omega_v'] if shared_params else None,
    )
    
    X_tune = tune_ds.X
    Y_tune = tune_ds.Y
    Y_ref_tune = Y_tune  # Use the same data as reference
    
    # Create dataset and split
    tune_dataset = MultivariateCorrelatedDataset(X=X_tune, Y=Y_tune)
    train_size = int(0.8 * len(tune_dataset))
    val_size = len(tune_dataset) - train_size
    train_ds, val_ds = random_split(tune_dataset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(tune_seed))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Get distance function
    dist_name = "euclidean"
    distance_fn = get_distance_fn(dist_name)
    
    # Grid search
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    all_configs = list(product(*param_values))
    
    best_loss = float('inf')
    best_params = None
    
    if verbose:
        print(f"  Grid search: {len(all_configs)} configurations")
        print(f"    train/val sizes: {len(train_ds)} / {len(val_ds)}")
    
    for i, config in enumerate(all_configs):
        config_dict = dict(zip(param_names, config))
        
        if verbose:
            config_str = ", ".join([f"{k}={v}" for k, v in config_dict.items()])
            print(f"    [{i+1}/{len(all_configs)}] {config_str}", end="  →  ")
        
        try:
            # Initialize model
            model = FrechetDRNN(
                input_dim=X_tune.shape[1],
                n_ref=len(tune_dataset),
                n_responses=n_responses,
                response_rank=config_dict.get("response_rank", 8),
                encoder_sizes=[128, 64],
                head_sizes=[64, 128],
                reduction_dim=config_dict["reduction_dim"],
                reduction_type=config_dict["reduction_type"],
                dropout=config_dict.get("dropout", 0.0),
            ).to(device)
            
            # Train
            history = train_frechet_model(
                model=model,
                Y_ref=Y_ref_tune,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=min(epochs, 50),  # fewer epochs for tuning
                lr=config_dict["lr"],
                entropy_reg=config_dict.get("entropy_reg", 0.0),
                nuclear_reg=config_dict.get("nuclear_reg", 0.0),
                device=device,
                verbose=False,
            )
            
            # Evaluate on validation set
            val_results = evaluate_frechet_model(
                model=model,
                Y_ref=Y_ref_tune,
                test_loader=val_loader,
                dist_name=dist_name,
                device=device,
            )
            val_loss = val_results["avg_dist_sq"]
            
            if verbose:
                print(f"val d²={val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = config_dict.copy()
                
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue
    
    if verbose:
        print(f"\n  Best params:    {best_params}")
        print(f"  Best val d²:    {best_loss:.6f}")
    
    return best_params

    def __init__(self, n: int, case: str = "NL", seed: int = 42, B0=None, A=None, c_v=None, b_v=None, omega_v=None):
        """
        Generate data for multivariate correlated regression.

        Args:
            n: number of samples
            case: "L" for linear, "NL" for nonlinear
            seed: random seed
            B0: optional pre-generated B0 [p, d0]
            A: optional pre-generated A [V, 3]
            c_v, b_v, omega_v: optional pre-generated vectors for delta
        """
        self.n = n
        self.case = case
        self.seed = seed

        # Parameters
        p = 20  # input dimension
        d0 = 2  # true structural dimension
        V = 6   # number of responses
        rho_X = 0.3
        rho_Y = 0.5

        self.p = p
        self.d0 = d0
        self.V = V

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate Σ_X
        Sigma_X = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma_X[i, j] = rho_X ** abs(i - j)
        Sigma_X = torch.tensor(Sigma_X, dtype=torch.float32)

        # Generate X ~ N(0, Σ_X)
        L = torch.linalg.cholesky(Sigma_X)
        X_normal = torch.randn(n, p)
        self.X = (L @ X_normal.T).T  # [n, p]

        # Generate B0, orthonormal
        if B0 is None:
            B0 = torch.randn(p, d0)
            B0, _ = torch.linalg.qr(B0)  # [p, d0]
        self.B0 = B0

        # Z = B0^T X
        self.Z = self.X @ B0  # [n, d0]

        # g(Z)
        Z1 = self.Z[:, 0]
        Z2 = self.Z[:, 1]
        if case == "L":
            g1 = Z1
            g2 = Z2
            g3 = torch.zeros_like(Z1)
        else:  # NL
            g1 = Z1
            g2 = Z2 ** 2
            g3 = torch.sin(math.pi * Z1 * Z2)
        self.g = torch.stack([g1, g2, g3], dim=1)  # [n, 3]

        # A: V x 3
        if A is None:
            A = torch.tensor([
                [1.2, 0.8, 0.0],
                [1.0, 0.6, 0.2],
                [0.8, 0.0, 0.9],
                [0.7, 0.2, 1.0],
                [1.1, -0.5, 0.4],
                [0.9, -0.3, 0.6]
            ], dtype=torch.float32)  # [V, 3]
        self.A = A

        # δ_v(Z)
        if c_v is None:
            c_v = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.12, 0.18])
        if b_v is None:
            b_v = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
        if omega_v is None:
            omega_v = torch.tensor([1.0, 1.5, 2.0, 0.5, 1.2, 0.8])
        delta = torch.zeros(n, V)
        for v in range(V):
            delta[:, v] = c_v[v] * torch.sin(omega_v[v] * Z1) + b_v[v] * Z2

        # μ = g @ A^T + delta
        self.mu = self.g @ A.T + delta  # [n, V]

        # Σ_Y
        Sigma_Y = torch.zeros(V, V)
        for u in range(V):
            for v in range(V):
                Sigma_Y[u, v] = rho_Y ** abs(u - v)
        L_Y = torch.linalg.cholesky(Sigma_Y)

        # ε ~ N(0, Σ_Y)
        eps_normal = torch.randn(n, V)
        eps = (L_Y @ eps_normal.T).T

        # Y = μ + ε
        self.Y = self.mu + eps  # [n, V]

        # Reshape Y to [n, V, 1, 1] for compatibility with manifold code
        self.Y = self.Y.view(self.n, V, 1, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_true_mean(self, idx):
        """Get true mean for sample idx."""
        return self.mu[idx].view(self.V, 1, 1)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_on_loader(model, loader, Y_ref, dist_name, device):
    """Compute avg MSE for *model* on *loader*."""
    model.eval()
    mses = []
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            W = model(X_b)
            W = torch.softmax(W, dim=1)  # normalize logits to weights
            if model.n_responses == 1:
                # Single-response
                if Y_ref.dim() == 4:
                    Y_ref_use = Y_ref[:, 0, :, :].to(device)
                else:
                    Y_ref_use = Y_ref.to(device)
                Y_hat = differentiable_frechet_mean(W, Y_ref_use, dist_name)
                mse = torch.mean((Y_hat - Y_b) ** 2, dim=[1,2])  # [B]
                mses.append(mse.cpu())
            else:
                # Multi-response
                V = W.shape[2]
                mse_batch = []
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v, :, :]
                    Y_b_v = Y_b[:, v, :, :]
                    Y_hat_v = differentiable_frechet_mean(w_v, Y_ref_v.to(device), dist_name)
                    mse_v = torch.mean((Y_hat_v - Y_b_v) ** 2, dim=[1,2])  # [B]
                    mse_batch.append(mse_v)
                mse_avg = torch.stack(mse_batch).mean(dim=0)  # [B]
                mses.append(mse_avg.cpu())
    cat = torch.cat(mses)
    return cat.mean().item(), (cat ** 2).mean().item()


def _effective_rank(model_fdrnn):
    """Compute rank metric of the first Linear layer in the reduction net."""
    with torch.no_grad():
        for mod in model_fdrnn.reduction.modules():
            if isinstance(mod, torch.nn.Linear):
                sv = torch.linalg.svdvals(mod.weight).cpu().numpy()
                break
        else:
            raise RuntimeError("No Linear layer found in reduction")
    eff = float((sv.sum() ** 2) / (sv ** 2).sum()) if (sv ** 2).sum() > 0 else 0.0
    return eff, sv


def _effective_response_rank(model):
    """Compute effective rank of the LoRA response matrix M = B A^T with adaptive pruning."""
    with torch.no_grad():
        alpha = model.response_refine.alpha
        rank = model.response_refine.rank
        B = model.response_refine.B
        A = model.response_refine.A
        delta = B @ A.T
        
        # SVD of delta for pruning
        U, s, Vh = torch.linalg.svd(delta, full_matrices=False)
        
        # Adaptive pruning: threshold based on max singular value
        threshold = 0.01 * s.max()  # prune if < 1% of max
        s_pruned = s * (s >= threshold)
        
        # Reconstruct pruned delta
        delta_pruned = U @ torch.diag(s_pruned) @ Vh
        
        # Pruned M
        I = torch.eye(model.n_responses, device=B.device, dtype=B.dtype)
        M_pruned = I + (alpha / rank) * delta_pruned
        
        sv_pruned = torch.linalg.svdvals(M_pruned).cpu().numpy()
    
    eff = float((sv_pruned.sum() ** 2) / (sv_pruned ** 2).sum()) if (sv_pruned ** 2).sum() > 0 else 0.0
    return eff, sv_pruned


def run_single_case(
    case: str,
    n_train: int = 400,
    n_test: int = 1000,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    entropy_reg: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
    structural_dim: int = 2,
    n_responses: int = 6,
    fdrnn_tuned_params: dict = None,
    shared_params: dict = None,
):
    """
    Run the full train + evaluate pipeline for one case.

    Methods: Global Mean, GFR, DFR, FDRNN (with adaptive LoRA).
    """
    p = 20  # input dim

    set_seed(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Case: {case}")
        print(f"  n_train={n_train}, n_test={n_test}, seed={seed}")
        print(f"  epochs={epochs}, lr={lr}, d0={structural_dim}")
        if fdrnn_tuned_params:
            print(f"  [TUNED FDRNN params: {fdrnn_tuned_params}]")
        print(f"{'='*70}")

    ds_train = MultivariateCorrelatedDataset(
        n=n_train, case=case, seed=seed,
        B0=shared_params['B0'] if shared_params else None,
        A=shared_params['A'] if shared_params else None,
        c_v=shared_params['c_v'] if shared_params else None,
        b_v=shared_params['b_v'] if shared_params else None,
        omega_v=shared_params['omega_v'] if shared_params else None,
    )
    ds_test = MultivariateCorrelatedDataset(
        n=n_test, case=case, seed=seed + 1000,
        B0=shared_params['B0'] if shared_params else None,
        A=shared_params['A'] if shared_params else None,
        c_v=shared_params['c_v'] if shared_params else None,
        b_v=shared_params['b_v'] if shared_params else None,
        omega_v=shared_params['omega_v'] if shared_params else None,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    if verbose:
        X_sample, Y_sample = ds_train[0]
        print(f"\n  Input shape:    X ~ {tuple(X_sample.shape)}")
        print(f"  Response shape: Y ~ {tuple(Y_sample.shape)}")
        true_mean = ds_train.get_true_mean(0)
        print(f"  True mean E[Y|X]: {true_mean.tolist()}")

    # Reference set
    Y_ref = ds_train.Y  # [n_train, V, 1, 1]

    # Early-stopping validation split (20%)
    val_n = max(1, int(0.2 * len(ds_train)))
    es_train_ds, es_val_ds = random_split(
        ds_train, [len(ds_train) - val_n, val_n],
        generator=torch.Generator().manual_seed(seed),
    )
    es_val_loader = DataLoader(es_val_ds, batch_size=batch_size, shuffle=False)

    # For this setup, use euclidean distance for vectors
    dist_name = "euclidean"

    # Global mean baseline
    global_mean = ds_train.Y.mean(dim=0)  # [V, 1, 1]

    baseline_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_batch = Y_batch.to(device)
            B = X_batch.size(0)
            gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1).to(device)  # [B, V, 1, 1]
            # MSE per sample
            d = torch.mean((gm - Y_batch) ** 2, dim=[1,2,3])  # [B]
            baseline_dists.append(d.cpu())

    baseline_dists_cat = torch.cat(baseline_dists)
    base_avg = baseline_dists_cat.mean().item()
    base_avg_sq = (baseline_dists_cat ** 2).mean().item()

    results = {}
    results["case"] = case
    results["baseline_avg_mse"] = base_avg
    results["baseline_avg_mse_sq"] = base_avg_sq
    results["baseline_time_sec"] = 0.0

    if verbose:
        print(f"\n  --- Competitors ---")
        print(f"  Global Mean avg MSE:   {base_avg:.6f}")

    # GFR: Global linear regression
    if verbose:
        print(f"\n  --- Global Linear Regression (GFR) ---")
    t_gfr = time.time()
    # Simple linear regression using torch
    X_train = ds_train.X.to(device)
    Y_train = ds_train.Y.view(ds_train.n, -1).to(device)  # [n, V]
    # Add bias
    X_train_bias = torch.cat([X_train, torch.ones(X_train.size(0), 1, device=device)], dim=1)
    # Solve (X^T X)^{-1} X^T Y
    XtX = X_train_bias.T @ X_train_bias
    XtY = X_train_bias.T @ Y_train
    W = torch.inverse(XtX) @ XtY  # [p+1, V]

    gfr_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            Y_batch_flat = Y_batch.view(Y_batch.size(0), -1)  # [B, V]
            X_batch_bias = torch.cat([X_batch, torch.ones(X_batch.size(0), 1, device=device)], dim=1)
            Y_pred_gfr_flat = X_batch_bias @ W  # [B, V]
            Y_pred_gfr = Y_pred_gfr_flat.view(Y_batch.shape)  # [B, V, 1, 1]
            d = torch.mean((Y_pred_gfr - Y_batch) ** 2, dim=[1,2,3])  # [B]
            gfr_dists.append(d.cpu())

    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_avg_sq = (gfr_dists_cat ** 2).mean().item()
    gfr_time = time.time() - t_gfr

    results["gfr_avg_mse"] = gfr_avg
    results["gfr_avg_mse_sq"] = gfr_avg_sq
    results["gfr_time_sec"] = gfr_time

    if verbose:
        print(f"  GFR avg MSE:   {gfr_avg:.6f} (time: {gfr_time:.1f}s)")

    # DFR: Deep regression (placeholder, using simple NN)
    if verbose:
        print(f"\n  --- Deep Regression (DFR) ---")
    t_dfr = time.time()
    # Simple MLP for regression
    dfr_model = torch.nn.Sequential(
        torch.nn.Linear(p, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, n_responses)
    ).to(device)
    optimizer = torch.optim.Adam(dfr_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        dfr_model.train()
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            Y_pred = dfr_model(X_b)  # [B, V]
            loss = criterion(Y_pred, Y_b.view(Y_b.size(0), -1))
            loss.backward()
            optimizer.step()

    dfr_dists = []
    dfr_model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred_dfr = dfr_model(X_batch)  # [B, V]
            Y_pred_dfr = Y_pred_dfr.view(Y_batch.shape)  # [B, V, 1, 1]
            d = torch.mean((Y_pred_dfr - Y_batch) ** 2, dim=[1,2,3])  # [B]
            dfr_dists.append(d.cpu())

    dfr_dists_cat = torch.cat(dfr_dists)
    dfr_avg = dfr_dists_cat.mean().item()
    dfr_avg_sq = (dfr_dists_cat ** 2).mean().item()
    dfr_time = time.time() - t_dfr

    results["dfr_avg_mse"] = dfr_avg
    results["dfr_avg_mse_sq"] = dfr_avg_sq
    results["dfr_time_sec"] = dfr_time

    if verbose:
        print(f"  DFR avg MSE:   {dfr_avg:.6f} (time: {dfr_time:.1f}s)")

    # FDRNN: Frechet Deep RNN with adaptive LoRA
    if fdrnn_tuned_params is not None:
        if verbose:
            print(f"\n  --- FDRNN (d={fdrnn_tuned_params['reduction_dim']}, adaptive LoRA) ---")
        t_fdrnn = time.time()
        
        # Initialize FDRNN model with tuned parameters
        fdrnn_model = FrechetDRNN(
            input_dim=p,
            n_ref=len(ds_train),
            n_responses=n_responses,
            response_rank=fdrnn_tuned_params["response_rank"],
            encoder_sizes=[128, 64],
            head_sizes=[64, 128],
            reduction_dim=fdrnn_tuned_params["reduction_dim"],
            reduction_type=fdrnn_tuned_params["reduction_type"],
            dropout=fdrnn_tuned_params["dropout"],
        ).to(device)
        
        if verbose:
            n_params = sum(p.numel() for p in fdrnn_model.parameters())
            print(f"  Parameters: {n_params:,}")
            print(f"  Reduction dim: {fdrnn_tuned_params['reduction_dim']}, nuclear_reg: {fdrnn_tuned_params['nuclear_reg']}")
            print(f"  Max response rank: {fdrnn_tuned_params['response_rank']} (adaptive LoRA)")
        
        # Train FDRNN
        history = train_frechet_model(
            model=fdrnn_model,
            Y_ref=Y_ref,
            train_loader=train_loader,
            val_loader=es_val_loader,
            epochs=epochs,
            lr=fdrnn_tuned_params["lr"],
            entropy_reg=fdrnn_tuned_params["entropy_reg"],
            nuclear_reg=fdrnn_tuned_params["nuclear_reg"],
            device=device,
            verbose=False,  # less verbose for individual runs
        )
        
        # Evaluate FDRNN
        fdrnn_results = evaluate_frechet_model(
            model=fdrnn_model,
            Y_ref=Y_ref,
            test_loader=test_loader,
            dist_name=dist_name,
            device=device,
        )
        fdrnn_avg = fdrnn_results["avg_dist"]
        fdrnn_avg_sq = fdrnn_results["avg_dist_sq"]
        
        # Evaluate on training set for gap computation
        fdrnn_train_results = evaluate_frechet_model(
            model=fdrnn_model,
            Y_ref=Y_ref,
            test_loader=train_loader,
            dist_name=dist_name,
            device=device,
        )
        fdrnn_train_avg = fdrnn_train_results["avg_dist"]
        fdrnn_train_avg_sq = fdrnn_train_results["avg_dist_sq"]
        
        fdrnn_time = time.time() - t_fdrnn
        
        # Compute effective ranks
        fdrnn_eff_rank, _ = _effective_rank(fdrnn_model)
        fdrnn_response_eff_rank, _ = _effective_response_rank(fdrnn_model)
        
        results["fdrnn_avg_mse"] = fdrnn_avg
        results["fdrnn_avg_mse_sq"] = fdrnn_avg_sq
        results["fdrnn_train_mse"] = fdrnn_train_avg
        results["fdrnn_train_mse_sq"] = fdrnn_train_avg_sq
        results["fdrnn_time_sec"] = fdrnn_time
        results["fdrnn_response_eff_rank"] = fdrnn_response_eff_rank
        
        if verbose:
            print(f"  FDRNN avg MSE:         {fdrnn_avg:.6f}")
            print(f"  FDRNN Improvement:   {(base_avg - fdrnn_avg) / base_avg * 100:.1f}%")
            print(f"  FDRNN time:          {fdrnn_time:.1f}s")
            print(f"  FDRNN train avg MSE:   {fdrnn_train_avg:.6f}  (gap={fdrnn_train_avg - fdrnn_avg:+.4f})")
            print(f"  FDRNN adaptive LoRA rank metric: {fdrnn_response_eff_rank:.2f}")

    # Summary table
    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE  ({case}, d0={structural_dim})")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'avg MSE':>14}")
        print(f"  {'-'*46}")

        def _imp(val):
            return (base_avg - val) / base_avg * 100 if base_avg > 0 else 0

        print(f"  {'Global Mean':<30} {base_avg:>14.6f}")
        print(f"  {'GFR':<30} {gfr_avg:>14.6f}")
        print(f"  {'DFR':<30} {dfr_avg:>14.6f}")
        if fdrnn_tuned_params is not None:
            print(f"  {f'FDRNN (d={fdrnn_tuned_params["reduction_dim"]}, adaptive)':<30} {fdrnn_avg:>14.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test: Multivariate correlated regression"
    )
    parser.add_argument(
        "--case",
        type=str,
        default="NL",
        choices=["L", "NL"],
        help="Case: L for linear, NL for nonlinear",
    )
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--entropy_reg", type=float, default=0.01,
                        help="entropy regularisation coefficient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=5,
                        help="Number of repetitions")
    parser.add_argument("--structural_dim", type=int, default=2,
                        help="True structural dimension d0")
    parser.add_argument("--n_responses", type=int, default=6,
                        help="Number of responses")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    n_reps = args.n_reps
    structural_dim = args.structural_dim
    p = 20
    V = args.n_responses

    # Generate shared parameters for fair train/test comparison
    print(f"\n  Generating shared structural parameters...")
    set_seed(args.seed)
    
    # Shared B0
    B0 = torch.randn(p, structural_dim)
    B0, _ = torch.linalg.qr(B0)  # [p, d0]
    
    # Shared A [V, 3]
    A = torch.tensor([
        [1.2, 0.8, 0.0],
        [1.0, 0.6, 0.2],
        [0.8, 0.0, 0.9],
        [0.7, 0.2, 1.0],
        [1.1, -0.5, 0.4],
        [0.9, -0.3, 0.6]
    ], dtype=torch.float32)  # [V, 3]
    
    # Shared c_v, b_v, omega_v
    c_v = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.12, 0.18])
    b_v = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
    omega_v = torch.tensor([1.0, 1.5, 2.0, 0.5, 1.2, 0.8])
    
    shared_params = {
        'B0': B0,
        'A': A,
        'c_v': c_v,
        'b_v': b_v,
        'omega_v': omega_v
    }
    print(f"  Shared parameters generated with seed {args.seed}")

    # FDRNN Hyperparameter Tuning (only for multiple reps to avoid overhead)
    fdrnn_tuned_params = None
    if n_reps > 1:
        print(f"\n  Tuning FDRNN hyperparameters...")
        
        # Stage A: Architecture selection
        print(f"  Stage A: Architecture selection (d, reduction_type, lr)")
        stage_a_params = {
            "reduction_dim": [2, 5, 8],
            "reduction_type": ["linear", "nonlinear"],
            "lr": [5e-4, 1e-3],
            "response_rank": [min(args.n_responses, 5)],  # from ADALORA
            "dropout": [0.0],
            "entropy_reg": [0.0],  # will be tuned in Stage B
            "nuclear_reg": [0.0],  # will be tuned in Stage B
        }
        
        fdrnn_tuned_params = _grid_search_fdrnn(
            case=args.case,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            structural_dim=structural_dim,
            n_responses=args.n_responses,
            shared_params=shared_params,
            param_grid=stage_a_params,
            tune_seed=args.seed + 1000,  # different seed for tuning
            verbose=True,
        )
        
        # Stage B: Regularization tuning
        print(f"\n  Stage B: Regularization tuning (lr, entropy_reg, nuclear_reg)")
        stage_b_params = {
            "reduction_dim": [fdrnn_tuned_params["reduction_dim"]],
            "reduction_type": [fdrnn_tuned_params["reduction_type"]],
            "lr": [5e-4, 1e-3],
            "response_rank": [min(args.n_responses, 5)],  # from ADALORA
            "dropout": [0.0],
            "entropy_reg": [0.0, 1e-3],
            "nuclear_reg": [0.0, 1e-4],
        }
        
        fdrnn_tuned_params = _grid_search_fdrnn(
            case=args.case,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            structural_dim=structural_dim,
            n_responses=args.n_responses,
            shared_params=shared_params,
            param_grid=stage_b_params,
            tune_seed=args.seed + 2000,  # different seed for tuning
            verbose=True,
        )
        
        print(f"\n  FDRNN tuned parameters:")
        print(f"    reduction_dim: {fdrnn_tuned_params['reduction_dim']}")
        print(f"    reduction_type: {fdrnn_tuned_params['reduction_type']}")
        print(f"    lr: {fdrnn_tuned_params['lr']}")
        print(f"    entropy_reg: {fdrnn_tuned_params['entropy_reg']}")
        print(f"    nuclear_reg: {fdrnn_tuned_params['nuclear_reg']}")
        print(f"    response_rank: {fdrnn_tuned_params['response_rank']}")
        print(f"    dropout: {fdrnn_tuned_params['dropout']}")

    # Main experiment
    print(f"\n{'#'*70}")
    print(f"  CASE: {args.case}  |  {n_reps} repetitions")
    print(f"  d=p={p} (structural d0={structural_dim})")
    print(f"{'#'*70}")

    # Collect results
    methods = [
        ("Global Mean", "baseline"),
        ("GFR", "gfr"),
        ("DFR", "dfr"),
    ]
    if fdrnn_tuned_params is not None:
        methods.append(("FDRNN", "fdrnn"))
    
    collect_suffixes = ["_avg_mse", "_avg_mse_sq", "_time_sec"]
    all_vals = {}
    for _, prefix in methods:
        for suf in collect_suffixes:
            all_vals[prefix + suf] = []

    for rep in range(n_reps):
        rep_seed = args.seed + rep
        verbose_rep = (rep == 0)
        if not verbose_rep:
            print(f"  rep {rep + 1}/{n_reps} (seed={rep_seed}) ...",
                  end="", flush=True)

        res = run_single_case(
            case=args.case,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            entropy_reg=args.entropy_reg,
            seed=rep_seed,
            device=device,
            verbose=verbose_rep,
            structural_dim=structural_dim,
            n_responses=args.n_responses,
            shared_params=shared_params,
            fdrnn_tuned_params=fdrnn_tuned_params,
        )
        for _, prefix in methods:
            for suf in collect_suffixes:
                all_vals[prefix + suf].append(res[prefix + suf])

        if not verbose_rep:
            print(f"  FDRNN(MSE)={res['fdrnn_avg_mse']:.4f}")

    # Paper-ready tables
    W = 82
    header_lines = [
        f"  RESULTS: {args.case}  ({n_reps} reps)",
        f"  n_train={args.n_train}, n_test={args.n_test}",
        f"  d0={structural_dim}",
    ]

    print(f"\n\n{'=' * W}")
    for hl in header_lines:
        print(hl)
    print(f"{'=' * W}")

    base_prefix = methods[0][1]
    for panel_label, suf in [("Panel A: MSE(pred, Y)", "_avg_mse")]:
        base_arr = np.array(all_vals[base_prefix + suf])
        print(f"\n  {panel_label}")
        print(f"  {'Method':<34} {'Mean':>10} {'(SE)':>10} "
              f"{'Improv%':>8} {'(SE)':>8} {'Time(s)':>8}")
        print(f"  {'-' * (W - 4)}")

        for label, prefix in methods:
            arr = np.array(all_vals[prefix + suf])
            t_arr = np.array(all_vals[prefix + "_time_sec"])
            mean = arr.mean()
            se = arr.std(ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
            t_mean = t_arr.mean()

            if prefix == base_prefix:
                print(f"  {label:<34} {mean:>10.4f} ({se:>7.4f}) "
                      f"{'---':>8} {'':>8} {t_mean:>7.1f}s")
            else:
                imp_arr = (base_arr - arr) / base_arr * 100
                imp_mean = imp_arr.mean()
                imp_se = (imp_arr.std(ddof=1) / np.sqrt(n_reps)
                          if n_reps > 1 else 0.0)
                print(f"  {label:<34} {mean:>10.4f} ({se:>7.4f}) "
                      f"{imp_mean:>7.1f}% ({imp_se:>5.1f}%) "
                      f"{t_mean:>7.1f}s")

    print(f"{'=' * W}\n")

    # Figure
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    panel_specs = [("MSE(pred, Y)", "_avg_mse")]
    short_labels = [label.split("(")[0].strip() for label, _ in methods]
    colors = ["#999999", "#4daf4a", "#e41a1c"]

    for ax, (ylabel, suf) in zip([axes], panel_specs):
        data = [np.array(all_vals[prefix + suf]) for _, prefix in methods]
        bplot = ax.boxplot(
            data, patch_artist=True, widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bplot["boxes"], colors[:len(methods)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Multivariate Correlated  |  case={args.case}  |  "
        f"n={args.n_train}, d0={structural_dim}  |  {n_reps} reps",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_dir = os.path.join(project_root, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(
        fig_dir, f"multivariate_{args.case}_errors.pdf",
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved -> {fig_path}")


if __name__ == "__main__":
    main()
