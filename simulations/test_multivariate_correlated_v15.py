#!/usr/bin/env python3
"""
Test case: Multivariate regression with correlated responses.

Setup:
  - X in R^p, p=20, X ~ N(0, Σ_X), Σ_X_{jk} = ρ_X^{|j-k|}, ρ_X=0.3
  - True structural dim d0=2, Z = B0^T X = (Z1, Z2)^T
  - Y = (Y1, ..., YV)^T in R^V, V=15
  - g(Z) = (Z1, Z2^2, sin(π Z1 Z2))^T for NL, or (Z1, Z2, 0)^T for L
  - μ_v(X) = a_v^T g(Z) + δ_v(Z)
  - δ_v(Z) = c_v sin(ω_v Z1) + b_v Z2
  - Y = μ(X) + ε, ε ~ N_V(0, Σ_Y), Σ_Y_{uv} = ρ_Y^{|u-v|}, ρ_Y=0.5

Methods compared:
  1. Global Mean           -- ignores X, constant prediction
  2. GFR                   -- Global linear regression
  3. DFR                   -- Deep regression
    4. FDRNN (d=p)           -- FDRNN without predictor reduction
    5. FDRNN * (full)        -- FDRNN with full predictor dimension, nuclear tuned

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
        V = 15  # number of responses
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
                [0.9, -0.3, 0.6],
                [0.6, 0.4, 0.8],
                [1.2, -0.2, 0.1],
                [0.8, 0.6, -0.3],
                [0.5, 0.9, 0.4],
                [1.0, 0.3, 0.7],
                [0.7, -0.1, 0.9],
                [0.9, 0.5, 0.2],
                [0.4, 0.8, 0.6],
                [1.1, 0.1, -0.2]
            ], dtype=torch.float32)  # [V, 3]
        self.A = A

        # δ_v(Z)
        if c_v is None:
            c_v = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.12, 0.18, 0.22, 0.14, 0.16, 0.19, 0.13, 0.21, 0.17, 0.23, 0.11])
        if b_v is None:
            b_v = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.03, 0.0, 0.04, 0.02, 0.0, 0.06, 0.01, 0.03, 0.0])
        if omega_v is None:
            omega_v = torch.tensor([1.0, 1.5, 2.0, 0.5, 1.2, 0.8, 1.3, 0.9, 1.7, 0.6, 1.4, 0.7, 1.1, 1.8, 0.4])
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
                    Y_ref_use = Y_ref[:, 0, :, :].unsqueeze(-1).to(device)
                else:
                    Y_ref_use = Y_ref.unsqueeze(-1).to(device)
                Y_hat = differentiable_frechet_mean(W, Y_ref_use, dist_name)
                mse = torch.mean((Y_hat - Y_b) ** 2, dim=[1,2,3])  # [B]
                mses.append(mse.cpu())
            else:
                # Multi-response
                V = W.shape[2]
                mse_batch = []
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v, :, :].unsqueeze(-1).to(device)
                    Y_b_v = Y_b[:, v, :, :]
                    Y_hat_v = differentiable_frechet_mean(w_v, Y_ref_v, dist_name)
                    mse_v = torch.mean((Y_hat_v - Y_b_v) ** 2, dim=[1,2,3])  # [B]
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
    fdrnn_tuned_params: dict = None,
    structural_dim: int = 2,
    n_responses: int = 15,
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
    results["baseline_avg_dist"] = base_avg
    results["baseline_avg_dist_sq"] = base_avg_sq
    results["baseline_time_sec"] = 0.0

    if verbose:
        print(f"\n  --- Competitors ---")
        print(f"  Global Mean avg Dist:   {base_avg:.6f}")

    # GFR: Global linear regression
    if verbose:
        print(f"\n  --- Global Linear Regression (GFR) ---")
    t_gfr = time.time()
    # Simple linear regression using torch
    X_train = ds_train.X.to(device)
    Y_train = ds_train.Y.view(ds_train.n, -1).to(device)  # [n, V]
    # Add bias
    X_train_bias = torch.cat([X_train, torch.ones(X_train.size(0), 1).to(X_train.device)], dim=1)
    # Solve (X^T X)^{-1} X^T Y
    XtX = X_train_bias.T @ X_train_bias
    XtY = X_train_bias.T @ Y_train
    W = torch.linalg.solve(XtX, XtY)  # [p+1, V]

    gfr_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            Y_batch_flat = Y_batch.view(Y_batch.size(0), -1).to(device)  # [B, V]
            X_batch_bias = torch.cat([X_batch, torch.ones(X_batch.size(0), 1).to(X_batch.device)], dim=1)
            Y_pred_gfr_flat = X_batch_bias @ W  # [B, V]
            Y_pred_gfr = Y_pred_gfr_flat.view(Y_batch.shape)  # [B, V, 1, 1]
            d = torch.mean((Y_pred_gfr - Y_batch) ** 2, dim=[1,2,3])  # [B]
            gfr_dists.append(d.cpu())

    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_avg_sq = (gfr_dists_cat ** 2).mean().item()
    gfr_time = time.time() - t_gfr

    results["gfr_avg_dist"] = gfr_avg
    results["gfr_avg_dist_sq"] = gfr_avg_sq
    results["gfr_time_sec"] = gfr_time

    if verbose:
        print(f"  GFR avg Dist:   {gfr_avg:.6f} (time: {gfr_time:.1f}s)")

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

    results["dfr_avg_dist"] = dfr_avg
    results["dfr_avg_dist_sq"] = dfr_avg_sq
    results["dfr_time_sec"] = dfr_time

    if verbose:
        print(f"  DFR avg Dist:   {dfr_avg:.6f} (time: {dfr_time:.1f}s)")

    # FDRNN (d=p) -- no predictor reduction
    if verbose:
        print(f"\n  --- FDRNN (d=p={p}) ---")
    set_seed(seed)
    _fdrnn_lr = fdrnn_tuned_params.get("lr", lr) if fdrnn_tuned_params else lr
    _fdrnn_er = fdrnn_tuned_params.get("entropy_reg", entropy_reg) if fdrnn_tuned_params else entropy_reg
    _fdrnn_wd = fdrnn_tuned_params.get("weight_decay", 1e-5) if fdrnn_tuned_params else 1e-5
    _fdrnn_nr = fdrnn_tuned_params.get("nuclear_reg", 0.01) if fdrnn_tuned_params else 0.01
    _fdrnn_rd = fdrnn_tuned_params.get("reduction_dim", p) if fdrnn_tuned_params else p
    _fdrnn_rt = fdrnn_tuned_params.get("reduction_type", "nonlinear") if fdrnn_tuned_params else "nonlinear"
    _fdrnn_rr = fdrnn_tuned_params.get("response_rank", min(5, n_responses)) if fdrnn_tuned_params else min(5, n_responses)
    _fdrnn_enc = fdrnn_tuned_params.get("encoder_sizes", [128, 64]) if fdrnn_tuned_params else [128, 64]
    _fdrnn_head = fdrnn_tuned_params.get("head_sizes", [64, 128]) if fdrnn_tuned_params else [64, 128]

    model_fdrnn = FrechetDRNN(
        input_dim=p,
        n_ref=n_train,
        reduction_dim=_fdrnn_rd,
        n_responses=n_responses,
        response_rank=_fdrnn_rr,
        response_alpha=1.0,
        reduction_type=_fdrnn_rt,
        encoder_sizes=_fdrnn_enc,
        head_sizes=_fdrnn_head,
        activation="relu",
        dropout=0.0,
    )
    if verbose:
        n_params_fdrnn = sum(pp.numel() for pp in model_fdrnn.parameters())
        print(f"  Parameters: {n_params_fdrnn:,}")
        print(f"  Reduction dim: {_fdrnn_rd}, nuclear_reg: {_fdrnn_nr}")
        print(f"  Max response rank: {_fdrnn_rr} (adaptive LoRA)")

    t_fdrnn = time.time()
    history_fdrnn = train_frechet_model(
        model=model_fdrnn,
        Y_ref=Y_ref,
        train_loader=train_loader,
        dist_name=dist_name,
        epochs=epochs,
        lr=_fdrnn_lr,
        weight_decay=_fdrnn_wd,
        entropy_reg=_fdrnn_er,
        nuclear_reg=_fdrnn_nr,
        device=device,
        verbose=verbose,
        val_loader=es_val_loader,
        patience=15,
    )
    fdrnn_time = time.time() - t_fdrnn

    fdrnn_avg, fdrnn_avg_sq = _evaluate_on_loader(
        model_fdrnn, test_loader, Y_ref, dist_name, device)
    fdrnn_train_d, fdrnn_train_d_sq = _evaluate_on_loader(
        model_fdrnn, train_loader, Y_ref, dist_name, device)
    fdrnn_resp_eff_rank, fdrnn_resp_sv = _effective_response_rank(model_fdrnn)

    if verbose:
        fdrnn_imp = (base_avg - fdrnn_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  FDRNN avg MSE:         {fdrnn_avg:.6f}")
        print(f"  FDRNN Improvement:   {fdrnn_imp:.1f}%")
        print(f"  FDRNN time:          {fdrnn_time:.1f}s")
        print(f"  FDRNN train avg MSE:   {fdrnn_train_d:.6f}  "
              f"(gap={fdrnn_avg - fdrnn_train_d:+.4f})")
        print(f"  FDRNN adaptive LoRA rank metric: {fdrnn_resp_eff_rank:.2f}")
        if fdrnn_resp_sv is not None:
            print(f"  FDRNN adaptive LoRA sv: {[f'{s:.4f}' for s in fdrnn_resp_sv]}")

    results["fdrnn_avg_dist"] = fdrnn_avg
    results["fdrnn_avg_dist_sq"] = fdrnn_avg_sq
    results["fdrnn_time_sec"] = fdrnn_time
    results["fdrnn_reduction_dim"] = _fdrnn_rd
    results["fdrnn_response_eff_rank"] = fdrnn_resp_eff_rank
    results["fdrnn_train_mse"] = fdrnn_train_d
    results["fdrnn_train_mse_sq"] = fdrnn_train_d_sq

    # Summary table
    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE  ({case}, d0={structural_dim})")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'avg Dist':>14}")
        print(f"  {'-'*46}")

        def _imp(val):
            return (base_avg - val) / base_avg * 100 if base_avg > 0 else 0

        print(f"  {'Global Mean':<30} {base_avg:>14.6f}")
        print(f"  {'GFR':<30} {gfr_avg:>14.6f}")
        print(f"  {'DFR':<30} {dfr_avg:>14.6f}")
        print(f"  {'FDRNN (d=':<13}{_fdrnn_rd}, adaptive){'':<15} {fdrnn_avg:>14.6f}")

        print(f"\n  --- Train / Test gap ---")
        print(f"  {'Method':<18} {'Train Dist':>10} {'Test Dist':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        for tag, tr, te in [
            ("FDRNN (d=p)", fdrnn_train_d, fdrnn_avg),
        ]:
            print(f"  {tag:<18} {tr:>10.4f} {te:>10.4f} {te - tr:>+10.4f}")

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
    parser.add_argument("--n_reps", type=int, default=15,
                        help="Number of repetitions")
    parser.add_argument("--structural_dim", type=int, default=2,
                        help="True structural dimension d0")
    parser.add_argument("--n_responses", type=int, default=15,
                        help="Number of responses")
    parser.add_argument(
        "--tune", action="store_true", default=True,
        help="Run grid search to select hyperparameters",
    )
    parser.add_argument(
        "--no-tune", dest="tune", action="store_false",
        help="Skip hyperparameter tuning",
    )
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
        [0.9, -0.3, 0.6],
        [0.6, 0.4, 0.8],
        [1.2, -0.2, 0.1],
        [0.8, 0.6, -0.3],
        [0.5, 0.9, 0.4],
        [1.0, 0.3, 0.7],
        [0.7, -0.1, 0.9],
        [0.9, 0.5, 0.2],
        [0.4, 0.8, 0.6],
        [1.1, 0.1, -0.2]
    ], dtype=torch.float32)  # [V, 3]
    
    # Shared c_v, b_v, omega_v
    c_v = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.12, 0.18, 0.22, 0.14, 0.16, 0.19, 0.13, 0.21, 0.17, 0.23, 0.11])
    b_v = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.03, 0.0, 0.04, 0.02, 0.0, 0.06, 0.01, 0.03, 0.0])
    omega_v = torch.tensor([1.0, 1.5, 2.0, 0.5, 1.2, 0.8, 1.3, 0.9, 1.7, 0.6, 1.4, 0.7, 1.1, 1.8, 0.4])
    
    shared_params = {
        'B0': B0,
        'A': A,
        'c_v': c_v,
        'b_v': b_v,
        'omega_v': omega_v
    }
    print(f"  Shared parameters generated with seed {args.seed}")

    # Hyperparameter tuning (default, can be skipped with --no-tune)
    fdrnn_tuned_params = None
    if args.tune:
        print(f"\n{'#'*70}")
        print(f"  TUNING: {args.case}")
        print(f"{'#'*70}")

        # Create a small dataset for tuning with shared params
        tune_ds = MultivariateCorrelatedDataset(
            n=100, case=args.case, seed=0, 
            B0=shared_params['B0'], A=shared_params['A'], 
            c_v=shared_params['c_v'], b_v=shared_params['b_v'], omega_v=shared_params['omega_v']
        )
        Y_ref_tune = tune_ds.Y

        # Adaptive LoRA: r = min(5, V)
        adaptive_r = min(5, V)
        print(f"  Adaptive r = {adaptive_r}")

        # Stage A: choose reduction dimension and reduction type
        print(f"\n  === FDRNN Stage A: choose (d, reduction_type) from 2×2×2=8 configs ===")
        stage_a_configs = [
            {"reduction_dim": d, "reduction_type": rt, "lr": lr}
            for d in [2, 5, 8]
            for rt in ["linear", "nonlinear"] 
            for lr in [5e-4, 1e-3]
        ]
        
        stage_a_val_losses = {}
        stage_a_best_params = {}
        
        for i, config in enumerate(stage_a_configs):
            d, rt, lr = config["reduction_dim"], config["reduction_type"], config["lr"]
            print(f"\n    --- Config {i+1}/8: d={d}, type={rt}, lr={lr:.1e} ---")
            
            fdrnn_stage_a_grid = {
                "lr": [lr],  # single value
                "entropy_reg": [0.0],  # fixed
                "nuclear_reg": [0.0],  # fixed
                "reduction_dim": [d],  # single value
                "reduction_type": [rt],  # single value
                "dropout": [0.0],  # fixed
            }
            
            result = grid_search_frechet(
                dataset=tune_ds,
                parent_Y=Y_ref_tune,
                model_class=FrechetDRNN,
                dist_name="euclidean",
                param_grid=fdrnn_stage_a_grid,
                fixed_model_kwargs={
                    "input_dim": p,
                    "n_responses": V,
                    "response_rank": 1,  # fixed for Stage A
                    "encoder_sizes": [128, 64],
                    "head_sizes": [64, 128],
                },
                fixed_train_kwargs={"epochs": 50},  # Reduced epochs for tuning
                val_frac=0.2,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
                verbose=False,  # less verbose for sub-searches
            )
            
            config_key = (d, rt, lr)
            stage_a_val_losses[config_key] = result["best_val_loss"]
            stage_a_best_params[config_key] = result["best_params"]
            print(f"    Config {i+1}: val loss = {stage_a_val_losses[config_key]:.4f}")

        # Choose best config from Stage A
        best_config = min(stage_a_val_losses, key=stage_a_val_losses.get)
        chosen_d, chosen_rt, chosen_lr = best_config
        print(f"\n    Chosen Stage A config: d={chosen_d}, type={chosen_rt}, lr={chosen_lr:.1e} (val loss {stage_a_val_losses[best_config]:.4f})")

        # Stage B: tune regularization and response coupling
        print(f"\n  === FDRNN Stage B: tune regularization for d={chosen_d}, type={chosen_rt}, adaptive r={adaptive_r} ===")
        fdrnn_stage_b_grid = {
            "lr": [5e-4, 1e-3],
            "entropy_reg": [0.0, 1e-3],
            "nuclear_reg": [0.0, 1e-4],
            "reduction_dim": [chosen_d],  # fixed from Stage A
            "reduction_type": [chosen_rt],  # fixed from Stage A
            "dropout": [0.0],  # fixed
        }
        fdrnn_result = grid_search_frechet(
            dataset=tune_ds,
            parent_Y=Y_ref_tune,
            model_class=FrechetDRNN,
            dist_name="euclidean",
            param_grid=fdrnn_stage_b_grid,
            fixed_model_kwargs={
                "input_dim": p,
                "n_responses": V,
                "response_rank": adaptive_r,  # adaptive LoRA
                "encoder_sizes": [128, 64],
                "head_sizes": [64, 128],
            },
            fixed_train_kwargs={"epochs": 50},  # Reduced epochs for tuning
            val_frac=0.2,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            verbose=True,
        )
        fdrnn_tuned_params = fdrnn_result["best_params"]
        fdrnn_tuned_params["reduction_dim"] = chosen_d  # ensure it's set
        fdrnn_tuned_params["response_rank"] = adaptive_r  # ensure it's set

        print(f"  Chosen FDRNN params: {fdrnn_tuned_params}")
    else:
        # Default params when not tuning
        adaptive_r = min(5, V)
        fdrnn_tuned_params = {
            "lr": args.lr,
            "entropy_reg": args.entropy_reg,
            "nuclear_reg": 0.01,
            "reduction_dim": p,  # default to full dim
            "reduction_type": "nonlinear",
            "response_rank": adaptive_r,
            "encoder_sizes": [128, 64],
            "head_sizes": [64, 128],
            "dropout": 0.0,
        }
        print(f"  Using default FDRNN params (no tuning): {fdrnn_tuned_params}")

    # Main experiment
    print(f"\n{'#'*70}")
    print(f"  CASE: {args.case}  |  {n_reps} repetitions")
    print(f"  d=p={p} (structural d0={structural_dim})")
    if fdrnn_tuned_params:
        print(f"  FDRNN params:  {fdrnn_tuned_params}")
    print(f"{'#'*70}")

    # Collect results
    methods = [
        ("Global Mean", "baseline"),
        ("GFR", "gfr"),
        ("DFR", "dfr"),
        (f"FDRNN (d={chosen_d}, adaptive LoRA)", "fdrnn"),
    ]
    collect_suffixes = ["_avg_dist", "_avg_dist_sq", "_time_sec"]
    all_vals = {}
    for _, prefix in methods:
        for suf in collect_suffixes:
            all_vals[prefix + suf] = []
    for prefix in ["fdrnn"]:
        for suf in ["_train_mse", "_train_mse_sq"]:
            all_vals[prefix + suf] = []
    all_vals["fdrnn_response_eff_rank"] = []

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
            fdrnn_tuned_params=fdrnn_tuned_params,
            structural_dim=structural_dim,
            n_responses=args.n_responses,
            shared_params=shared_params,
        )
        for _, prefix in methods:
            for suf in collect_suffixes:
                all_vals[prefix + suf].append(res[prefix + suf])
        for prefix in ["fdrnn"]:
            for suf in ["_train_mse", "_train_mse_sq"]:
                key = prefix + suf
                if key in res:
                    all_vals[key].append(res[key])
        all_vals["fdrnn_response_eff_rank"].append(res.get("fdrnn_response_eff_rank", float('nan')))

        if not verbose_rep:
            print(f"  FDRNN(Dist)={res['fdrnn_avg_dist']:.4f}")

    # Paper-ready tables
    W = 82
    eff_ranks = np.array(all_vals["fdrnn_response_eff_rank"])
    header_lines = [
        f"  RESULTS: {args.case} (V=15)  ({n_reps} reps)",
        f"  n_train={args.n_train}, n_test={args.n_test}",
        f"  d0={structural_dim}, FDRNN adaptive LoRA rank metric: "
        f"{eff_ranks.mean():.2f} (SE {eff_ranks.std(ddof=1)/np.sqrt(n_reps):.2f})"
        if n_reps > 1 else f"{eff_ranks.mean():.2f}",
    ]
    if fdrnn_tuned_params:
        chosen_r = fdrnn_tuned_params.get("response_rank", "N/A")
        header_lines.append(f"  FDRNN chosen r: {chosen_r}")
        header_lines.append(f"  FDRNN params:  {fdrnn_tuned_params}")

    print(f"\n\n{'=' * W}")
    for hl in header_lines:
        print(hl)
    print(f"{'=' * W}")

    base_prefix = methods[0][1]
    for panel_label, suf in [("Panel A: Dist(pred, Y)", "_avg_dist")]:
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
    panel_specs = [("Dist(pred, Y)", "_avg_dist")]
    short_labels = [label.split("(")[0].strip() for label, _ in methods]
    colors = ["#999999", "#4daf4a", "#e41a1c", "#377eb8"]

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
        f"Multivariate Correlated (V=15)  |  case={args.case}  |  "
        f"n={args.n_train}, d0={structural_dim}  |  {n_reps} reps",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_dir = os.path.join(project_root, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(
        fig_dir, f"multivariate_v15_{args.case}_errors.pdf",
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved -> {fig_path}")

    # Print averages with SE
    print("\nAverage results with SE:")
    for key in sorted(all_vals.keys()):
        values = all_vals[key]
        if values:
            mean = np.mean(values)
            se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            print(f"{key}: {mean:.6f} ± {se:.6f}")


if __name__ == "__main__":
    main()