#!/usr/bin/env python3
"""
Test case: Multi-response Fréchet mean regression on SPD matrices.

Setup:
  - X in R^12 with components:
      X1 ~ U(0,1),  X2 ~ U(-1/2, 1/2),  X3 ~ U(1,2),
      X4 ~ Gamma(3,2),  X5 ~ Gamma(4,2),  X6 ~ Gamma(5,2),
      X7 ~ N(0,1),  X8 ~ N(0,1),  X9 ~ N(0,1),
      X10 ~ Ber(0.4),  X11 ~ Ber(0.5),  X12 ~ Ber(0.6)
  - Responses: Y_v ~ Wishart(Sigma(X), df) for v=1,...,V, each a 5x5 SPD matrix
  - Sigma(X) = diag(s11,...,s55) with input-dependent diagonal entries
  - The NN learns response-specific weighted Fréchet means:
      m_{w,v} = argmin_{y} sum_i w_{i,v} d^2(y, Y_{i,v})

Methods compared:
  1. Global Mean           -- ignores X, constant prediction
  2. GFR                   -- Global Fréchet Regression (linear)
  3. DFR                   -- Deep Fréchet Regression (Petersen & Mueller)
  4. FSDRNN                -- Frechet SDR Neural Network with adaptive LoRA
  5. IFR                   -- Single-index Fréchet Regression
  6. Fréchet SDR           -- Fréchet Sufficient Dimension Reduction

Usage:
  python simulations/test_spd_frechet.py              # default: power
  python simulations/test_spd_frechet.py --metric power
  python simulations/test_spd_frechet.py --metric frobenius
  python simulations/test_spd_frechet.py --metric affine_invariant
  python simulations/test_spd_frechet.py --metric log_cholesky
  python simulations/test_spd_frechet.py --metric bures_wasserstein
  python simulations/test_spd_frechet.py --metric all  # run all five metrics
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
import scipy.optimize

from src.spd_frechet_adaptive import (
    WishartSPDDataset,
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


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_on_loader(model, loader, Y_ref, dist_name, device):
    """Compute avg d and avg d**2 for *model* on *loader*."""
    dist_fn = get_distance_fn(dist_name)
    model.eval()
    dists = []
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
                dists.append(dist_fn(Y_hat, Y_b).cpu())
            else:
                # Multi-response
                V = W.shape[2]
                d_batch = []
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v, :, :]
                    Y_b_v = Y_b[:, v, :, :]
                    Y_hat_v = differentiable_frechet_mean(w_v, Y_ref_v.to(device), dist_name)
                    d_v = dist_fn(Y_hat_v, Y_b_v)
                    d_batch.append(d_v)
                d_avg = torch.stack(d_batch).mean(dim=0)  # [B]
                dists.append(d_avg.cpu())
    cat = torch.cat(dists)
    return cat.mean().item(), (cat ** 2).mean().item()


def _effective_rank(model_fdrnn):
    """Compute rank metric of the first Linear layer in the reduction net.

    Rank metric = (sum sigma_i)^2 / (sum sigma_i^2), where sigma_i are
    the singular values.  Returns (rank_metric, singular_values_np).
    """
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
    """Compute effective rank of the LoRA response matrix M = B A^T with adaptive pruning.

    Prunes singular values below threshold for adaptive rank selection.
    Returns (rank_metric, singular_values_np).
    """
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


def local_frechet_predict_1d(t_train, Y_train, t_query, dist_name, frechet_mean_fn, h):
    """Local Fréchet prediction in 1D index space."""
    # Convert inputs to tensors
    t_train = torch.tensor(t_train, dtype=torch.float32) if isinstance(t_train, np.ndarray) else t_train
    t_query = torch.tensor(t_query, dtype=torch.float32) if isinstance(t_query, np.ndarray) else t_query
    h = torch.tensor(h, dtype=torch.float32) if isinstance(h, (int, float, np.ndarray)) else h
    
    # Gaussian kernel weights
    w = torch.exp(-0.5 * ((t_train - t_query) / h) ** 2)
    w = w / (torch.sum(w) + 1e-12)
    
    # Convert to torch tensors
    w_torch = w.unsqueeze(0)  # [1, n]
    Y_train_torch = torch.stack(Y_train).float()  # [n, p, p]
    
    # Compute weighted Fréchet mean
    pred = frechet_mean_fn(w_torch, Y_train_torch, dist_name)
    pred = pred.squeeze(0)  # [1, p, p] -> [p, p]
    return pred.detach().numpy()


def ifr_loss(beta, X, Y, dist_name, frechet_mean_fn, h):
    """Loss function for IFR optimization."""
    from src.spd_frechet_adaptive import get_distance_fn
    dist_fn = get_distance_fn(dist_name)
    
    theta = beta / (np.linalg.norm(beta) + 1e-12)
    t = X @ theta
    loss = 0.0
    
    for i in range(len(Y)):
        yhat_i = local_frechet_predict_1d(t, Y, t[i], dist_name, frechet_mean_fn, h)
        # Convert to torch tensors for distance computation
        yhat_i_tensor = torch.tensor(yhat_i, dtype=torch.float32)
        y_i_tensor = Y[i].detach().clone() if isinstance(Y[i], torch.Tensor) else torch.tensor(Y[i], dtype=torch.float32)
        loss += (dist_fn(yhat_i_tensor, y_i_tensor) ** 2).item()
    
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
            # Convert to torch tensors for distance computation
            Y_i_tensor = Y[i].detach().clone() if isinstance(Y[i], torch.Tensor) else torch.tensor(Y[i], dtype=torch.float32)
            Y_j_tensor = Y[j].detach().clone() if isinstance(Y[j], torch.Tensor) else torch.tensor(Y[j], dtype=torch.float32)
            dij = dist_fn(Y_i_tensor, Y_j_tensor)
            D[i, j] = D[j, i] = dij.item()
    
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


def run_single_metric(
    metric_name: str,
    n_train: int = 400,
    n_test: int = 1000,
    df: int = 6,
    hidden_sizes: list = None,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    entropy_reg: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
    dfr_tuned_params: dict = None,
    fdrnn_tuned_params: dict = None,
    structural_dim: int = 5,
    n_responses: int = 2,
):
    """
    Run the full train + evaluate pipeline for one distance metric.

    Methods:  Global Mean, GFR, DFR, FSDRNN (with adaptive LoRA), IFR, Fréchet SDR.

    Args:
        metric_name:  one of the keys in DISTANCE_FUNCTIONS
        n_train:      training set size
        n_test:       test set size
        df:           Wishart degrees of freedom
        hidden_sizes: (unused, kept for CLI compat)
        epochs:       training epochs
        lr:           learning rate
        batch_size:   mini-batch size b
        entropy_reg:  entropy regularisation coefficient
        seed:         random seed
        device:       'cpu' or 'cuda'
        verbose:      print progress
        dfr_tuned_params:      dict of tuned params for DFR
        fdrnn_tuned_params:    dict of tuned params for FSDRNN
        structural_dim:        oracle reduction dimension d0 (default 5)

    Returns:
        results dict with evaluation metrics for all methods
    """
    p = 12  # input dimension for Wishart simulation

    set_seed(seed)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Metric: {metric_name}")
        print(f"  n_train={n_train}, n_test={n_test}, df={df}, seed={seed}")
        print(f"  epochs={epochs}, lr={lr}, d0={structural_dim}")
        if dfr_tuned_params:
            print(f"  [TUNED DFR params: {dfr_tuned_params}]")
        if fdrnn_tuned_params:
            print(f"  [TUNED FSDRNN params: {fdrnn_tuned_params}]")
        print(f"{'='*70}")

    ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
    ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    if verbose:
        X_sample, Y_sample = ds_train[0]
        print(f"\n  Input shape:    X ~ {tuple(X_sample.shape)}")
        print(f"  Response shape: Y ~ {tuple(Y_sample.shape)}")
        print(f"  Y is SPD: eigenvalues = {torch.linalg.eigvalsh(Y_sample).tolist()}")
        true_mean = ds_train.get_true_mean(0)
        if true_mean.dim() == 2:
            print(f"  True mean E[Y|X] diag: {torch.diag(true_mean).tolist()}")
        else:
            for v in range(true_mean.shape[0]):
                print(f"  True mean E[Y|X] response {v} diag: {torch.diag(true_mean[v]).tolist()}")

    # ------------------------------------------------------------------
    # 2. Reference set, validation split, & baseline
    # ------------------------------------------------------------------
    Y_ref = ds_train.Y  # [n_train, 5, 5]

    # Early-stopping validation split (10%)
    val_n = max(1, int(0.1 * len(ds_train)))
    es_train_ds, es_val_ds = random_split(
        ds_train, [len(ds_train) - val_n, val_n],
        generator=torch.Generator().manual_seed(seed),
    )
    es_val_loader = DataLoader(es_val_ds, batch_size=batch_size, shuffle=False)

    dist_fn = get_distance_fn(metric_name)

    # Global mean baseline (ignoring X)
    global_mean = ds_train.Y.mean(dim=0)  # [2, 5, 5]

    baseline_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_batch = Y_batch.to(device)
            B = X_batch.size(0)
            gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1).to(device)  # [B, 2, 5, 5]
            # Compute distance per response and average
            d_per_response = []
            for v in range(2):
                d_v = dist_fn(gm[:, v, :, :], Y_batch[:, v, :, :])
                d_per_response.append(d_v)
            d_base = torch.stack(d_per_response).mean(dim=0)  # [B]
            baseline_dists.append(d_base.cpu())

    baseline_dists_cat = torch.cat(baseline_dists)
    base_avg = baseline_dists_cat.mean().item()
    base_avg_sq = (baseline_dists_cat ** 2).mean().item()

    # AMSPE for baseline: squared distance between global_mean and true_mean
    # Only valid for Frobenius metric where true mean calculation is correct
    if metric_name == "frobenius":
        baseline_amspe_dists = []
        true_mean_test = ds_test.get_true_mean()  # [n_test, V, 5, 5]
        with torch.no_grad():
            for i in range(len(ds_test)):
                true_mean_i = true_mean_test[i].to(device)  # [V, 5, 5]
                gm = global_mean.unsqueeze(0).expand(1, -1, -1, -1).to(device)  # [1, V, 5, 5]
                d_per_response = []
                for v in range(n_responses):
                    d_v = dist_fn(gm[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                baseline_amspe_dists.append(d.cpu())

        baseline_amspe_dists_cat = torch.cat(baseline_amspe_dists)
        base_amspe = baseline_amspe_dists_cat.mean().item()
    else:
        base_amspe = None

    results = {}
    results["metric_name"] = metric_name
    results["baseline_avg_dist"] = base_avg
    results["baseline_avg_dist_sq"] = base_avg_sq
    results["baseline_amspe"] = base_amspe
    results["baseline_time_sec"] = 0.0

    if verbose:
        print(f"\n  --- Competitors ---")
        print(f"  Global Mean avg d:   {base_avg:.6f}")

    # ------------------------------------------------------------------
    # 3. Global Frechet Regression (GFR)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- Global Frechet Regression ---")
    t_gfr = time.time()
    gfr = GlobalFrechetRegression(dist_name=metric_name)
    gfr.fit(ds_train.X, ds_train.Y)

    gfr_dists = []
    for X_batch, Y_batch in test_loader:
        Y_pred_gfr = gfr.predict(X_batch)  # [B, V, 5, 5]
        # Compute distance per response and average
        d_per_response = []
        for v in range(2):
            d_v = dist_fn(Y_pred_gfr[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_gfr = torch.stack(d_per_response).mean(dim=0)  # [B]
        gfr_dists.append(d_gfr)

    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_avg_sq = (gfr_dists_cat ** 2).mean().item()
    gfr_time = time.time() - t_gfr

    # AMSPE for GFR
    # Only valid for Frobenius metric where true mean calculation is correct
    if metric_name == "frobenius":
        gfr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                X_i = ds_test.X[i:i+1].to(device)  # [1, p]
                true_mean_i = true_mean_test[i].to(device)  # [V, 5, 5]
                Y_pred_gfr_i = gfr.predict(X_i)  # [1, V, 5, 5]
                d_per_response = []
                for v in range(n_responses):
                    d_v = dist_fn(Y_pred_gfr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                gfr_amspe_dists.append(d.cpu())

        gfr_amspe_dists_cat = torch.cat(gfr_amspe_dists)
        gfr_amspe = gfr_amspe_dists_cat.mean().item()
    else:
        gfr_amspe = None

    if verbose:
        gfr_imp = (base_avg - gfr_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  GFR avg d:           {gfr_avg:.6f}")
        print(f"  GFR Improvement:     {gfr_imp:.1f}%")
        print(f"  GFR time:            {gfr_time:.1f}s")

    results["gfr_avg_dist"] = gfr_avg
    results["gfr_avg_dist_sq"] = gfr_avg_sq
    results["gfr_amspe"] = gfr_amspe
    results["gfr_time_sec"] = gfr_time

    # ------------------------------------------------------------------
    # 4. Deep Frechet Regression (DFR)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- Deep Frechet Regression (DFR) ---")
    t_dfr = time.time()
    _dfr_md = 2
    _dfr_mk = 10
    _dfr_h = 32
    _dfr_l = 4
    _dfr_ne = epochs  # Use same epochs as FDRNN
    _dfr_lr = lr  # Use same lr
    _dfr_do = 0.3  # Use same dropout
    if dfr_tuned_params:
        _dfr_md = dfr_tuned_params.get("manifold_dim", _dfr_md)
        _dfr_mk = dfr_tuned_params.get("manifold_k", _dfr_mk)
        _dfr_h = dfr_tuned_params.get("hidden", _dfr_h)
        _dfr_l = dfr_tuned_params.get("layer", _dfr_l)
        _dfr_ne = dfr_tuned_params.get("num_epochs", _dfr_ne)
        _dfr_lr = dfr_tuned_params.get("lr", _dfr_lr)
        _dfr_do = dfr_tuned_params.get("dropout", _dfr_do)
    dfr = DeepFrechetRegression(
        dist_name=metric_name,
        manifold_method="isomap",
        manifold_dim=_dfr_md,
        manifold_k=_dfr_mk,
        hidden=_dfr_h,
        layer=_dfr_l,
        num_epochs=_dfr_ne,
        lr=_dfr_lr,
        dropout=_dfr_do,
        seed=seed,
    )
    dfr.fit(ds_train.X, ds_train.Y, verbose=verbose)

    dfr_dists = []
    for X_batch, Y_batch in test_loader:
        Y_pred_dfr = dfr.predict(X_batch)  # [B, V, 5, 5]
        # Compute distance per response and average
        d_per_response = []
        for v in range(2):
            d_v = dist_fn(Y_pred_dfr[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_dfr = torch.stack(d_per_response).mean(dim=0)  # [B]
        dfr_dists.append(d_dfr)

    dfr_dists_cat = torch.cat(dfr_dists)
    dfr_avg = dfr_dists_cat.mean().item()
    dfr_avg_sq = (dfr_dists_cat ** 2).mean().item()
    dfr_time = time.time() - t_dfr

    # AMSPE for DFR
    # Only valid for Frobenius metric where true mean calculation is correct
    if metric_name == "frobenius":
        dfr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                X_i = ds_test.X[i:i+1].to(device)  # [1, p]
                true_mean_i = true_mean_test[i].to(device)  # [V, 5, 5]
                Y_pred_dfr_i = dfr.predict(X_i)  # [1, V, 5, 5]
                d_per_response = []
                for v in range(n_responses):
                    d_v = dist_fn(Y_pred_dfr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                dfr_amspe_dists.append(d.cpu())

        dfr_amspe_dists_cat = torch.cat(dfr_amspe_dists)
        dfr_amspe = dfr_amspe_dists_cat.mean().item()
    else:
        dfr_amspe = None

    if verbose:
        dfr_imp = (base_avg - dfr_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  DFR avg d:           {dfr_avg:.6f}")
        print(f"  DFR Improvement:     {dfr_imp:.1f}%")
        print(f"  DFR time:            {dfr_time:.1f}s")

    results["dfr_avg_dist"] = dfr_avg
    results["dfr_avg_dist_sq"] = dfr_avg_sq
    results["dfr_amspe"] = dfr_amspe
    results["dfr_time_sec"] = dfr_time

    # ------------------------------------------------------------------
    # 5. FSDRNN -- with chosen reduction dimension and adaptive LoRA
    # ------------------------------------------------------------------
    set_seed(seed)
    _fdrnn_lr = fdrnn_tuned_params.get("lr", lr) if fdrnn_tuned_params else lr
    _fdrnn_er = fdrnn_tuned_params.get("entropy_reg", entropy_reg) if fdrnn_tuned_params else entropy_reg
    _fdrnn_wd = fdrnn_tuned_params.get("weight_decay", 1e-5) if fdrnn_tuned_params else 1e-5
    _fdrnn_do = fdrnn_tuned_params.get("dropout", 0.3) if fdrnn_tuned_params else 0.3
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
        dropout=_fdrnn_do,
    )
    if verbose:
        print(f"\n  --- FSDRNN (d={_fdrnn_rd}) ---")
        n_params_fdrnn = sum(pp.numel() for pp in model_fdrnn.parameters())
        print(f"  Parameters: {n_params_fdrnn:,}")
        print(f"  Reduction dim: {_fdrnn_rd}, nuclear_reg: {_fdrnn_nr}")
        print(f"  Max response rank: {_fdrnn_rr} (adaptive LoRA)")

    t_fdrnn = time.time()
    history_fdrnn = train_frechet_model(
        model=model_fdrnn,
        Y_ref=Y_ref,
        train_loader=train_loader,
        dist_name=metric_name,
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
        model_fdrnn, test_loader, Y_ref, metric_name, device)
    fdrnn_train_d, fdrnn_train_d_sq = _evaluate_on_loader(
        model_fdrnn, train_loader, Y_ref, metric_name, device)
    fdrnn_resp_eff_rank, fdrnn_resp_sv = _effective_response_rank(model_fdrnn)

    # AMSPE for FSDRNN
    # Only valid for Frobenius metric where true mean calculation is correct
    if metric_name == "frobenius":
        fdrnn_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                X_i = ds_test.X[i:i+1].to(device)  # [1, p]
                true_mean_i = true_mean_test[i].to(device)  # [V, 5, 5]
                # Get FSDRNN prediction
                W_i = model_fdrnn(X_i)  # [1, n_ref, V]
                W_i = torch.softmax(W_i, dim=1)  # [1, n_ref, V]
                Y_pred_fdrnn_i = []
                for v in range(n_responses):
                    w_v = W_i[:, :, v]  # [1, n_ref]
                    Y_ref_v = Y_ref[:, v, :, :]  # [n_ref, 5, 5]
                    Y_hat_v = differentiable_frechet_mean(w_v, Y_ref_v.to(device), metric_name)  # [1, 5, 5]
                    Y_pred_fdrnn_i.append(Y_hat_v)
                Y_pred_fdrnn_i = torch.stack(Y_pred_fdrnn_i, dim=1)  # [1, V, 5, 5]
                
                d_per_response = []
                for v in range(n_responses):
                    d_v = dist_fn(Y_pred_fdrnn_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                fdrnn_amspe_dists.append(d.cpu())

        fdrnn_amspe_dists_cat = torch.cat(fdrnn_amspe_dists)
        fdrnn_amspe = fdrnn_amspe_dists_cat.mean().item()
    else:
        fdrnn_amspe = None

    if verbose:
        fdrnn_imp = (base_avg - fdrnn_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  FSDRNN avg d:         {fdrnn_avg:.6f}")
        print(f"  FSDRNN Improvement:   {fdrnn_imp:.1f}%")
        print(f"  FSDRNN time:          {fdrnn_time:.1f}s")
        print(f"  FSDRNN train avg d:   {fdrnn_train_d:.6f}  "
              f"(gap={fdrnn_avg - fdrnn_train_d:+.4f})")
        print(f"  FSDRNN adaptive LoRA rank metric: {fdrnn_resp_eff_rank:.2f}")
        if fdrnn_resp_sv is not None:
            print(f"  FSDRNN adaptive LoRA sv: {[f'{s:.4f}' for s in fdrnn_resp_sv]}")

    results["fdrnn_avg_dist"] = fdrnn_avg
    results["fdrnn_avg_dist_sq"] = fdrnn_avg_sq
    results["fdrnn_amspe"] = fdrnn_amspe
    results["fdrnn_time_sec"] = fdrnn_time
    results["fdrnn_reduction_dim"] = _fdrnn_rd
    results["fdrnn_response_eff_rank"] = fdrnn_resp_eff_rank
    results["fdrnn_train_dist"] = fdrnn_train_d
    results["fdrnn_train_dist_sq"] = fdrnn_train_d_sq

    # ------------------------------------------------------------------
    # 6. FSDRNN response LoRA analysis
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- FSDRNN response LoRA analysis ---")
        print(f"  Response LoRA rank metric: {fdrnn_resp_eff_rank:.2f}")
        if fdrnn_resp_sv is not None:
            print(f"  Response LoRA sv: {[f'{s:.4f}' for s in fdrnn_resp_sv]}")

    # ------------------------------------------------------------------
    # 7. IFR: Single-index Fréchet Regression (response-by-response)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- IFR (Single-index Fréchet Regression) ---")
    t_ifr = time.time()

    # Convert data to numpy for IFR
    X_train_np = ds_train.X.numpy()
    X_test_np = ds_test.X.numpy()

    # Bandwidth for kernel (rule of thumb)
    h = 1.06 * X_train_np.std() * (len(X_train_np)) ** (-1/5)

    ifr_preds = []
    for v in range(n_responses):
        if verbose and v == 0:
            print(f"    Fitting IFR for response {v+1}/{n_responses}...")

        # Extract response v
        Y_train_v = [ds_train.Y[i, v] for i in range(len(ds_train))]
        Y_test_v = [ds_test.Y[i, v] for i in range(len(ds_test))]

        # Fit IFR for this response
        theta_v = fit_ifr(X_train_np, Y_train_v, metric_name, differentiable_frechet_mean, h)

        # Predict
        preds_v = predict_ifr(X_train_np, Y_train_v, theta_v, X_test_np, metric_name, differentiable_frechet_mean, h)
        ifr_preds.append(preds_v)

    # Stack predictions: [n_test, V, 5, 5]
    ifr_preds_stacked = torch.stack([
        torch.stack([
            pred.detach().clone() if isinstance(pred, torch.Tensor) else torch.tensor(pred, dtype=torch.float32)
            for pred in ifr_preds[v]
        ], dim=0)
        for v in range(n_responses)
    ], dim=1)

    ifr_dists = []
    with torch.no_grad():
        for i in range(len(ds_test)):
            Y_true_i = ds_test.Y[i].unsqueeze(0).to(device)  # [1, V, 5, 5]
            Y_pred_ifr_i = ifr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 5, 5]

            d_per_response = []
            for v in range(n_responses):
                d_v = dist_fn(Y_pred_ifr_i[:, v, :, :], Y_true_i[:, v, :, :])  # [1]
                d_per_response.append(d_v)
            d = torch.stack(d_per_response).mean(dim=0)  # [1]
            ifr_dists.append(d.cpu())

    ifr_dists_cat = torch.cat(ifr_dists)
    ifr_avg = ifr_dists_cat.mean().item()
    ifr_avg_sq = (ifr_dists_cat ** 2).mean().item()
    ifr_time = time.time() - t_ifr

    # AMSPE for IFR (only for Frobenius)
    if metric_name == "frobenius":
        ifr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                true_mean_i = true_mean_test[i].to(device)  # [V, 5, 5]
                Y_pred_ifr_i = ifr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 5, 5]

                d_per_response = []
                for v in range(n_responses):
                    d_v = dist_fn(Y_pred_ifr_i[:, v, :, :], true_mean_i[v:v+1, :, :])  # [1]
                    d_per_response.append(d_v ** 2)  # Square the distance
                d = torch.stack(d_per_response).mean(dim=0)  # [1]
                ifr_amspe_dists.append(d.cpu())

        ifr_amspe_dists_cat = torch.cat(ifr_amspe_dists)
        ifr_amspe = ifr_amspe_dists_cat.mean().item()
    else:
        ifr_amspe = None

    results["ifr_avg_dist"] = ifr_avg
    results["ifr_avg_dist_sq"] = ifr_avg_sq
    results["ifr_amspe"] = ifr_amspe
    results["ifr_time_sec"] = ifr_time

    if verbose:
        print(f"  IFR avg Dist:          {ifr_avg:.6f} (time: {ifr_time:.1f}s)")

    # ------------------------------------------------------------------
    # 8. Fréchet SDR (SIR ensemble, response-by-response)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- Fréchet SDR (SIR ensemble) ---")
    t_fsdr = time.time()

    fsdr_preds = []
    for v in range(n_responses):
        if verbose and v == 0:
            print(f"    Fitting Fréchet SDR for response {v+1}/{n_responses}...")

        # Extract response v
        Y_train_v = [ds_train.Y[i, v] for i in range(len(ds_train))]
        Y_test_v = [ds_test.Y[i, v] for i in range(len(ds_test))]

        # Fit Fréchet SDR for this response
        B_hat, _ = frechet_sdr(X_train_np, Y_train_v, get_distance_fn(metric_name), 1)
        theta_v = B_hat[:, 0]

        # Predict using local Fréchet regression along the SDR direction
        preds_v = predict_ifr(X_train_np, Y_train_v, theta_v, X_test_np, metric_name, differentiable_frechet_mean, h)
        fsdr_preds.append(preds_v)

    # Stack predictions: [n_test, V, 5, 5]
    fsdr_preds_stacked = torch.stack([
        torch.stack([
            pred.detach().clone() if isinstance(pred, torch.Tensor) else torch.tensor(pred, dtype=torch.float32)
            for pred in fsdr_preds[v]
        ], dim=0)
        for v in range(n_responses)
    ], dim=1)

    fsdr_dists = []
    with torch.no_grad():
        for i in range(len(ds_test)):
            Y_true_i = ds_test.Y[i].unsqueeze(0).to(device)  # [1, V, 5, 5]
            Y_pred_fsdr_i = fsdr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 5, 5]

            d_per_response = []
            for v in range(n_responses):
                d_v = dist_fn(Y_pred_fsdr_i[:, v, :, :], Y_true_i[:, v, :, :])  # [1]
                d_per_response.append(d_v)
            d = torch.stack(d_per_response).mean(dim=0)  # [1]
            fsdr_dists.append(d.cpu())

    fsdr_dists_cat = torch.cat(fsdr_dists)
    fsdr_avg = fsdr_dists_cat.mean().item()
    fsdr_avg_sq = (fsdr_dists_cat ** 2).mean().item()
    fsdr_time = time.time() - t_fsdr

    # AMSPE for Fréchet SDR (only for Frobenius)
    if metric_name == "frobenius":
        fsdr_amspe_dists = []
        with torch.no_grad():
            for i in range(len(ds_test)):
                true_mean_i = true_mean_test[i].to(device)  # [V, 5, 5]
                Y_pred_fsdr_i = fsdr_preds_stacked[i].unsqueeze(0).to(device)  # [1, V, 5, 5]

                d_per_response = []
                for v in range(n_responses):
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

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print(f"  COMPARISON TABLE  ({metric_name}, d0={structural_dim})")
        print(f"{'='*80}")
        print(f"  {'Method':<25} {'avg d(pred,Y)':>14} {'AMSP':>14} {'Improv%':>8} {'Time(s)':>8}")
        print(f"  {'-'*77}")

        def _imp(val):
            return (base_avg - val) / base_avg * 100 if base_avg > 0 else 0

        def _format_amspe(amspe_val):
            return f"{amspe_val:>14.6f}" if amspe_val is not None else f"{'---':>14}"

        print(f"  {'Global Mean':<25} {base_avg:>14.6f} {_format_amspe(base_amspe)} {'---':>8} {'0.0':>8}")
        print(f"  {'GFR':<25} {gfr_avg:>14.6f} {_format_amspe(gfr_amspe)} {_imp(gfr_avg):>7.1f}% {gfr_time:>7.1f}")
        print(f"  {'DFR':<25} {dfr_avg:>14.6f} {_format_amspe(dfr_amspe)} {_imp(dfr_avg):>7.1f}% {dfr_time:>7.1f}")
        print(f"  {'FSDRNN (d=':<8}{_fdrnn_rd}, adaptive){'':<10} {fdrnn_avg:>14.6f} {_format_amspe(fdrnn_amspe)} {_imp(fdrnn_avg):>7.1f}% {fdrnn_time:>7.1f}")
        print(f"  {'IFR':<25} {ifr_avg:>14.6f} {_format_amspe(ifr_amspe)} {_imp(ifr_avg):>7.1f}% {ifr_time:>7.1f}")
        print(f"  {'Fréchet SDR':<25} {fsdr_avg:>14.6f} {_format_amspe(fsdr_amspe)} {_imp(fsdr_avg):>7.1f}% {fsdr_time:>7.1f}")

        print(f"\n  --- Train / Test gap (d) ---")
        print(f"  {'Method':<18} {'Train d':>10} {'Test d':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        for tag, tr, te in [
            ("FSDRNN (adaptive)", fdrnn_train_d, fdrnn_avg),
        ]:
            print(f"  {tag:<18} {tr:>10.4f} {te:>10.4f} {te - tr:>+10.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test: Frechet mean regression for SPD matrix responses"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="power",
        choices=list(DISTANCE_FUNCTIONS.keys()) + ["all"],
        help="Distance metric to use (default: power)",
    )
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--df", type=int, default=6, help="Wishart degrees of freedom")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entropy_reg", type=float, default=0.01,
                        help="entropy regularisation coefficient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=15,
                        help="Number of repetitions (default: 15)")
    parser.add_argument("--structural_dim", type=int, default=5,
                        help="Oracle reduction dimension d0 (default: 5)")
    parser.add_argument("--n_responses", type=int, default=5,
                        help="Number of responses per sample (default: 5)")
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[128, 128, 64],
        help="Hidden layer sizes (unused, kept for compat)",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--tune", action="store_true", default=True,
        help="Run grid search to select hyperparameters before the experiment",
    )
    parser.add_argument(
        "--no-tune", dest="tune", action="store_false",
        help="Skip hyperparameter tuning and use default parameters",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    metrics_to_run = (
        list(DISTANCE_FUNCTIONS.keys()) if args.metric == "all" else [args.metric]
    )

    n_reps = args.n_reps
    structural_dim = args.structural_dim
    p = 12  # input dim for Wishart simulation

    for metric_name in metrics_to_run:
        # ==============================================================
        # Optional: hyperparameter tuning via grid search
        # ==============================================================
        dfr_tuned_params = None
        fdrnn_tuned_params = None
        chosen_d = None

        # Use pre-tuned hyperparameters
        dfr_tuned_params = {'hidden': 16, 'layer': 3, 'lr': 0.0005, 'manifold_dim': 2}
        fdrnn_tuned_params = {'dropout': 0.3, 'entropy_reg': 0.0, 'lr': 0.001, 'nuclear_reg': 0.0001, 'reduction_dim': 2, 'reduction_type': 'nonlinear', 'response_rank': 5}
        chosen_d = 2

        print(f"\n  Using pre-tuned hyperparameters:")
        print(f"  DFR: {dfr_tuned_params}")
        print(f"  FSDRNN: {fdrnn_tuned_params}")
        print()

        # ==============================================================
        # Main experiment
        # ==============================================================
        print(f"  METRIC: {metric_name}  |  {n_reps} repetitions")
        print(f"  reduction dim d={chosen_d} (pre-tuned)")
        if dfr_tuned_params:
            print(f"  DFR pre-tuned:       {dfr_tuned_params}")
        if fdrnn_tuned_params:
            print(f"  FSDRNN (d={chosen_d}, adaptive r) pre-tuned:  {fdrnn_tuned_params}")
        print(f"{'#'*70}")

        # Resolve FDRNN reduction_dim labels
        if fdrnn_tuned_params is None:
            chosen_d = p  # default to no reduction
            _fdrnn_rr = "adaptive"
        else:
            _fdrnn_rr = "adaptive"

        # Collect per-rep results
        methods = [
            ("Global Mean",                       "baseline"),
            ("GFR",                               "gfr"),
            ("DFR",                               "dfr"),
            (f"FSDRNN (d={chosen_d}, adaptive LoRA)", "fdrnn"),
        ]
        collect_suffixes = ["_avg_dist", "_avg_dist_sq", "_time_sec"]
        all_vals = {}
        for _, prefix in methods:
            for suf in collect_suffixes:
                all_vals[prefix + suf] = []
        # Train-set evaluation for FDRNN methods
        for prefix in ["fdrnn"]:
            for suf in ["_train_dist", "_train_dist_sq"]:
                all_vals[prefix + suf] = []
        # Rank metric tracking
        all_vals["fdrnn_response_eff_rank"] = []

        for rep in range(n_reps):
            rep_seed = args.seed + rep
            verbose_rep = (rep == 0)
            if not verbose_rep:
                print(f"  rep {rep + 1}/{n_reps} (seed={rep_seed}) ...",
                      end="", flush=True)

            res = run_single_metric(
                metric_name=metric_name,
                n_train=args.n_train,
                n_test=args.n_test,
                df=args.df,
                hidden_sizes=args.hidden_sizes,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                entropy_reg=args.entropy_reg,
                seed=rep_seed,
                device=device,
                verbose=verbose_rep,
                dfr_tuned_params=dfr_tuned_params,
                fdrnn_tuned_params=fdrnn_tuned_params,
                structural_dim=structural_dim,
                n_responses=args.n_responses,
            )
            for _, prefix in methods:
                for suf in collect_suffixes:
                    all_vals[prefix + suf].append(res[prefix + suf])
            for prefix in ["fdrnn"]:
                for suf in ["_train_dist", "_train_dist_sq"]:
                    key = prefix + suf
                    if key in res:
                        all_vals[key].append(res[key])
            all_vals["fdrnn_response_eff_rank"].append(res.get("fdrnn_response_eff_rank", float('nan')))

            if not verbose_rep:
                print(f"  FDRNN(response_rank)={res['fdrnn_avg_dist']:.4f}")

        # ------------------------------------------------------------------
        # Paper-ready tables: d and d^2  (mean, SE, improvement%, runtime)
        # ------------------------------------------------------------------
        W = 82
        eff_ranks = np.array(all_vals["fdrnn_response_eff_rank"])
        header_lines = [
            f"  RESULTS: {metric_name}  ({n_reps} reps)",
            f"  n_train={args.n_train}, n_test={args.n_test}, "
            f"df={args.df}, epochs={args.epochs}",
        ]
        if n_reps > 1:
            header_lines.append(
                f"  d={chosen_d}, FSDRNN adaptive LoRA rank metric: "
                f"{eff_ranks.mean():.2f} (SE {eff_ranks.std(ddof=1)/np.sqrt(n_reps):.2f})"
            )
        else:
            header_lines.append(
                f"  d={chosen_d}, FSDRNN adaptive LoRA rank metric: {eff_ranks.mean():.2f}"
            )
        if dfr_tuned_params:
            header_lines.append(f"  DFR pre-tuned:       {dfr_tuned_params}")
        if fdrnn_tuned_params:
            header_lines.append(f"  FSDRNN (adaptive) pre-tuned:  {fdrnn_tuned_params}")

        print(f"\n\n{'=' * W}")
        for hl in header_lines:
            print(hl)
        print(f"{'=' * W}")

        base_prefix = methods[0][1]  # "baseline"
        for panel_label, suf in [("Panel A: d(pred, Y)", "_avg_dist"),
                                  ("Panel B: d^2(pred, Y)", "_avg_dist_sq")]:
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

        # ------------------------------------------------------------------
        # Train / Test gap summary (overfitting check)
        # ------------------------------------------------------------------
        gap_methods = [
            (f"FSDRNN (adaptive)", "fdrnn"),
            ("FSDRNN *", "fdrnn_td"),
        ]
        has_gap = all(all_vals.get(f"{pfx}_train_dist", [])
                      for _, pfx in gap_methods)
        if has_gap:
            print(f"\n  Train / Test gap (d, higher => more overfit)")
            print(f"  {'Method':<28} {'Train':>10} {'Test':>10} {'Gap':>10}")
            print(f"  {'-' * (W - 4)}")
            for label, prefix in gap_methods:
                tr = np.mean(all_vals[f"{prefix}_train_dist"])
                te = np.mean(all_vals[f"{prefix}_avg_dist"])
                gap = te - tr
                print(f"  {label:<28} {tr:>10.4f} {te:>10.4f} {gap:>10.4f}")

        # ------------------------------------------------------------------
        # GFR diagnostic
        # ------------------------------------------------------------------
        gfr_avg_d = np.mean(all_vals["gfr_avg_dist"])
        fdrnn_avg_d = np.mean(all_vals["fdrnn_avg_dist"])
        if gfr_avg_d <= fdrnn_avg_d:
            print(f"\n  WARNING: GFR (avg d={gfr_avg_d:.4f}) <= FSDRNN "
                  f"(avg d={fdrnn_avg_d:.4f}). Nonlinear capacity may be "
                  f"unnecessary for this metric.")

        # --------------------------------------------------------------
        # Figure: distribution of prediction errors across reps
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        panel_specs = [
            ("$d(\\hat{Y}, Y)$", "_avg_dist"),
            ("$d^2(\\hat{Y}, Y)$", "_avg_dist_sq"),
        ]
        short_labels = [label.split("(")[0].strip() for label, _ in methods]
        colors = ["#999999", "#4daf4a", "#e41a1c", "#377eb8", "#984ea3"]

        for ax, (ylabel, suf) in zip(axes, panel_specs):
            data = [np.array(all_vals[prefix + suf]) for _, prefix in methods]
            bplot = ax.boxplot(
                data, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch, c in zip(bplot["boxes"], colors[:len(methods)]):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_xticklabels(short_labels, rotation=25,
                               ha="right", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Wishart SPD  |  metric={metric_name}  |  "
            f"n={args.n_train}, df={args.df}  |  {n_reps} reps",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fig_dir = os.path.join(project_root, "logs")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(
            fig_dir, f"wishart_{metric_name}_errors.pdf",
        )
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Figure saved -> {fig_path}")


if __name__ == "__main__":
    main()
