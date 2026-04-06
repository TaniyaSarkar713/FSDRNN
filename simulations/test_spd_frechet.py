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
    4. FDRNN (d=p)           -- Frechet NN without predictor reduction
    5. FDRNN * (full)        -- FDRNN with full predictor dimension, nuclear tuned

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

from src.spd_frechet import (
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
    """Compute effective rank of the LoRA response matrix M = B A^T.

    Returns (rank_metric, singular_values_np) if response_rank is not None,
    else (n_responses, None).
    """
    if model.response_rank is None:
        return model.n_responses, None
    with torch.no_grad():
        alpha = model.response_refine.alpha
        rank = model.response_refine.rank
        B = model.response_refine.B
        A = model.response_refine.A
        I = torch.eye(model.n_responses, device=B.device, dtype=B.dtype)
        delta = B @ A.T
        M = I + (alpha / rank) * delta
        sv = torch.linalg.svdvals(M).cpu().numpy()
    eff = float((sv.sum() ** 2) / (sv ** 2).sum()) if (sv ** 2).sum() > 0 else 0.0
    return eff, sv


def run_single_metric(
    metric_name: str,
    n_train: int = 200,
    n_test: int = 50,
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

    Methods:  Global Mean, GFR, DFR, FDRNN (with tuned d and r).

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
        fdrnn_tuned_params:    dict of tuned params for FDRNN with d=p
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
            print(f"  [TUNED FDRNN(d=p) params: {fdrnn_tuned_params}]")
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

    # Early-stopping validation split (20%)
    val_n = max(1, int(0.2 * len(ds_train)))
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

    results = {}
    results["metric_name"] = metric_name
    results["baseline_avg_dist"] = base_avg
    results["baseline_avg_dist_sq"] = base_avg_sq
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

    if verbose:
        gfr_imp = (base_avg - gfr_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  GFR avg d:           {gfr_avg:.6f}")
        print(f"  GFR Improvement:     {gfr_imp:.1f}%")
        print(f"  GFR time:            {gfr_time:.1f}s")

    results["gfr_avg_dist"] = gfr_avg
    results["gfr_avg_dist_sq"] = gfr_avg_sq
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
    _dfr_ne = 200
    _dfr_lr = 5e-4
    _dfr_do = 0.0
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

    if verbose:
        dfr_imp = (base_avg - dfr_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  DFR avg d:           {dfr_avg:.6f}")
        print(f"  DFR Improvement:     {dfr_imp:.1f}%")
        print(f"  DFR time:            {dfr_time:.1f}s")

    results["dfr_avg_dist"] = dfr_avg
    results["dfr_avg_dist_sq"] = dfr_avg_sq
    results["dfr_time_sec"] = dfr_time

    # ------------------------------------------------------------------
    # 5. FDRNN (d < p) with LoRA for response rank reduction
    # ------------------------------------------------------------------
    set_seed(seed)
    _fdrnn_lr = fdrnn_tuned_params.get("lr", lr) if fdrnn_tuned_params else lr
    _fdrnn_er = fdrnn_tuned_params.get("entropy_reg", entropy_reg) if fdrnn_tuned_params else entropy_reg
    _fdrnn_wd = fdrnn_tuned_params.get("weight_decay", 1e-5) if fdrnn_tuned_params else 1e-5
    _fdrnn_do = fdrnn_tuned_params.get("dropout", 0.0) if fdrnn_tuned_params else 0.0
    _fdrnn_nr = fdrnn_tuned_params.get("nuclear_reg", 0.01) if fdrnn_tuned_params else 0.01
    _fdrnn_rd = fdrnn_tuned_params.get("reduction_dim", p) if fdrnn_tuned_params else p
    if verbose:
        print(f"\n  --- FDRNN (d={_fdrnn_rd}) with LoRA ---")
    _fdrnn_rt = fdrnn_tuned_params.get("reduction_type", "nonlinear") if fdrnn_tuned_params else "nonlinear"
    _fdrnn_rr = fdrnn_tuned_params.get("response_rank", 5) if fdrnn_tuned_params else 5
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
        n_params_fdrnn = sum(pp.numel() for pp in model_fdrnn.parameters())
        print(f"  Parameters: {n_params_fdrnn:,}")
        print(f"  Reduction dim: {_fdrnn_rd}, nuclear_reg: {_fdrnn_nr}")
        print(f"  Response rank: {_fdrnn_rr}")

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

    if verbose:
        fdrnn_imp = (base_avg - fdrnn_avg) / base_avg * 100 if base_avg > 0 else 0
        print(f"  FDRNN avg d:         {fdrnn_avg:.6f}")
        print(f"  FDRNN Improvement:   {fdrnn_imp:.1f}%")
        print(f"  FDRNN time:          {fdrnn_time:.1f}s")
        print(f"  FDRNN train avg d:   {fdrnn_train_d:.6f}  "
              f"(gap={fdrnn_avg - fdrnn_train_d:+.4f})")
        print(f"  FDRNN response LoRA rank metric:  {fdrnn_resp_eff_rank:.2f}")
        if fdrnn_resp_sv is not None:
            print(f"  FDRNN response LoRA sv: {['.4f' for s in fdrnn_resp_sv]}")

    results["fdrnn_avg_dist"] = fdrnn_avg
    results["fdrnn_avg_dist_sq"] = fdrnn_avg_sq
    results["fdrnn_time_sec"] = fdrnn_time
    results["fdrnn_reduction_dim"] = _fdrnn_rd
    results["fdrnn_response_eff_rank"] = fdrnn_resp_eff_rank
    results["fdrnn_train_dist"] = fdrnn_train_d
    results["fdrnn_train_dist_sq"] = fdrnn_train_d_sq

    # ------------------------------------------------------------------
    # 6. FDRNN (d=p) response LoRA analysis
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- FDRNN (d=p) response LoRA analysis ---")
        print(f"  Response LoRA rank metric: {fdrnn_resp_eff_rank:.2f}")
        if fdrnn_resp_sv is not None:
            print(f"  Response LoRA sv: {[f'{s:.4f}' for s in fdrnn_resp_sv]}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE  ({metric_name}, d0={structural_dim})")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'avg d(pred,Y)':>14} {'Improv%':>8}")
        print(f"  {'-'*52}")

        def _imp(val):
            return (base_avg - val) / base_avg * 100 if base_avg > 0 else 0

        print(f"  {'Global Mean':<30} {base_avg:>14.6f} {'---':>8}")
        print(f"  {'GFR':<30} {gfr_avg:>14.6f} {_imp(gfr_avg):>7.1f}%")
        print(f"  {'DFR':<30} {dfr_avg:>14.6f} {_imp(dfr_avg):>7.1f}%")
        print(f"  {'FDRNN (d=p=':<13}{_fdrnn_rd}){'':<15} {fdrnn_avg:>14.6f} {_imp(fdrnn_avg):>7.1f}%")

        print(f"\n  --- Train / Test gap (d) ---")
        print(f"  {'Method':<18} {'Train d':>10} {'Test d':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        for tag, tr, te in [
            ("FDRNN (d=p)", fdrnn_train_d, fdrnn_avg),
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
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--df", type=int, default=6, help="Wishart degrees of freedom")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--entropy_reg", type=float, default=0.01,
                        help="entropy regularisation coefficient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=5,
                        help="Number of repetitions (default: 5)")
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

        if args.tune:
            print(f"\n{'#'*70}")
            print(f"  TUNING: {metric_name}")
            print(f"{'#'*70}")

            tune_ds = WishartSPDDataset(
                n=args.n_train, n_responses=args.n_responses, df=args.df, seed=0
            )

            # --- DFR grid search ---
            print("\n  === DFR hyperparameter search ===")
            dfr_grid = {
                "manifold_dim": [1, 2],
                "hidden": [16, 32],
                "layer": [3, 4],
                "lr": [5e-4, 1e-3],
            }
            dfr_result = grid_search_dfr(
                dataset=tune_ds,
                parent_X=tune_ds.X,
                parent_Y=tune_ds.Y,
                dist_name=metric_name,
                param_grid=dfr_grid,
                val_frac=0.2,
                batch_size=args.batch_size,
                seed=0,
                verbose=True,
            )
            dfr_tuned_params = dfr_result["best_params"]

            # --- FDRNN two-stage tuning: d then r ---
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
                    "response_rank": [1],  # fixed for Stage A
                    "reduction_type": [rt],  # single value
                    "dropout": [0.0],  # fixed
                }
                
                result = grid_search_frechet(
                    dataset=tune_ds,
                    parent_Y=tune_ds.Y,
                    model_class=FrechetDRNN,
                    dist_name=metric_name,
                    param_grid=fdrnn_stage_a_grid,
                    fixed_model_kwargs={
                        "input_dim": p,
                        "n_responses": args.n_responses,
                        "encoder_sizes": [128, 64],
                        "head_sizes": [64, 128],
                    },
                    fixed_train_kwargs={"epochs": 50},  # Reduced epochs for tuning
                    val_frac=0.2,
                    batch_size=args.batch_size,
                    device=device,
                    seed=0,
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

            print(f"\n  === FDRNN Stage B: tune regularization for d={chosen_d}, type={chosen_rt} ===")
            fdrnn_stage_b_grid = {
                "lr": [5e-4, 1e-3],
                "entropy_reg": [0.0, 1e-3],
                "nuclear_reg": [0.0, 1e-4],
                "response_rank": [r for r in [2, 5, 8] if r < args.n_responses] or [1],  # standard LoRA
                "reduction_dim": [chosen_d],  # fixed from Stage A
                "reduction_type": [chosen_rt],  # fixed from Stage A
                "dropout": [0.0],  # fixed
            }
            fdrnn_result = grid_search_frechet(
                dataset=tune_ds,
                parent_Y=tune_ds.Y,
                model_class=FrechetDRNN,
                dist_name=metric_name,
                param_grid=fdrnn_stage_b_grid,
                fixed_model_kwargs={
                    "input_dim": p,
                    "n_responses": args.n_responses,
                    "encoder_sizes": [128, 64],
                    "head_sizes": [64, 128],
                },
                fixed_train_kwargs={"epochs": 50},  # Reduced epochs for tuning
                val_frac=0.2,
                batch_size=args.batch_size,
                device=device,
                seed=0,
                verbose=True,
            )
            fdrnn_tuned_params = fdrnn_result["best_params"]
            fdrnn_tuned_params["reduction_dim"] = chosen_d  # ensure it's set

            print(f"\n  --- Chosen DFR params:              {dfr_tuned_params}")
            print(f"  --- Chosen FDRNN (d={chosen_d}) params: {fdrnn_tuned_params}")
            print()

        # ==============================================================
        # Main experiment
        # ==============================================================
        print(f"\n{'#'*70}")
        print(f"  METRIC: {metric_name}  |  {n_reps} repetitions")
        print(f"  no reduction: d=p={p} (structural d0={structural_dim})")
        if dfr_tuned_params:
            print(f"  DFR tuned:          {dfr_tuned_params}")
        if fdrnn_tuned_params:
            print(f"  FDRNN (d={chosen_d}) tuned:  {fdrnn_tuned_params}")
        print(f"{'#'*70}")

        # Resolve FDRNN reduction_dim labels
        if fdrnn_tuned_params is None:
            chosen_d = p  # default to no reduction
            _fdrnn_rr = 5  # default LoRA rank
        else:
            _fdrnn_rr = fdrnn_tuned_params.get("response_rank", None)

        # Collect per-rep results
        methods = [
            ("Global Mean",                       "baseline"),
            ("GFR",                               "gfr"),
            ("DFR",                               "dfr"),
            (f"FDRNN (d={chosen_d}, response_rank={_fdrnn_rr})", "fdrnn"),
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
                f"  d0={structural_dim}, FDRNN response LoRA rank metric: "
                f"{eff_ranks.mean():.2f} (SE {eff_ranks.std(ddof=1)/np.sqrt(n_reps):.2f})"
            )
        else:
            header_lines.append(
                f"  d0={structural_dim}, FDRNN response LoRA rank metric: {eff_ranks.mean():.2f}"
            )
        if dfr_tuned_params:
            header_lines.append(f"  DFR tuned:          {dfr_tuned_params}")
        if fdrnn_tuned_params:
            header_lines.append(f"  FDRNN (d=p) tuned:  {fdrnn_tuned_params}")

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
            (f"FDRNN (d=p={p})", "fdrnn"),
            ("FDRNN *", "fdrnn_td"),
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
            print(f"\n  WARNING: GFR (avg d={gfr_avg_d:.4f}) <= FDRNN "
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
