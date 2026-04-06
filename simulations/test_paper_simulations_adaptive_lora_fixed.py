#!/usr/bin/env python3
"""
Simulation setups from "Deep Fréchet Regression" paper (Petersen & Müller, 2025)
Comparison of Fréchet regression methods on SPD manifolds.

This script implements the simulation studies from the paper, comparing:
- Global Mean (baseline)
- Global Fréchet Regression (GFR)
- Deep Fréchet Regression (DFR)
- Fréchet Deep Neural Networks (FDRNN variants with adaptive LoRA)

Data generating mechanisms include:
1. Wishart SPD matrices with varying degrees of freedom
2. Different predictor distributions and relationships
3. Multiple distance metrics on SPD manifolds
4. Varying sample sizes and dimensions
"""

import sys
import os
import argparse
import math
import time
from typing import Dict, List, Tuple, Optional

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


def setup_1_basic(n_train: int = 200, n_test: int = 100, df: int = 6,
                  n_responses: int = 1, seed: int = 42):
    """Basic simulation setup with standard parameters."""
    set_seed(seed)
    ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
    ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
    return ds_train, ds_test


def setup_2_multi_response(n_train: int = 200, n_test: int = 100, df: int = 6,
                          n_responses: int = 3, seed: int = 42):
    """Multi-response simulation setup."""
    set_seed(seed)
    ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
    ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
    return ds_train, ds_test


def setup_3_low_df(n_train: int = 200, n_test: int = 100, df: int = 6,
                   n_responses: int = 1, seed: int = 42):
    """Low degrees of freedom simulation setup."""
    set_seed(seed)
    ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
    ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
    return ds_train, ds_test


def setup_4_high_df(n_train: int = 200, n_test: int = 100, df: int = 12,
                    n_responses: int = 1, seed: int = 42):
    """High degrees of freedom simulation setup."""
    set_seed(seed)
    ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
    ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
    return ds_train, ds_test


def setup_5_large_scale(n_train: int = 500, n_test: int = 200, df: int = 6,
                       n_responses: int = 1, seed: int = 42):
    """Large-scale simulation setup."""
    set_seed(seed)
    ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
    ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
    return ds_train, ds_test


def _distance_to_batch_vector(d: torch.Tensor, batch_size: int, name: str = "distance") -> torch.Tensor:
    """
    Coerce a distance tensor into shape [batch_size].

    This is a defensive wrapper for cases where a distance function or
    prediction path leaves an extra response axis in the output, e.g.
    [V, B] or [B, V], instead of the expected [B].
    """
    d = torch.as_tensor(d)

    if d.ndim == 0:
        return d.repeat(batch_size)

    if d.ndim == 1:
        if d.numel() != batch_size:
            raise ValueError(
                f"{name} has shape {tuple(d.shape)} but expected length {batch_size}."
            )
        return d

    batch_axes = [i for i, s in enumerate(d.shape) if s == batch_size]
    if not batch_axes:
        raise ValueError(
            f"{name} has shape {tuple(d.shape)} and no axis matches batch size {batch_size}."
        )

    batch_axis = batch_axes[0]
    if batch_axis != 0:
        d = torch.movedim(d, batch_axis, 0)

    return d.reshape(batch_size, -1).mean(dim=1)


@torch.no_grad()
def evaluate_frechet_model_safe(
    model,
    Y_ref: torch.Tensor,
    test_loader,
    dist_name: str = "frobenius",
    device: str = "cpu",
):
    """
    Local evaluation wrapper that is robust to extra non-batch dimensions in
    the returned distance tensors.

    This avoids shape-mismatch failures inside the library evaluator without
    changing spd_frechet_adaptive.py.
    """
    model.eval()
    model.to(device)
    Y_ref = Y_ref.to(device)
    dist_fn = get_distance_fn(dist_name)

    all_dists = []

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        B = X_batch.size(0)

        logits = model(X_batch)
        W = torch.softmax(logits, dim=1)

        if W.dim() == 2:
            if Y_ref.dim() == 4:
                Y_ref_use = Y_ref[:, 0, :, :]
            else:
                Y_ref_use = Y_ref
            m_w = differentiable_frechet_mean(W, Y_ref_use, dist_name)
            d = dist_fn(m_w, Y_batch)
            all_dists.append(_distance_to_batch_vector(d, B, name="single-response distance").cpu())
        else:
            V = W.shape[2]
            d_per_response = []
            for v in range(V):
                w_v = W[:, :, v]
                Y_ref_v = Y_ref[:, v, :, :]
                Y_batch_v = Y_batch[:, v, :, :]
                m_w_v = differentiable_frechet_mean(w_v, Y_ref_v, dist_name)
                d_v = dist_fn(m_w_v, Y_batch_v)
                d_per_response.append(
                    _distance_to_batch_vector(d_v, B, name=f"multi-response distance (v={v})")
                )

            d_avg = torch.stack(d_per_response, dim=0).mean(dim=0)
            all_dists.append(d_avg.cpu())

    all_dists = torch.cat(all_dists, dim=0)
    return {
        "avg_dist": all_dists.mean().item(),
        "avg_dist_sq": (all_dists ** 2).mean().item(),
    }



def run_single_experiment(
    setup_func,
    metric_name: str = "power",
    n_train: int = 200,
    n_test: int = 100,
    df: int = 6,
    n_responses: int = 1,
    epochs: int = 200,
    lr: float = 1e-3,
    entropy_reg: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    dfr_tuned_params: Optional[Dict] = None,
    fdrnn_tuned_params: Optional[Dict] = None,
) -> Dict:
    """
    Run a single experiment comparing all methods on one setup.

    Returns dict with results for each method.
    """
    set_seed(seed)

    # Get datasets
    ds_train, ds_test = setup_func(n_train=n_train, n_test=n_test, df=df,
                                   n_responses=n_responses, seed=seed)

    # Data loaders
    batch_size = min(32, len(ds_train))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Early stopping validation set (10% of training data)
    val_size = max(1, len(ds_train) // 10)
    train_size = len(ds_train) - val_size
    train_subset, val_subset = random_split(
        ds_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    es_val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Reference matrices for FDRNN
    Y_ref = ds_train.Y  # [n_train, n_responses, 5, 5]
    p = ds_train.X.shape[1]  # predictor dimension

    dist_fn = get_distance_fn(metric_name)
    results = {
        "setup": setup_func.__name__,
        "metric": metric_name,
        "n_train": n_train,
        "n_test": n_test,
        "df": df,
        "n_responses": n_responses,
        "seed": seed,
    }

    print(f"  n_train={n_train}, n_test={n_test}, df={df}, n_responses={n_responses}")

    # Global mean baseline
    print(f"  Running Global Mean...")
    global_mean = ds_train.Y.mean(dim=0)

    base_dists = []
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        B = X_batch.size(0)
        gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1).to(device)
        d_per_response = []
        for v in range(n_responses):
            d_v = dist_fn(gm[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_base = torch.stack(d_per_response).mean(dim=0)
        base_dists.append(d_base)
    base_dists_cat = torch.cat(base_dists)
    base_avg = base_dists_cat.mean().item()

    print(f"  Global Mean: {base_avg:.6f}")
    results.update({"global_mean_avg_dist": base_avg})

    Y_ref = ds_train.Y

    # Global Fréchet Regression (GFR)
    print(f"  Running GFR...")
    t_gfr = time.time()
    gfr = GlobalFrechetRegression(dist_name=metric_name)
    gfr.fit(ds_train.X, ds_train.Y)

    gfr_dists = []
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_pred_gfr = gfr.predict(X_batch).to(device)
        d_per_response = []
        for v in range(n_responses):
            d_v = dist_fn(Y_pred_gfr[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_gfr = torch.stack(d_per_response).mean(dim=0)
        gfr_dists.append(d_gfr)
    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_time = time.time() - t_gfr

    print(f"  GFR: {gfr_avg:.6f} (time: {gfr_time:.1f}s)")
    results.update({
        "gfr_avg_dist": gfr_avg,
        "gfr_time_sec": gfr_time
    })

    # Deep Fréchet Regression (DFR)
    print(f"  Running DFR...")
    t_dfr = time.time()

    # Use tuned parameters if available, otherwise defaults
    dfr_params = {
        "manifold_dim": 2,
        "hidden": 32,
        "layer": 4,
        "num_epochs": 200,
        "lr": 5e-4,
        "dropout": 0.0
    }
    if dfr_tuned_params:
        dfr_params.update(dfr_tuned_params)

    dfr = DeepFrechetRegression(
        dist_name=metric_name,
        manifold_method="isomap",
        manifold_dim=dfr_params["manifold_dim"],
        manifold_k=10,
        hidden=dfr_params["hidden"],
        layer=dfr_params["layer"],
        num_epochs=dfr_params["num_epochs"],
        lr=dfr_params["lr"],
        dropout=dfr_params["dropout"],
        seed=seed,
    )
    dfr.fit(ds_train.X, ds_train.Y, verbose=False)

    dfr_dists = []
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_pred_dfr = dfr.predict(X_batch).to(device)
        d_per_response = []
        for v in range(n_responses):
            d_v = dist_fn(Y_pred_dfr[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_dfr = torch.stack(d_per_response).mean(dim=0)
        dfr_dists.append(d_dfr)
    dfr_dists_cat = torch.cat(dfr_dists)
    dfr_avg = dfr_dists_cat.mean().item()
    dfr_time = time.time() - t_dfr

    print(f"  DFR: {dfr_avg:.6f} (time: {dfr_time:.1f}s)")
    results.update({
        "dfr_avg_dist": dfr_avg,
        "dfr_time_sec": dfr_time
    })

    # FDRNN (with adaptive LoRA)
    print(f"  Running FDRNN (adaptive LoRA)...")
    set_seed(seed)

    # Use tuned parameters if available, otherwise defaults
    fdrnn_params = {
        "lr": lr,
        "entropy_reg": entropy_reg,
        "nuclear_reg": 0.01,
        "reduction_dim": min(p, 8),  # Default reduction
        "response_rank": n_responses,
        "response_alpha": 1.0,
        "reduction_type": "nonlinear",
        "dropout": 0.0,
        "encoder_sizes": [128, 64],
        "head_sizes": [64, 128]
    }
    if fdrnn_tuned_params:
        fdrnn_params.update(fdrnn_tuned_params)

    model_fdrnn = FrechetDRNN(
        input_dim=p,
        n_ref=n_train,
        reduction_dim=fdrnn_params["reduction_dim"],
        n_responses=n_responses,
        response_rank=fdrnn_params["response_rank"],
        response_alpha=fdrnn_params["response_alpha"],
        reduction_type=fdrnn_params["reduction_type"],
        encoder_sizes=fdrnn_params["encoder_sizes"],
        head_sizes=fdrnn_params["head_sizes"],
        activation="relu",
        dropout=fdrnn_params["dropout"],
    )

    t_fdrnn = time.time()
    history_fdrnn = train_frechet_model(
        model=model_fdrnn,
        Y_ref=Y_ref,
        train_loader=train_loader,
        dist_name=metric_name,
        epochs=epochs,
        lr=fdrnn_params["lr"],
        weight_decay=1e-5,
        entropy_reg=fdrnn_params["entropy_reg"],
        nuclear_reg=fdrnn_params["nuclear_reg"],
        device=device,
        verbose=False,
        val_loader=es_val_loader,
        patience=15,
    )
    fdrnn_time = time.time() - t_fdrnn

    fdrnn_results = evaluate_frechet_model_safe(
        model_fdrnn, Y_ref, test_loader, metric_name, device)

    fdrnn_avg = fdrnn_results["avg_dist"]

    print(f"  FDRNN (adaptive LoRA): {fdrnn_avg:.6f} (time: {fdrnn_time:.1f}s)")
    results.update({
        "fdrnn_avg_dist": fdrnn_avg,
        "fdrnn_time_sec": fdrnn_time,
        "fdrnn_reduction_dim": fdrnn_params["reduction_dim"],
        "fdrnn_response_rank": fdrnn_params["response_rank"]
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Deep Fréchet Regression Simulation Setups")
    parser.add_argument(
        "--setup",
        type=str,
        default="all",
        choices=["setup1", "setup2", "setup3", "setup4", "setup5", "all"],
        help="Simulation setup to run (default: all)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="power",
        choices=list(DISTANCE_FUNCTIONS.keys()) + ["all"],
        help="Distance metric (default: power)"
    )
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--df", type=int, default=6, help="Wishart degrees of freedom")
    parser.add_argument("--n_responses", type=int, default=1, help="Number of responses")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=15, help="Number of repetitions")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--output_file", type=str, default="simulation_results_adaptive_lora.csv")

    args = parser.parse_args()

    # Setup functions
    setups = {
        "setup1": setup_1_basic,
        "setup2": setup_2_multi_response,
        "setup3": setup_3_low_df,
        "setup4": setup_4_high_df,
        "setup5": setup_5_large_scale,
    }

    # Setup parameters
    setup_params = {
        "setup1": {"n_responses": 1, "df": 6},
        "setup2": {"n_responses": 3, "df": 6},
        "setup3": {"n_responses": 1, "df": 6},
        "setup4": {"n_responses": 1, "df": 12},
        "setup5": {"n_train": 500, "n_test": 200, "n_responses": 1, "df": 6},
    }

    # Metrics to test
    if args.metric == "all":
        metrics = ["frobenius", "affine_invariant", "power"]
    else:
        metrics = [args.metric]

    # Setups to run
    if args.setup == "all":
        selected_setups = list(setups.keys())
    else:
        selected_setups = [args.setup]

    print(f"Running Deep Fréchet Regression simulations with adaptive LoRA")
    print(f"Setups: {selected_setups}")
    print(f"Metrics: {metrics}")
    print(f"Repetitions: {args.n_reps}")
    print(f"Output: {args.output_file}")
    print()

    all_results = []

    for setup_name in selected_setups:
        setup_func = setups[setup_name]
        params = setup_params[setup_name]

        for metric in metrics:
            print(f"Running {setup_name} with {metric} metric...")

            # Hyperparameter tuning if requested
            dfr_tuned = None
            fdrnn_tuned = None
            if args.tune:
                print(f"  Tuning hyperparameters for {setup_name}...")
                # Note: Tuning implementation would go here
                pass

            for rep in range(args.n_reps):
                seed = args.seed + rep * 100

                result = run_single_experiment(
                    setup_func=setup_func,
                    metric_name=metric,
                    n_train=params.get("n_train", args.n_train),
                    n_test=params.get("n_test", args.n_test),
                    df=params.get("df", args.df),
                    n_responses=params.get("n_responses", args.n_responses),
                    epochs=args.epochs,
                    lr=args.lr,
                    entropy_reg=args.entropy_reg,
                    seed=seed,
                    device=args.device,
                    dfr_tuned_params=dfr_tuned,
                    fdrnn_tuned_params=fdrnn_tuned,
                )

                result["repetition"] = rep
                all_results.append(result)

                print(f"  Rep {rep+1}/{args.n_reps}: Complete")
                print()

    # Save results
    if all_results:
        import csv
        fieldnames = sorted(all_results[0].keys())

        with open(args.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)

        print(f"\nResults saved to {args.output_file}")
        print("Simulation complete!")

        # Compute and print averages with SE
        import numpy as np
        from collections import defaultdict

        # Group results by setup and metric
        grouped = defaultdict(lambda: defaultdict(list))
        for result in all_results:
            key = (result["setup"], result["metric"])
            for k, v in result.items():
                if k.endswith("_avg_dist") and isinstance(v, (int, float)):
                    grouped[key][k].append(v)

        for (setup, metric), methods in grouped.items():
            print(f"\n{setup} with {metric} metric:")
            for method_key, values in methods.items():
                if values:
                    mean = np.mean(values)
                    se = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                    method_name = method_key.replace("_avg_dist", "").upper()
                    print(f"  {method_name}: {mean:.6f} ± {se:.6f}")


if __name__ == "__main__":
    main()