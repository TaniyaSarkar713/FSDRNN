#!/usr/bin/env python3
"""
Simulation setups from "Deep Fréchet Regression" paper (Petersen & Müller, 2025)
Comparison of Fréchet regression methods on SPD manifolds.

This script implements the simulation studies from the paper, comparing:
- Global Mean (baseline)
- Global Fréchet Regression (GFR)
- Deep Fréchet Regression (DFR)
- Fréchet Deep Neural Networks (FDRNN variants)

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


class SimulationSetup:
    """Container for different simulation setups from the paper."""

    @staticmethod
    def setup_1_basic(n_train: int = 200, n_test: int = 100, df: int = 6,
                     n_responses: int = 1, seed: int = 42) -> Tuple[WishartSPDDataset, WishartSPDDataset]:
        """
        Setup 1: Basic Wishart SPD regression
        - Single response (V=1)
        - Standard Wishart with df=6
        - 12-dimensional predictors with mixed distributions
        """
        ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
        ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
        return ds_train, ds_test

    @staticmethod
    def setup_2_multi_response(n_train: int = 200, n_test: int = 100, df: int = 6,
                              n_responses: int = 3, seed: int = 42) -> Tuple[WishartSPDDataset, WishartSPDDataset]:
        """
        Setup 2: Multi-response Wishart SPD regression
        - Multiple responses (V=3)
        - Independent Wishart matrices with shared predictors
        """
        ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
        ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
        return ds_train, ds_test

    @staticmethod
    def setup_3_low_df(n_train: int = 200, n_test: int = 100, df: int = 3,
                      n_responses: int = 1, seed: int = 42) -> Tuple[WishartSPDDataset, WishartSPDDataset]:
        """
        Setup 3: Low degrees of freedom
        - Single response
        - Wishart with df=3 (more variable SPD matrices)
        """
        ds_train = WishartSPDDataset(n=n_train, p=3, n_responses=n_responses, df=df, seed=seed)
        ds_test = WishartSPDDataset(n=n_test, p=3, n_responses=n_responses, df=df, seed=seed + 1000)
        return ds_train, ds_test

    @staticmethod
    def setup_4_high_df(n_train: int = 200, n_test: int = 100, df: int = 12,
                       n_responses: int = 1, seed: int = 42) -> Tuple[WishartSPDDataset, WishartSPDDataset]:
        """
        Setup 4: High degrees of freedom
        - Single response
        - Wishart with df=12 (less variable, more concentrated SPD matrices)
        """
        ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
        ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
        return ds_train, ds_test

    @staticmethod
    def setup_5_large_scale(n_train: int = 500, n_test: int = 200, df: int = 6,
                           n_responses: int = 1, seed: int = 42) -> Tuple[WishartSPDDataset, WishartSPDDataset]:
        """
        Setup 5: Large scale simulation
        - Larger sample sizes for more stable estimates
        - Single response with standard parameters
        """
        ds_train = WishartSPDDataset(n=n_train, n_responses=n_responses, df=df, seed=seed)
        ds_test = WishartSPDDataset(n=n_test, n_responses=n_responses, df=df, seed=seed + 1000)
        return ds_train, ds_test


def run_single_experiment(
    setup_name: str,
    setup_func,
    metric_name: str,
    n_train: int = 200,
    n_test: int = 100,
    df: int = 6,
    n_responses: int = 1,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    entropy_reg: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    tune: bool = False,
    dfr_tuned_params: Optional[Dict] = None,
    fdrnn_tuned_params: Optional[Dict] = None,
    structural_dim: int = 5,
) -> Dict:
    """
    Run a single experiment with the specified setup and metric.

    Args:
        setup_name: Name of the simulation setup
        setup_func: Function to create the dataset
        metric_name: Distance metric name
        Other args: Standard training parameters

    Returns:
        Dictionary with results for all methods
    """
    p = 12  # input dimension

    set_seed(seed)

    print(f"\n{'='*80}")
    print(f"  Setup: {setup_name} | Metric: {metric_name}")
    print(f"  n_train={n_train}, n_test={n_test}, df={df}, n_responses={n_responses}")
    print(f"  epochs={epochs}, seed={seed}")
    print(f"{'='*80}")

    # Generate data
    ds_train, ds_test = setup_func(n_train=n_train, n_test=n_test, df=df,
                                   n_responses=n_responses, seed=seed)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Reference set and validation split
    Y_ref = ds_train.Y
    val_n = max(1, int(0.2 * len(ds_train)))
    es_train_ds, es_val_ds = random_split(
        ds_train, [len(ds_train) - val_n, val_n],
        generator=torch.Generator().manual_seed(seed),
    )
    es_val_loader = DataLoader(es_val_ds, batch_size=batch_size, shuffle=False)

    dist_fn = get_distance_fn(metric_name)

    # Global mean baseline
    global_mean = ds_train.Y.mean(dim=0)
    baseline_dists = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_batch = Y_batch.to(device)
            B = X_batch.size(0)
            gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1).to(device)
            d_per_response = []
            for v in range(n_responses):
                d_v = dist_fn(gm[:, v, :, :], Y_batch[:, v, :, :])
                d_per_response.append(d_v)
            d_base = torch.stack(d_per_response).mean(dim=0)
            baseline_dists.append(d_base.cpu())
    baseline_dists_cat = torch.cat(baseline_dists)
    base_avg = baseline_dists_cat.mean().item()

    results = {
        "setup": setup_name,
        "metric": metric_name,
        "baseline_avg_dist": base_avg,
        "baseline_time_sec": 0.0
    }

    print(f"  Global Mean: {base_avg:.6f}")

    # Global Fréchet Regression (GFR)
    print(f"  Running GFR...")
    t_gfr = time.time()
    gfr = GlobalFrechetRegression(dist_name=metric_name)
    gfr.fit(ds_train.X, ds_train.Y)

    gfr_dists = []
    for X_batch, Y_batch in test_loader:
        Y_pred_gfr = gfr.predict(X_batch)
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
        Y_pred_dfr = dfr.predict(X_batch)
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

    # FDRNN (with dimension reduction)
    print(f"  Running FDRNN...")
    set_seed(seed)

    # Use tuned parameters if available, otherwise defaults
    fdrnn_params = {
        "lr": lr,
        "entropy_reg": entropy_reg,
        "nuclear_reg": 0.01,
        "reduction_dim": min(p, 8),  # Default reduction
        "response_rank": n_responses,
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
        response_alpha=1.0,
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

    fdrnn_results = evaluate_frechet_model(
        model_fdrnn, Y_ref, test_loader, metric_name, device)

    fdrnn_avg = fdrnn_results["avg_dist"]

    print(f"  FDRNN: {fdrnn_avg:.6f} (time: {fdrnn_time:.1f}s)")
    results.update({
        "fdrnn_avg_dist": fdrnn_avg,
        "fdrnn_time_sec": fdrnn_time,
        "fdrnn_reduction_dim": fdrnn_params["reduction_dim"]
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
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--df", type=int, default=6, help="Wishart degrees of freedom")
    parser.add_argument("--n_responses", type=int, default=1, help="Number of responses")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--output_file", type=str, default="simulation_results.csv")

    args = parser.parse_args()

    # Define setups
    setups = {
        "setup1": ("Basic SPD Regression", SimulationSetup.setup_1_basic),
        "setup2": ("Multi-Response SPD Regression", SimulationSetup.setup_2_multi_response),
        "setup3": ("Low DF SPD Regression", SimulationSetup.setup_3_low_df),
        "setup4": ("High DF SPD Regression", SimulationSetup.setup_4_high_df),
        "setup5": ("Large Scale SPD Regression", SimulationSetup.setup_5_large_scale),
    }

    # Define metrics
    if args.metric == "all":
        metrics = ["power", "frobenius", "affine_invariant"]
    else:
        metrics = [args.metric]

    # Select setups to run
    if args.setup == "all":
        selected_setups = list(setups.keys())
    else:
        selected_setups = [args.setup]

    # Adjust parameters based on setup
    setup_params = {
        "setup1": {"n_responses": 1, "df": 6},
        "setup2": {"n_responses": 3, "df": 6},
        "setup3": {"n_responses": 1, "df": 3},
        "setup4": {"n_responses": 1, "df": 12},
        "setup5": {"n_responses": 1, "df": 6, "n_train": 500, "n_test": 200},
    }

    all_results = []

    for setup_name in selected_setups:
        setup_desc, setup_func = setups[setup_name]
        params = setup_params.get(setup_name, {})

        # Override default parameters
        n_train = params.get("n_train", args.n_train)
        n_test = params.get("n_test", args.n_test)
        df = params.get("df", args.df)
        n_responses = params.get("n_responses", args.n_responses)

        print(f"\n{'#'*100}")
        print(f"Running {setup_desc} ({setup_name})")
        print(f"Parameters: n_train={n_train}, n_test={n_test}, df={df}, n_responses={n_responses}")
        print(f"{'#'*100}")

        for metric in metrics:
            print(f"\n--- Metric: {metric} ---")

            # Perform tuning if requested
            dfr_tuned_params = None
            fdrnn_tuned_params = None

            if args.tune:
                print("Performing hyperparameter tuning...")

                # Create tuning dataset
                tune_ds, _ = setup_func(n_train=n_train, n_test=n_test, df=df,
                                       n_responses=n_responses, seed=0)

                # Tune DFR
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
                    dist_name=metric,
                    param_grid=dfr_grid,
                    val_frac=0.2,
                    batch_size=args.batch_size,
                    seed=0,
                    verbose=False,
                )
                dfr_tuned_params = dfr_result["best_params"]

                # Tune FDRNN (simplified for speed)
                fdrnn_grid = {
                    "lr": [5e-4, 1e-3],
                    "entropy_reg": [0.0, 0.001],
                    "nuclear_reg": [0.0],
                    "reduction_dim": [2, 5, 8],
                    "reduction_type": ["linear", "nonlinear"],
                    "dropout": [0.0],
                }
                fdrnn_result = grid_search_frechet(
                    dataset=tune_ds,
                    parent_Y=tune_ds.Y,
                    model_class=FrechetDRNN,
                    dist_name=metric,
                    param_grid=fdrnn_grid,
                    fixed_model_kwargs={
                        "input_dim": 12,
                        "n_responses": n_responses,
                        "encoder_sizes": [128, 64],
                        "head_sizes": [64, 128],
                    },
                    fixed_train_kwargs={"epochs": 50},  # Reduced for tuning
                    val_frac=0.2,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=0,
                    verbose=False,
                )
                fdrnn_tuned_params = fdrnn_result["best_params"]

            # Run multiple repetitions
            rep_results = []
            for rep in range(args.n_reps):
                rep_seed = args.seed + rep
                print(f"  Rep {rep + 1}/{args.n_reps} (seed={rep_seed})...", end="", flush=True)

                result = run_single_experiment(
                    setup_name=setup_name,
                    setup_func=setup_func,
                    metric_name=metric,
                    n_train=n_train,
                    n_test=n_test,
                    df=df,
                    n_responses=n_responses,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    entropy_reg=args.entropy_reg,
                    seed=rep_seed,
                    device=args.device,
                    tune=False,  # Tuning already done
                    dfr_tuned_params=dfr_tuned_params,
                    fdrnn_tuned_params=fdrnn_tuned_params,
                    structural_dim=5,
                )
                rep_results.append(result)
                print(" Done")

            # Aggregate results
            methods = ["baseline", "gfr", "dfr", "fdrnn"]
            for method in methods:
                dist_key = f"{method}_avg_dist"
                time_key = f"{method}_time_sec"

                dists = [r[dist_key] for r in rep_results]
                times = [r[time_key] for r in rep_results]

                mean_dist = np.mean(dists)
                std_dist = np.std(dists, ddof=1) if args.n_reps > 1 else 0
                mean_time = np.mean(times)

                # Store aggregated result
                agg_result = {
                    "setup": setup_name,
                    "metric": metric,
                    "method": method,
                    "mean_dist": mean_dist,
                    "std_dist": std_dist,
                    "mean_time": mean_time,
                    "n_reps": args.n_reps,
                    "n_train": n_train,
                    "n_test": n_test,
                    "df": df,
                    "n_responses": n_responses,
                }
                all_results.append(agg_result)

                print(f"  {method.upper()}: {mean_dist:.6f} ± {std_dist:.6f} (time: {mean_time:.1f}s)")

    # Save results to CSV
    import csv
    with open(args.output_file, 'w', newline='') as csvfile:
        fieldnames = ["setup", "metric", "method", "mean_dist", "std_dist", "mean_time",
                     "n_reps", "n_train", "n_test", "df", "n_responses"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\nResults saved to {args.output_file}")
    print("Simulation complete!")

if __name__ == "__main__":
    main()
