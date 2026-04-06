#!/usr/bin/env python3
"""
Test case: SPD responses with shared latent factors.

Setup:
  - X in R^p, p=20, X ~ N(0, Σ_X), Σ_X_{jk} = ρ_X^{|j-k|}, ρ_X=0.3
  - B0 in R^{p x 2}, B0^T B0 = I_2, Z = B0^T X
  - g(Z) = [Z1, Z2^2, sin(π Z1 Z2)]^T
  - Y_v in S++_m, m=4, V=6
  - S_v(X) = S0 + sum_k a_vk g_k(Z) U_k + delta_v(Z) R_v
  - Sigma_v(X) = exp(S_v(X))
  - Y_v = Sigma_v(X) + E_v, with small symmetric E_v

Cases:
  1. Linear: g = [Z1, Z2, 0]^T
  2. Nonlinear: g = [Z1, Z2^2, sin(π Z1 Z2)]^T
  3. Nonlinear + heterogeneous: + delta_v
  4. Nonlinear + heterogeneous + low-rank: low-rank A

Methods compared:
  1. Global Mean
  2. GFR
  3. DFR
  4. FDRNN (adaptive LoRA)
"""

import sys
import os
import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
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
    DeepFrechetRegression,
    grid_search_frechet,
    grid_search_dfr,
)

from src.spd_frechet import GlobalFrechetRegression


class SharedFactorsSPDDataset(torch.utils.data.Dataset):
    """Dataset for SPD responses with shared latent factors."""

    def __init__(self, n: int, case: str = "NL", seed: int = 42, p: int = 20, m: int = 4, V: int = 6, B0=None, U=None, S0=None, A=None, R=None):
        """
        Generate SPD data with shared factors.

        Args:
            n: number of samples
            case: "L" for linear, "NL" for nonlinear, "NLH" for nonlinear+heterogeneous, "NLR" for low-rank
            seed: random seed
            p: input dimension
            m: SPD matrix size
            V: number of responses
            B0: optional pre-generated B0 [p, 2]
            U: optional pre-generated U list of [m, m]
            S0: optional pre-generated S0 [m, m]
            A: optional pre-generated A [V, 3]
            R: optional pre-generated R list of [m, m]
        """
        self.n = n
        self.case = case
        self.seed = seed
        self.p = p
        self.m = m
        self.V = V

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Parameters
        rho_X = 0.3
        nu = 15  # for Wishart, but we'll use additive noise

        # Σ_X
        Sigma_X = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma_X[i, j] = rho_X ** abs(i - j)
        Sigma_X = torch.tensor(Sigma_X, dtype=torch.float32)

        # X ~ N(0, Σ_X)
        L = torch.linalg.cholesky(Sigma_X)
        X_normal = torch.randn(n, p)
        self.X = (L @ X_normal.T).T  # [n, p]

        # B0: p x 2, orthogonal
        if B0 is None:
            B0 = torch.randn(p, 2)
            B0, _ = torch.linalg.qr(B0)  # [p, 2]
        self.B0 = B0

        # Z = B0^T X
        self.Z = self.X @ B0  # [n, 2]

        # g(Z)
        Z1 = self.Z[:, 0]
        Z2 = self.Z[:, 1]
        if case in ["L", "NLH", "NLR"]:
            g1 = Z1
            g2 = Z2
            g3 = torch.zeros_like(Z1)
        else:  # NL
            g1 = Z1
            g2 = Z2 ** 2
            g3 = torch.sin(math.pi * Z1 * Z2)
        self.g = torch.stack([g1, g2, g3], dim=1)  # [n, 3]

        # Basis matrices U1, U2, U3: m x m symmetric
        if U is None:
            U1 = torch.tensor([
                [1.0, 0.3, 0.0, 0.0],
                [0.3, 0.0, 0.2, 0.0],
                [0.0, 0.2, -1.0, 0.1],
                [0.0, 0.0, 0.1, 0.0]
            ], dtype=torch.float32)
            U2 = torch.tensor([
                [0.0, 0.2, 0.1, 0.0],
                [0.2, 1.0, 0.0, 0.2],
                [0.1, 0.0, 0.0, 0.3],
                [0.0, 0.2, 0.3, -1.0]
            ], dtype=torch.float32)
            U3 = torch.tensor([
                [0.5, 0.1, 0.0, 0.2],
                [0.1, 0.0, 0.3, 0.0],
                [0.0, 0.3, 1.0, 0.1],
                [0.2, 0.0, 0.1, 0.5]
            ], dtype=torch.float32)
            U = [U1, U2, U3]
        self.U = U

        # S0: baseline m x m symmetric
        if S0 is None:
            S0 = torch.eye(m, dtype=torch.float32) * 0.5
        self.S0 = S0

        # A: V x 3 loadings
        if A is None:
            if case == "NLR":
                # Low-rank A
                A = torch.tensor([
                    [1.0, 0.8, 0.0],
                    [0.9, 0.7, 0.2],
                    [0.8, 0.1, 1.0],
                    [0.7, 0.2, 0.9],
                    [1.1, -0.4, 0.3],
                    [0.9, -0.3, 0.5]
                ], dtype=torch.float32)
            else:
                A = torch.tensor([
                    [1.0, 0.8, 0.0],
                    [0.9, 0.7, 0.2],
                    [0.8, 0.1, 1.0],
                    [0.7, 0.2, 0.9],
                    [1.1, -0.4, 0.3],
                    [0.9, -0.3, 0.5]
                ], dtype=torch.float32)
        self.A = A

        # R_v: response-specific basis, m x m symmetric
        if R is None:
            R = []
            for v in range(V):
                R_v = torch.randn(m, m)
                R_v = (R_v + R_v.T) / 2  # symmetric
                R.append(R_v)
        self.R = R

        # delta_v(Z)
        if case in ["NLH", "NLR"]:
            c_v = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.12, 0.18])
            b_v = torch.tensor([0.0, 0.05, 0.0, 0.05, 0.0, 0.05])
            omega_v = torch.tensor([1.0, 1.5, 2.0, 0.5, 1.2, 0.8])
            self.delta = torch.zeros(n, V)
            for v in range(V):
                self.delta[:, v] = c_v[v] * torch.sin(omega_v[v] * Z1) + b_v[v] * Z2
        else:
            self.delta = torch.zeros(n, V)

        # Generate Y_v
        self.Y = []
        for v in range(V):
            S_v = torch.zeros(n, m, m)
            S_v += self.S0.clone()
            for k in range(3):
                S_v += self.A[v, k] * self.g[:, k].unsqueeze(-1).unsqueeze(-1) * self.U[k]
            S_v += self.delta[:, v].unsqueeze(-1).unsqueeze(-1) * self.R[v]
            Sigma_v = torch.matrix_exp(S_v)  # [n, m, m]

            # Y_v = Sigma_v + E_v
            E_v = torch.randn(n, m, m) * 0.1
            E_v = (E_v + E_v.transpose(-2, -1)) / 2  # symmetric
            Y_v = Sigma_v + E_v

            # Ensure SPD by adding small diagonal
            Y_v = Y_v + torch.eye(m).unsqueeze(0) * 1e-6

            self.Y.append(Y_v)  # list of [n, m, m]

        # Stack Y: [n, V, m, m]
        self.Y = torch.stack(self.Y, dim=1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_true_mean(self, idx):
        """Get true mean for sample idx (approximate)."""
        return self.Y[idx]


def set_seed(seed: int):
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
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
):
    """
    Run the full train + evaluate pipeline for one case.

    Methods: Global Mean, GFR, DFR, FDRNN.
    """
    p = 20
    m = 4
    V = 6

    set_seed(seed)

    # Generate shared structural parameters
    np.random.seed(seed)
    torch.manual_seed(seed)

    # B0: p x 2, orthogonal
    B0 = torch.randn(p, 2)
    B0, _ = torch.linalg.qr(B0)  # [p, 2]

    # Basis matrices U1, U2, U3: m x m symmetric
    U1 = torch.tensor([
        [1.0, 0.3, 0.0, 0.0],
        [0.3, 0.0, 0.2, 0.0],
        [0.0, 0.2, -1.0, 0.1],
        [0.0, 0.0, 0.1, 0.0]
    ], dtype=torch.float32)
    U2 = torch.tensor([
        [0.0, 0.2, 0.1, 0.0],
        [0.2, 1.0, 0.0, 0.2],
        [0.1, 0.0, 0.0, 0.3],
        [0.0, 0.2, 0.3, -1.0]
    ], dtype=torch.float32)
    U3 = torch.tensor([
        [0.5, 0.1, 0.0, 0.2],
        [0.1, 0.0, 0.3, 0.0],
        [0.0, 0.3, 1.0, 0.1],
        [0.2, 0.0, 0.1, 0.5]
    ], dtype=torch.float32)
    U = [U1, U2, U3]

    # S0: baseline m x m symmetric
    S0 = torch.eye(m, dtype=torch.float32) * 0.5

    # A: V x 3 loadings
    if case == "NLR":
        # Low-rank A
        A = torch.tensor([
            [1.0, 0.8, 0.0],
            [0.9, 0.7, 0.2],
            [0.8, 0.1, 1.0],
            [0.7, 0.2, 0.9],
            [1.1, -0.4, 0.3],
            [0.9, -0.3, 0.5]
        ], dtype=torch.float32)
    else:
        A = torch.tensor([
            [1.0, 0.8, 0.0],
            [0.9, 0.7, 0.2],
            [0.8, 0.1, 1.0],
            [0.7, 0.2, 0.9],
            [1.1, -0.4, 0.3],
            [0.9, -0.3, 0.5]
        ], dtype=torch.float32)

    # R_v: response-specific basis, m x m symmetric
    R = []
    for v in range(V):
        R_v = torch.randn(m, m)
        R_v = (R_v + R_v.T) / 2  # symmetric
        R.append(R_v)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Case: {case}")
        print(f"  n_train={n_train}, n_test={n_test}, seed={seed}")
        print(f"  p={p}, m={m}, V={V}")
        if fdrnn_tuned_params:
            print(f"  [TUNED FDRNN params: {fdrnn_tuned_params}]")
        print(f"{'='*70}")

    ds_train = SharedFactorsSPDDataset(n=n_train, case=case, seed=seed, p=p, m=m, V=V, B0=B0, U=U, S0=S0, A=A, R=R)
    ds_test = SharedFactorsSPDDataset(n=n_test, case=case, seed=seed + 1000, p=p, m=m, V=V, B0=B0, U=U, S0=S0, A=A, R=R)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    if verbose:
        X_sample, Y_sample = ds_train[0]
        print(f"\n  Input shape:    X ~ {tuple(X_sample.shape)}")
        print(f"  Response shape: Y ~ {tuple(Y_sample.shape)}")

    # Reference set
    Y_ref = ds_train.Y  # [n_train, V, m, m]

    # Early-stopping validation split (20%)
    val_n = max(1, int(0.2 * len(ds_train)))
    es_train_ds, es_val_ds = random_split(
        ds_train, [len(ds_train) - val_n, val_n],
        generator=torch.Generator().manual_seed(seed),
    )
    es_val_loader = DataLoader(es_val_ds, batch_size=batch_size, shuffle=False)

    # Distance metric
    dist_name = "frobenius"

    # Global mean baseline
    global_mean = ds_train.Y.mean(dim=0)  # [V, m, m]

    baseline_dists = []
    dist_fn = get_distance_fn(dist_name)
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_batch = Y_batch.to(device)
            B = X_batch.size(0)
            gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1).to(device)  # [B, V, m, m]
            d = dist_fn(gm, Y_batch).mean(dim=1)  # average over responses
            baseline_dists.append(d.cpu())

    baseline_dists_cat = torch.cat(baseline_dists)
    base_avg = baseline_dists_cat.mean().item()

    results = {}
    results["case"] = case
    results["baseline_avg_dist"] = base_avg
    results["baseline_time_sec"] = 0.0

    if verbose:
        print(f"\n  --- Competitors ---")
        print(f"  Global Mean avg dist:   {base_avg:.6f}")

    # GFR: Global linear regression
    if verbose:
        print(f"\n  --- Global Linear Regression (GFR) ---")
    t_gfr = time.time()
    gfr = GlobalFrechetRegression(dist_name=dist_name)
    gfr.fit(ds_train.X, ds_train.Y)

    gfr_dists = []
    for X_batch, Y_batch in test_loader:
        Y_pred_gfr = gfr.predict(X_batch)
        d_per_response = []
        for v in range(V):
            d_v = dist_fn(Y_pred_gfr[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_gfr = torch.stack(d_per_response).mean(dim=0)
        gfr_dists.append(d_gfr)
    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_time = time.time() - t_gfr

    results["gfr_avg_dist"] = gfr_avg
    results["gfr_time_sec"] = gfr_time

    if verbose:
        print(f"  GFR avg dist:   {gfr_avg:.6f} (time: {gfr_time:.1f}s)")

    # DFR: Deep regression
    if verbose:
        print(f"\n  --- Deep Regression (DFR) ---")
    t_dfr = time.time()
    dfr = DeepFrechetRegression(
        dist_name=dist_name,
        manifold_method="isomap",
        manifold_dim=2,
        manifold_k=10,
        hidden=32,
        layer=4,
        num_epochs=200,
        lr=lr,
        dropout=0.0,
        seed=seed,
    )
    dfr.fit(ds_train.X, ds_train.Y, verbose=verbose)

    dfr_dists = []
    for X_batch, Y_batch in test_loader:
        Y_pred_dfr = dfr.predict(X_batch)
        d_per_response = []
        for v in range(V):
            d_v = dist_fn(Y_pred_dfr[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d_dfr = torch.stack(d_per_response).mean(dim=0)
        dfr_dists.append(d_dfr)
    dfr_dists_cat = torch.cat(dfr_dists)
    dfr_avg = dfr_dists_cat.mean().item()
    dfr_time = time.time() - t_dfr

    results["dfr_avg_dist"] = dfr_avg
    results["dfr_time_sec"] = dfr_time

    if verbose:
        print(f"  DFR avg dist:   {dfr_avg:.6f} (time: {dfr_time:.1f}s)")

    # FDRNN
    if verbose:
        print(f"\n  --- FDRNN (adaptive LoRA) ---")
    set_seed(seed)
    _fdrnn_lr = fdrnn_tuned_params.get("lr", lr) if fdrnn_tuned_params else lr
    _fdrnn_er = fdrnn_tuned_params.get("entropy_reg", entropy_reg) if fdrnn_tuned_params else entropy_reg
    _fdrnn_rd = fdrnn_tuned_params.get("reduction_dim", p) if fdrnn_tuned_params else p
    _fdrnn_rr = fdrnn_tuned_params.get("response_rank", min(5, V)) if fdrnn_tuned_params else min(5, V)

    model_fdrnn = FrechetDRNN(
        input_dim=p,
        n_ref=n_train,
        reduction_dim=_fdrnn_rd,
        n_responses=V,
        response_rank=_fdrnn_rr,
        response_alpha=1.0,
        reduction_type="nonlinear",
        encoder_sizes=[128, 64],
        head_sizes=[64, 128],
        activation="relu",
        dropout=0.0,
    ).to(device)

    t_fdrnn = time.time()
    history_fdrnn = train_frechet_model(
        model=model_fdrnn,
        Y_ref=Y_ref,
        train_loader=train_loader,
        dist_name=dist_name,
        epochs=epochs,
        lr=_fdrnn_lr,
        weight_decay=1e-5,
        entropy_reg=_fdrnn_er,
        nuclear_reg=0.01,
        device=device,
        verbose=verbose,
        val_loader=es_val_loader,
        patience=15,
    )
    fdrnn_time = time.time() - t_fdrnn

    fdrnn_results = evaluate_frechet_model(
        model_fdrnn, Y_ref, test_loader, dist_name, device)

    fdrnn_avg = fdrnn_results["avg_dist"]

    results["fdrnn_avg_dist"] = fdrnn_avg
    results["fdrnn_time_sec"] = fdrnn_time

    if verbose:
        print(f"  FDRNN avg dist: {fdrnn_avg:.6f} (time: {fdrnn_time:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test: SPD shared factors"
    )
    parser.add_argument(
        "--case",
        type=str,
        default="NL",
        choices=["L", "NL", "NLH", "NLR"],
        help="Case: L=linear, NL=nonlinear, NLH=nonlinear+heterogeneous, NLR=low-rank",
    )
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--output_file", type=str, default="spd_shared_factors.csv")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    n_reps = args.n_reps

    # Main experiment
    print(f"\n{'#'*70}")
    print(f"  SPD SHARED FACTORS: {args.case}  |  {n_reps} repetitions")
    print(f"{'#'*70}")

    # Collect results
    methods = [
        ("Global Mean", "baseline"),
        ("GFR", "gfr"),
        ("DFR", "dfr"),
        ("FDRNN (adaptive LoRA)", "fdrnn"),
    ]
    collect_suffixes = ["_avg_dist", "_time_sec"]
    all_vals = {}
    for _, prefix in methods:
        for suf in collect_suffixes:
            all_vals[prefix + suf] = []

    # Hyperparameter tuning (default, can be skipped with --no-tune)
    dfr_tuned_params = None
    fdrnn_tuned_params = None
    if not args.no_tune:
        print("\n  --- Hyperparameter Tuning ---")
        # Create a small dataset for tuning
        ds_tune = SharedFactorsSPDDataset(n=100, case=args.case, seed=args.seed, p=20, m=4, V=6)
        Y_ref_tune = ds_tune.Y

        # Tune DFR
        print("  Tuning DFR...")
        dfr_param_grid = {
            "manifold_dim": [1, 2],
            "hidden": [16, 32],
            "layer": [3, 4],
            "lr": [5e-4, 1e-3],
        }
        dfr_tuned_params = grid_search_dfr(
            dataset=ds_tune,
            parent_X=ds_tune.X,
            parent_Y=Y_ref_tune,
            dist_name="frobenius",
            param_grid=dfr_param_grid,
            val_frac=0.2,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=True,
        )
        print(f"  DFR tuned params: {dfr_tuned_params}")

        # Tune FDRNN with adaptive LoRA
        print("  Tuning FDRNN with adaptive LoRA...")
        adaptive_r = min(5, 6)  # V=6
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
                dataset=ds_tune,
                parent_Y=Y_ref_tune,
                model_class=FrechetDRNN,
                dist_name="frobenius",
                param_grid=fdrnn_stage_a_grid,
                fixed_model_kwargs={
                    "input_dim": 20,
                    "n_responses": 6,
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
            dataset=ds_tune,
            parent_Y=Y_ref_tune,
            model_class=FrechetDRNN,
            dist_name="frobenius",
            param_grid=fdrnn_stage_b_grid,
            fixed_model_kwargs={
                "input_dim": 20,
                "n_responses": 6,
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

    for rep in range(n_reps):
        rep_seed = args.seed + rep
        verbose_rep = (rep == 0)
        if not verbose_rep:
            print(f"  rep {rep + 1}/{n_reps} (seed={rep_seed}) ...",
                  end="", flush=True)

        res = run_single_experiment(
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
        )
        for _, prefix in methods:
            for suf in collect_suffixes:
                all_vals[prefix + suf].append(res[prefix + suf])

        if not verbose_rep:
            print(f"  FDRNN={res['fdrnn_avg_dist']:.4f}")

    # Paper-ready tables
    W = 82
    header_lines = [
        f"  RESULTS: {args.case}  ({n_reps} reps)",
        f"  n_train={args.n_train}, n_test={args.n_test}",
    ]

    print(f"\n\n{'=' * W}")
    for hl in header_lines:
        print(hl)
    print(f"{'=' * W}")

    base_prefix = methods[0][1]
    for panel_label, suf in [("Panel A: Avg Distance", "_avg_dist")]:
        base_arr = np.array(all_vals[base_prefix + suf])
        print(f"\n  {panel_label}")
        print(f"  {'Method':<30} {'Mean':>10} {'(SE)':>10} "
              f"{'Improv%':>8} {'(SE)':>8} {'Time(s)':>8}")
        print(f"  {'-' * (W - 4)}")

        for label, prefix in methods:
            arr = np.array(all_vals[prefix + suf])
            t_arr = np.array(all_vals[prefix + "_time_sec"])
            mean = arr.mean()
            se = arr.std(ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
            t_mean = t_arr.mean()

            if prefix == base_prefix:
                print(f"  {label:<30} {mean:>10.4f} ({se:>7.4f}) "
                      f"{'---':>8} {'':>8} {t_mean:>7.1f}s")
            else:
                imp_arr = (base_arr - arr) / base_arr * 100
                imp_mean = imp_arr.mean()
                imp_se = (imp_arr.std(ddof=1) / np.sqrt(n_reps)
                          if n_reps > 1 else 0.0)
                print(f"  {label:<30} {mean:>10.4f} ({se:>7.4f}) "
                      f"{imp_mean:>7.1f}% ({imp_se:>5.1f}%) "
                      f"{t_mean:>7.1f}s")

    print(f"{'=' * W}\n")

    # Save to CSV
    import csv
    with open(args.output_file, 'w', newline='') as csvfile:
        fieldnames = list(all_vals.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Since all lists have the same length, write row by row
        n_rows = len(all_vals[fieldnames[0]])
        for i in range(n_rows):
            row = {k: all_vals[k][i] for k in fieldnames}
            writer.writerow(row)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()