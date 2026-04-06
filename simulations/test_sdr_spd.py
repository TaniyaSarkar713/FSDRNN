#!/usr/bin/env python3
"""
Simulation: Dimension Reduction for Fréchet Regression on SPD matrices.

Reproduces Section 6.3 (Scenario II) from
  "Dimension Reduction for Fréchet Regression"

Setup
-----
- X ∈ R^p (p = 10 or 20)
- Model II-1 (d₀ = 1): Y is a 2×2 SPD matrix depending on X only through β₁ᵀX
- Model II-2 (d₀ = 2): Y is a 3×3 SPD matrix depending on X through (β₁ᵀX, β₃ᵀX)
- Y is generated via  log(Y) ~ N_{rr}(log D(X), 0.25)
- Directions:
    β₁ = (1,1,0,...,0),  β₃ = (1,2,0,...,0,2)

Competitors
-----------
  1. Global Mean    — ignores X entirely
  2. GFR            — Global Fréchet Regression (OLS weights)
  3. DFR            — Deep Fréchet Regression (manifold + DNN + kernel)
  4. FDRNN (Proposed) — Fréchet Dimension Reduction Neural Network:
                       (i)  shared SDR reduction  f : R^p → R^d
                       (ii) V independent weight heads
                       (iii) weight refining via low-rank mixing

Hyperparameter tuning is ON by default (grid search on a validation split).
Use ``--no-tune`` to skip.

Usage
-----
  python simulations/test_sdr_spd.py                       # default: Model II-2
  python simulations/test_sdr_spd.py --model II-1
  python simulations/test_sdr_spd.py --model II-2 --setting b
  python simulations/test_sdr_spd.py --model II-2
"""

import sys
import os
import argparse
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster/CI
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.spd_frechet import (
    SDRCorrSPDDataset,
    FrechetDRNN,
    train_frechet_model,
    evaluate_frechet_model,
    differentiable_frechet_mean,
    get_distance_fn,
    GlobalFrechetRegression,
    DeepFrechetRegression,
    grid_search_frechet,
    grid_search_dfr,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_on_loader(model, loader, Y_ref, dist_fn, metric_name, device):
    """Evaluate a trained weight-net on a DataLoader; return avg d and avg d²."""
    model.eval()
    all_d = []
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            W = model(X_b)
            if W.dim() == 2:
                m = differentiable_frechet_mean(W, Y_ref.to(device), metric_name)
                all_d.append(dist_fn(m, Y_b).cpu())
            else:
                V = W.shape[2]
                d_batch = []
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v, :, :]
                    Y_b_v = Y_b[:, v, :, :]
                    m_v = differentiable_frechet_mean(w_v, Y_ref_v.to(device), metric_name)
                    d_v = dist_fn(m_v, Y_b_v)
                    d_batch.append(d_v)
                d_avg = torch.stack(d_batch).mean(dim=0)  # [B]
                all_d.append(d_avg.cpu())
    cat = torch.cat(all_d)
    return cat.mean().item(), (cat ** 2).mean().item()


def _eval_vs_dtrue(pred_mats, D_true, dist_fn):
    """Compute avg d(pred, D_true) and avg d²(pred, D_true)."""
    if pred_mats.dim() == 3:
        d = dist_fn(pred_mats, D_true)
    else:
        # Multi-response
        V = pred_mats.shape[1]
        d_list = []
        for v in range(V):
            d_v = dist_fn(pred_mats[:, v, :, :], D_true[:, v, :, :])
            d_list.append(d_v)
        d = torch.stack(d_list).mean(dim=0)  # [n]
    return d.mean().item(), (d ** 2).mean().item()


def _nn_predict_all(model, loader, Y_ref, metric_name, device):
    """Collect all predictions from a weight-net model on *loader*."""
    model.eval()
    preds = []
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b = X_b.to(device)
            W = model(X_b)
            if W.dim() == 2:
                m = differentiable_frechet_mean(W, Y_ref.to(device), metric_name)
            else:
                V = W.shape[2]
                m_v = []
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v, :, :]
                    m_v.append(differentiable_frechet_mean(w_v, Y_ref_v.to(device), metric_name))
                m = torch.stack(m_v, dim=1)  # [B, V, p, p]
            preds.append(m.cpu())
    return torch.cat(preds)


def _effective_rank(model_fdrnn):
    """Compute rank metric of the first Linear layer in the reduction net.

    Rank metric = (\u03a3 \u03c3_i)\u00b2 / (\u03a3 \u03c3_i\u00b2), where \u03c3_i are the singular
    values.  Returns (rank_metric, singular_values_np).
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


def run_sdr_simulation(
    model: str = "II-2",
    setting: str = "a",
    n_train: int = 200,
    n_test: int = 100,
    p: int = 10,
    n_responses: int = 5,
    metric_name: str = "frobenius",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    entropy_reg: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
    dfr_tuned_params: dict = None,
    fdrnn_tuned_params: dict = None,
):
    """
    Run the SDR simulation comparing all methods.

    Args:
        model:          'II-1' (d₀=1, r=2) or 'II-2' (d₀=2, r=3)
        setting:        'a' (i.i.d. Gaussian X) or 'b' (correlated non-elliptical)
        n_train:        training set size
        n_test:         test set size
        p:              predictor dimension
        metric_name:    SPD distance metric
        epochs:         training epochs
        lr:             learning rate
        batch_size:     mini-batch size
        entropy_reg:    entropy regularisation λ
        seed:           random seed
        device:         'cpu' or 'cuda'
        verbose:        print progress
        dfr_tuned_params: tuned params for DFR
        fdrnn_tuned_params:   tuned params for FDRNN with d=p (fixed)

    Returns:
        dict with results for all methods
    """
    set_seed(seed)

    structural_dim = 1 if model == "II-1" else 2

    if verbose:
        print(f"\n{'='*70}")
        print(f"  SDR Simulation — {model}-({setting})")
        print(f"  p={p}, n_train={n_train}, n_test={n_test}")
        print(f"  structural dim d₀={structural_dim}")
        print(f"  metric={metric_name}, epochs={epochs}, lr={lr}")
        print(f"  entropy_reg (λ)={entropy_reg}")
        print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    ds_train = SDRCorrSPDDataset(
        n=n_train, p=p, model=model, setting=setting, n_responses=n_responses, seed=seed
    )
    ds_test = SDRCorrSPDDataset(
        n=n_test, p=p, model=model, setting=setting, n_responses=n_responses, seed=seed + 1000
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # Early stopping: hold out 20% of training data as validation
    from torch.utils.data import Subset
    n_val_es = max(1, int(n_train * 0.2))
    n_trn_es = n_train - n_val_es
    gen = torch.Generator().manual_seed(seed)
    es_indices = torch.randperm(n_train, generator=gen).tolist()
    es_train_sub = Subset(ds_train, es_indices[:n_trn_es])
    es_val_sub = Subset(ds_train, es_indices[n_trn_es:])
    es_train_loader = DataLoader(es_train_sub, batch_size=batch_size, shuffle=True)
    es_val_loader = DataLoader(es_val_sub, batch_size=batch_size, shuffle=False)

    mat_dim = ds_train.mat_dim
    Y_ref = ds_train.Y  # [n_train, r, r]

    if verbose:
        X_s, Y_s = ds_train[0]
        print(f"\n  X shape:  {tuple(X_s.shape)}")
        print(f"  Y shape:  {tuple(Y_s.shape)}  (r={mat_dim})")
        print(f"  Y eigenvalues: {torch.linalg.eigvalsh(Y_s).tolist()}")
        print(f"  True central subspace B₀ shape: "
              f"{tuple(ds_train.B0.shape)}")

    dist_fn = get_distance_fn(metric_name)
    D_true_test = ds_test.D_true   # [n_test, r, r] conditional Fréchet centre
    results = {"model": model, "setting": setting, "p": p,
               "n_train": n_train, "n_test": n_test,
               "structural_dim": structural_dim,
               "metric_name": metric_name}

    # ------------------------------------------------------------------
    # 2. Global Mean baseline
    # ------------------------------------------------------------------
    global_mean = ds_train.Y.mean(dim=0)  # [n_responses, r, r]
    t0 = time.time()
    baseline_dists = []
    for X_batch, Y_batch in test_loader:
        B = X_batch.size(0)
        gm = global_mean.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_responses, r, r]
        # Compute distance per response and average
        d_per_response = []
        for v in range(n_responses):
            d_v = dist_fn(gm[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d = torch.stack(d_per_response).mean(dim=0)  # [B]
        baseline_dists.append(d)
    baseline_dists_cat = torch.cat(baseline_dists)
    base_avg = baseline_dists_cat.mean().item()
    base_avg_sq = (baseline_dists_cat ** 2).mean().item()
    base_time = time.time() - t0
    # D_true eval
    gm_expand = global_mean.unsqueeze(0).expand(n_test, -1, -1, -1)  # [n_test, 2, r, r]
    base_dtrue, base_dtrue_sq = _eval_vs_dtrue(gm_expand, D_true_test, dist_fn)
    results["global_mean_avg_dist"] = base_avg
    results["global_mean_avg_dist_sq"] = base_avg_sq
    results["global_mean_time"] = base_time
    results["global_mean_dtrue"] = base_dtrue
    results["global_mean_dtrue_sq"] = base_dtrue_sq
    if verbose:
        print(f"\n  Global Mean avg d(pred,Y): {base_avg:.6f}  "
              f"d(pred,D_true): {base_dtrue:.6f}")

    # ------------------------------------------------------------------
    # 3. Global Fréchet Regression (GFR)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- Global Fréchet Regression ---")
    t0 = time.time()
    gfr = GlobalFrechetRegression(dist_name=metric_name)
    gfr.fit(ds_train.X, ds_train.Y)
    gfr_dists = []
    gfr_preds = []
    for X_batch, Y_batch in test_loader:
        Y_pred = gfr.predict(X_batch)  # [B, n_responses, r, r]
        # Compute distance per response and average
        d_per_response = []
        for v in range(n_responses):
            d_v = dist_fn(Y_pred[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d = torch.stack(d_per_response).mean(dim=0)  # [B]
        gfr_dists.append(d)
        gfr_preds.append(Y_pred)
    gfr_dists_cat = torch.cat(gfr_dists)
    gfr_avg = gfr_dists_cat.mean().item()
    gfr_avg_sq = (gfr_dists_cat ** 2).mean().item()
    gfr_time = time.time() - t0
    gfr_preds_cat = torch.cat(gfr_preds)
    gfr_dtrue, gfr_dtrue_sq = _eval_vs_dtrue(gfr_preds_cat, D_true_test, dist_fn)
    results["gfr_avg_dist"] = gfr_avg
    results["gfr_avg_dist_sq"] = gfr_avg_sq
    results["gfr_time"] = gfr_time
    results["gfr_dtrue"] = gfr_dtrue
    results["gfr_dtrue_sq"] = gfr_dtrue_sq
    if verbose:
        print(f"  GFR avg d(pred,Y):     {gfr_avg:.6f}  "
              f"d(pred,D_true): {gfr_dtrue:.6f}  ({gfr_time:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Deep Fréchet Regression (DFR)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- Deep Fréchet Regression (DFR) ---")
    t0 = time.time()
    _dfr_md = min(structural_dim, 2)
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
    dfr_preds = []
    for X_batch, Y_batch in test_loader:
        Y_pred = dfr.predict(X_batch)  # [B, n_responses, r, r]
        # Compute distance per response and average
        d_per_response = []
        for v in range(n_responses):
            d_v = dist_fn(Y_pred[:, v, :, :], Y_batch[:, v, :, :])
            d_per_response.append(d_v)
        d = torch.stack(d_per_response).mean(dim=0)  # [B]
        dfr_dists.append(d)
        dfr_preds.append(Y_pred)
    dfr_dists_cat = torch.cat(dfr_dists)
    dfr_avg = dfr_dists_cat.mean().item()
    dfr_avg_sq = (dfr_dists_cat ** 2).mean().item()
    dfr_time = time.time() - t0
    dfr_preds_cat = torch.cat(dfr_preds)
    dfr_dtrue, dfr_dtrue_sq = _eval_vs_dtrue(dfr_preds_cat, D_true_test, dist_fn)
    results["dfr_avg_dist"] = dfr_avg
    results["dfr_avg_dist_sq"] = dfr_avg_sq
    results["dfr_time"] = dfr_time
    results["dfr_dtrue"] = dfr_dtrue
    results["dfr_dtrue_sq"] = dfr_dtrue_sq
    if verbose:
        print(f"  DFR avg d(pred,Y): {dfr_avg:.6f}  "
              f"d(pred,D_true): {dfr_dtrue:.6f}  ({dfr_time:.1f}s)")

    # ------------------------------------------------------------------
    # 5. FDRNN — Fréchet Dimension Reduction Neural Network
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  --- FDRNN (reduction dim={fdrnn_tuned_params.get('reduction_dim', p) if fdrnn_tuned_params else p}) ---")
    set_seed(seed)
    _fdrnn_lr = fdrnn_tuned_params.get("lr", lr) if fdrnn_tuned_params else lr
    _fdrnn_er = fdrnn_tuned_params.get("entropy_reg", entropy_reg) if fdrnn_tuned_params else entropy_reg
    _fdrnn_wd = fdrnn_tuned_params.get("weight_decay", 1e-5) if fdrnn_tuned_params else 1e-5
    _fdrnn_do = fdrnn_tuned_params.get("dropout", 0.0) if fdrnn_tuned_params else 0.0
    _fdrnn_nr = fdrnn_tuned_params.get("nuclear_reg", 0.01) if fdrnn_tuned_params else 0.01
    _fdrnn_rd = fdrnn_tuned_params.get("reduction_dim", p) if fdrnn_tuned_params else p
    _fdrnn_rt = fdrnn_tuned_params.get("reduction_type", "nonlinear") if fdrnn_tuned_params else "nonlinear"
    _fdrnn_rr = fdrnn_tuned_params.get("response_rank", 2) if fdrnn_tuned_params else 2
    _fdrnn_enc = fdrnn_tuned_params.get("encoder_sizes", [32]) if fdrnn_tuned_params else [32]
    _fdrnn_head = fdrnn_tuned_params.get("head_sizes", [32]) if fdrnn_tuned_params else [32]
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
    n_params_fdrnn = sum(pp.numel() for pp in model_fdrnn.parameters())
    if verbose:
        print(f"  Parameters: {n_params_fdrnn:,}")
        print(f"  Reduction type: {_fdrnn_rt}, nuclear_reg: {_fdrnn_nr}")
        print(f"  Response rank: {_fdrnn_rr}")

    t0 = time.time()
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
    fdrnn_time = time.time() - t0

    fdrnn_avg, fdrnn_avg_sq = _evaluate_on_loader(model_fdrnn, test_loader, Y_ref, dist_fn, metric_name, device)
    fdrnn_train_avg, fdrnn_train_sq = _evaluate_on_loader(model_fdrnn, train_loader, Y_ref, dist_fn, metric_name, device)
    fdrnn_preds = _nn_predict_all(model_fdrnn, test_loader, Y_ref, metric_name, device)
    fdrnn_dtrue, fdrnn_dtrue_sq = _eval_vs_dtrue(fdrnn_preds, D_true_test, dist_fn)
    results["fdrnn_avg_dist"] = fdrnn_avg
    results["fdrnn_avg_dist_sq"] = fdrnn_avg_sq
    results["fdrnn_train_dist"] = fdrnn_train_avg
    results["fdrnn_train_dist_sq"] = fdrnn_train_sq
    results["fdrnn_time"] = fdrnn_time
    results["fdrnn_final_loss"] = history_fdrnn[-1]
    results["fdrnn_reduction_dim"] = _fdrnn_rd
    results["fdrnn_dtrue"] = fdrnn_dtrue
    results["fdrnn_dtrue_sq"] = fdrnn_dtrue_sq
    if _fdrnn_rr is not None:
        resp_eff_rank, resp_sv = _effective_response_rank(model_fdrnn)
        results["fdrnn_response_eff_rank"] = resp_eff_rank
        if verbose:
            print(f"  Response LoRA effective rank: {resp_eff_rank:.2f}")
            print(f"  Response LoRA singular values: {[f'{s:.4f}' for s in resp_sv]}")
    if verbose:
        print(f"  FDRNN avg d(pred,Y): {fdrnn_avg:.6f}  "
              f"d(pred,D_true): {fdrnn_dtrue:.6f}  ({fdrnn_time:.1f}s)")
        print(f"  FDRNN train d:     {fdrnn_train_avg:.6f}  "
              f"(gap={fdrnn_avg - fdrnn_train_avg:+.4f})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE  ({model}-({setting}), p={p}, "
              f"d₀={structural_dim})")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'avg d(pred,Y)':>14} {'Improv%':>8}")
        print(f"  {'-'*52}")

        def _imp(val):
            return (base_avg - val) / base_avg * 100 if base_avg > 0 else 0

        print(f"  {'Global Mean':<30} {base_avg:>14.6f} {'---':>8}")
        print(f"  {'Global Fréchet Reg (GFR)':<30} {gfr_avg:>14.6f} {_imp(gfr_avg):>7.1f}%")
        print(f"  {'Deep Fréchet Reg (DFR)':<30} {dfr_avg:>14.6f} {_imp(dfr_avg):>7.1f}%")
        print(f"  {'FDRNN (response_rank=':<20}{_fdrnn_rr}){'':<10} {fdrnn_avg:>14.6f} {_imp(fdrnn_avg):>7.1f}%")

        # Train / test gap summary
        print(f"\n  --- Train / Test gap (d) ---")
        print(f"  {'Method':<18} {'Train d':>10} {'Test d':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        for tag, tr, te in [
            ("FDRNN (response_rank)", fdrnn_train_avg, fdrnn_avg),
        ]:
            print(f"  {tag:<18} {tr:>10.4f} {te:>10.4f} {te - tr:>+10.4f}")

        # Red Flag 4: diagnostic if GFR wins
        if gfr_avg <= fdrnn_avg:
            print(f"\n  ⚠  WARNING: GFR (linear) is competitive with / better than "
                  f"all NN methods.")
            print(f"     This may indicate the simulation is too easy / too linear ")
            print(f"     for the nonlinear FDRNN to show advantage.")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SDR Simulation: Fréchet Regression on SPD matrices "
                    "(Section 6.3)"
    )
    parser.add_argument("--model", type=str, default="II-2",
                        choices=["II-1", "II-2"],
                        help="Model (II-1: d₀=1, II-2: d₀=2)")
    parser.add_argument("--setting", type=str, default="a",
                        choices=["a", "b"],
                        help="Predictor setting (a: Gaussian, b: correlated)")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--p", type=int, default=10,
                        help="Predictor dimension")
    parser.add_argument("--n_responses", type=int, default=20,
                        help="Number of responses per sample (default: 20)")
    parser.add_argument("--metric", type=str, default="frobenius",
                        help="SPD distance metric")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--entropy_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_reps", type=int, default=10,
                        help="Number of repetitions (default: 10)")
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

    structural_dim = 1 if args.model == "II-1" else 2
    n_reps = args.n_reps

    # ==================================================================
    # Optional: hyperparameter tuning via grid search
    # ==================================================================
    dfr_tuned_params = None
    fdrnn_tuned_params = None
    chosen_d = None

    if args.tune:
        print(f"\n{'#'*70}")
        print(f"  TUNING: {args.model}-({args.setting})")
        print(f"{'#'*70}")

        tune_ds = SDRCorrSPDDataset(
            n=args.n_train, p=args.p,
            model=args.model, setting=args.setting, n_responses=args.n_responses, seed=0,
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
            dist_name=args.metric,
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
                dist_name=args.metric,
                param_grid=fdrnn_stage_a_grid,
                fixed_model_kwargs={
                    "input_dim": args.p,
                    "n_responses": args.n_responses,
                    "encoder_sizes": [32],
                    "head_sizes": [32],
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
            "response_rank": [min(10, args.n_responses)],  # adaptive LoRA
            "reduction_dim": [chosen_d],  # fixed from Stage A
            "reduction_type": [chosen_rt],  # fixed from Stage A
            "dropout": [0.0],  # fixed
        }
        fdrnn_result = grid_search_frechet(
            dataset=tune_ds,
            parent_Y=tune_ds.Y,
            model_class=FrechetDRNN,
            dist_name=args.metric,
            param_grid=fdrnn_stage_b_grid,
            fixed_model_kwargs={
                "input_dim": args.p,
                "n_responses": args.n_responses,
                "encoder_sizes": [32],
                "head_sizes": [32],
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

        print(f"\n  --- Chosen DFR params:            {dfr_tuned_params}")
        print(f"  --- Chosen FDRNN (d={chosen_d}) params: {fdrnn_tuned_params}")
        print()

    # ==================================================================
    # Main experiment
    # ==================================================================
    print(f"\n{'#'*70}")
    print(f"  SDR Simulation — {args.model}-({args.setting})")
    print(f"  p={args.p}, n_train={args.n_train}, n_test={args.n_test}")
    print(f"  d₀={structural_dim}")
    print(f"  metric={args.metric}, epochs={args.epochs}, {n_reps} reps")
    if dfr_tuned_params:
        print(f"  DFR tuned:       {dfr_tuned_params}")
    if fdrnn_tuned_params:
        print(f"  FDRNN (d={chosen_d}) tuned:  {fdrnn_tuned_params}")
    print(f"{'#'*70}")

    # Resolve FDRNN reduction_dim labels
    if fdrnn_tuned_params is None:
        chosen_d = args.p  # default to no reduction
        _fdrnn_rr = None
    else:
        _fdrnn_rr = fdrnn_tuned_params.get("response_rank", None)

    # Collect per-rep results
    # FDRNN is the proposed method — listed last so it stands out
    methods = [
        ("Global Mean",                     "global_mean"),
        ("Global Fr\u00e9chet Reg (GFR)",        "gfr"),
        ("Deep Fr\u00e9chet Reg (DFR)",          "dfr"),
        (f"FDRNN (d={chosen_d}, response_rank={_fdrnn_rr})",   "fdrnn"),
    ]
    collect_suffixes = ["_avg_dist", "_avg_dist_sq", "_time", "_dtrue", "_dtrue_sq"]
    all_vals = {}
    response_eff_ranks = []  # collect response LoRA rank metric values
    for _, prefix in methods:
        for suf in collect_suffixes:
            all_vals[prefix + suf] = []

    for rep in range(n_reps):
        rep_seed = args.seed + rep
        verbose_rep = (rep == 0)
        if not verbose_rep:
            print(f"  rep {rep + 1}/{n_reps} (seed={rep_seed}) …", end="", flush=True)

        res = run_sdr_simulation(
            model=args.model,
            setting=args.setting,
            n_train=args.n_train,
            n_test=args.n_test,
            p=args.p,
            n_responses=args.n_responses,
            metric_name=args.metric,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            entropy_reg=args.entropy_reg,
            seed=rep_seed,
            device=device,
            verbose=verbose_rep,
            dfr_tuned_params=dfr_tuned_params,
            fdrnn_tuned_params=fdrnn_tuned_params,
        )
        for _, prefix in methods:
            for suf in collect_suffixes:
                all_vals[prefix + suf].append(res[prefix + suf])
        response_eff_ranks.append(res.get("fdrnn_response_eff_rank", float('nan')))

        if not verbose_rep:
            print(f"  FDRNN(response_rank)={res['fdrnn_avg_dist']:.4f}")

    # ------------------------------------------------------------------
    # Paper-ready tables: d and d²  (mean, SE, improvement%, runtime)
    # ------------------------------------------------------------------
    W = 82
    header_lines = [
        f"  RESULTS: {args.model}-({args.setting})  ({n_reps} reps)",
        f"  p={args.p}, n_train={args.n_train}, n_test={args.n_test}, "
        f"d₀={structural_dim}, metric={args.metric}",
    ]
    if dfr_tuned_params:
        header_lines.append(f"  DFR tuned:       {dfr_tuned_params}")
    if fdrnn_tuned_params:
        header_lines.append(f"  FDRNN (d=p) tuned: {fdrnn_tuned_params}")
    resp_eff_mean = np.nanmean(response_eff_ranks) if response_eff_ranks else 0.0
    resp_eff_se = np.nanstd(response_eff_ranks, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
    if not np.isnan(resp_eff_mean):
        header_lines.append(f"  FDRNN response LoRA rank metric: {resp_eff_mean:.2f} (SE {resp_eff_se:.2f})")

    print(f"\n\n{'=' * W}")
    for hl in header_lines:
        print(hl)
    print(f"{'=' * W}")

    base_prefix = methods[0][1]  # "global_mean"
    for panel_label, suf in [("Panel A: d(pred, Y)", "_avg_dist"),
                              ("Panel B: d²(pred, Y)", "_avg_dist_sq"),
                              ("Panel C: d(pred, D_true) — centre recovery", "_dtrue"),
                              ("Panel D: d²(pred, D_true)", "_dtrue_sq")]:
        base_arr = np.array(all_vals[base_prefix + suf])
        print(f"\n  {panel_label}")
        print(f"  {'Method':<34} {'Mean':>10} {'(SE)':>10} "
              f"{'Improv%':>8} {'(SE)':>8} {'Time(s)':>8}")
        print(f"  {'─' * (W - 4)}")

        for label, prefix in methods:
            arr = np.array(all_vals[prefix + suf])
            t_arr = np.array(all_vals[prefix + "_time"])
            mean = arr.mean()
            se = arr.std(ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
            t_mean = t_arr.mean()

            if prefix == base_prefix:
                print(f"  {label:<34} {mean:>10.4f} ({se:>7.4f}) "
                      f"{'—':>8} {'':>8} {t_mean:>7.1f}s")
            else:
                imp_arr = (base_arr - arr) / base_arr * 100
                imp_mean = imp_arr.mean()
                imp_se = imp_arr.std(ddof=1) / np.sqrt(n_reps) if n_reps > 1 else 0.0
                print(f"  {label:<34} {mean:>10.4f} ({se:>7.4f}) "
                      f"{imp_mean:>7.1f}% ({imp_se:>5.1f}%) {t_mean:>7.1f}s")

    print(f"{'=' * W}\n")

    # ------------------------------------------------------------------
    # Figure: distribution of prediction errors across reps
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_specs = [
        ("$d(\\hat{Y}, Y)$", "_avg_dist"),
        ("$d^2(\\hat{Y}, Y)$", "_avg_dist_sq"),
        ("$d(\\hat{Y}, D_{true})$", "_dtrue"),
        ("$d^2(\\hat{Y}, D_{true})$", "_dtrue_sq"),
    ]
    short_labels = [label.split("(")[0].strip() for label, _ in methods]
    colors = ["#999999", "#4daf4a", "#e41a1c", "#377eb8", "#984ea3"]

    for ax, (ylabel, suf) in zip(axes.flat, panel_specs):
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
        f"SDR {args.model}-({args.setting})  |  "
        f"n={args.n_train}, p={args.p}, {args.metric}  |  {n_reps} reps",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_dir = os.path.join(project_root, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(
        fig_dir,
        f"sdr_spd_{args.model}_{args.setting}_{args.metric}_errors.pdf",
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")


if __name__ == "__main__":
    main()
