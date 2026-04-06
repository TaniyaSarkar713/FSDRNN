#!/usr/bin/env python3
"""Diagnose why oracle performs worse than proposed in setup6."""
import sys
import os
workspace = '/Users/taniyamac/Documents/FSU/course materials/Tensor research/FSDRNN/FSDRNN/code_taniya'
sys.path.insert(0, workspace)
sys.path.insert(0, os.path.join(workspace, 'simulations_sdr'))
os.chdir(workspace)

import numpy as np
import torch
import torch.nn as nn

# Import from setup6
import setup6_wasserstein_distributions as setup6
generate_synthetic_data = setup6.generate_synthetic_data
FSdrnnWrapper = setup6.FSdrnnWrapper
OracleFSdrnnWrapper = setup6.OracleFSdrnnWrapper
evaluate_mse = setup6.evaluate_mse

# Run single simulation
print("=" * 80)
print("ORACLE DIAGNOSIS FOR SETUP6")
print("=" * 80)

n_train = 50
n_test = 30
seed = 42

print(f"\nGenerating synthetic data (n_train={n_train}, n_test={n_test})...")
X_train, Y_train_quant, z_train, beta = generate_synthetic_data(n_train, seed=seed, nonlinear=False)
X_test, Y_test_quant, z_test, _ = generate_synthetic_data(n_test, seed=seed+1000, nonlinear=False)

# Extract dimensions
V = Y_train_quant.shape[1]  # 8 responses
n_q = Y_train_quant.shape[2]  # 50 quantiles

print(f"  X_train shape: {X_train.shape}")
print(f"  Y_train_quant shape: {Y_train_quant.shape}")
print(f"  z_train shape: {z_train.shape}")
print(f"  beta shape: {beta.shape}")
print(f"  beta (first 3 rows):\n{beta[:3]}")

print(f"\nData ready for full quantile training:")
print(f"  X_train: {X_train.shape}, Y_train_quant: {Y_train_quant.shape}")
print(f"  X_test: {X_test.shape}, Y_test_quant: {Y_test_quant.shape}")

# Test 1: Training FSDRNN on full quantiles
print("\n" + "-" * 80)
print("Training FSDRNN (learned encoder, full quantiles)...")
wrapper = FSdrnnWrapper(X_train.shape[1], output_dim=V, n_quantiles=n_q, 
                        lr=5e-4, epochs=1000, device='cpu')
wrapper.fit(X_train, Y_train_quant)
Y_pred_fsdrnn = wrapper.predict(X_test)
mse_fsdrnn = evaluate_mse(Y_test_quant, Y_pred_fsdrnn)
print(f"  FSDRNN Test MSE (on full {n_q}D quantiles): {mse_fsdrnn:.6f}")

# Test 2: Training Oracle on full quantiles
print("\n" + "-" * 80)
print("Training Oracle FSDRNN (true B_0, 1000 epochs to match FSDRNN)...")
oracle = OracleFSdrnnWrapper(output_dim=V, latent_dim=2, B_true=beta, n_quantiles=n_q,
                             lr=5e-4, epochs=1000, device='cpu')
oracle.fit(X_train, Y_train_quant)
Y_pred_oracle = oracle.predict(X_test)
mse_oracle = evaluate_mse(Y_test_quant, Y_pred_oracle)
print(f"  Oracle Test MSE (on full {n_q}D quantiles): {mse_oracle:.6f}")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
ratio = mse_fsdrnn / (mse_oracle + 1e-10)
print(f"FSDRNN MSE: {mse_fsdrnn:.6f}")
print(f"Oracle MSE: {mse_oracle:.6f}")
print(f"Ratio (FSDRNN / Oracle): {ratio:.4f}")

if ratio > 1.0:
    print(f"✓ Oracle is BETTER (ratio = {ratio:.4f}x improvement)")
else:
    print(f"✗ Oracle is WORSE (ratio = {ratio:.4f}, oracle at disadvantage)")

print(f"\nNote: Training on full {n_q}D quantile functions (not lossy summaries)")
print(f"Expected: Oracle advantage should manifest (>1.3-2.5x ratio)")
print("\n✓ Test complete")
