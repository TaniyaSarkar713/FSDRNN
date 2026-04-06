#!/usr/bin/env python3
"""Check if oracle overfitting due to too many epochs."""
import sys
import os
workspace = '/Users/taniyamac/Documents/FSU/course materials/Tensor research/FSDRNN/FSDRNN/code_taniya'
sys.path.insert(0, workspace)
sys.path.insert(0, os.path.join(workspace, 'simulations_sdr'))
os.chdir(workspace)

import numpy as np
import setup6_wasserstein_distributions as setup6

generate_synthetic_data = setup6.generate_synthetic_data
OracleFSdrnnWrapper = setup6.OracleFSdrnnWrapper
evaluate_mse = setup6.evaluate_mse

# Test oracle with different epoch counts
n_train = 50
n_test = 30
seed = 42

print("=" * 80)
print("TESTING ORACLE WITH DIFFERENT EPOCH COUNTS")
print("=" * 80)

X_train, Y_train_quant, z_train, beta = generate_synthetic_data(n_train, seed=seed)
X_test, Y_test_quant, z_test, _ = generate_synthetic_data(n_test, seed=seed+1000)

_, V, n_q = Y_train_quant.shape

# Extract parameters
Y_train_params = np.zeros((n_train, V, 2))
for i in range(n_train):
    for v in range(V):
        Y_train_params[i, v, 0] = np.median(Y_train_quant[i, v, :])
        Y_train_params[i, v, 1] = np.std(Y_train_quant[i, v, :]) + 0.1

n_test_actual = X_test.shape[0]
Y_test_params = np.zeros((n_test_actual, V, 2))
for i in range(n_test_actual):
    for v in range(V):
        Y_test_params[i, v, 0] = np.median(Y_test_quant[i, v, :])
        Y_test_params[i, v, 1] = np.std(Y_test_quant[i, v, :]) + 0.1

# Test with different epochs
for epochs in [500, 1000, 1500, 3000]:
    print(f"\nTesting Oracle with {epochs} epochs...")
    oracle = OracleFSdrnnWrapper(output_dim=V, latent_dim=2, B_true=beta, n_quantiles=n_q,
                                 lr=5e-4, epochs=epochs, device='cpu')
    oracle.fit(X_train, Y_train_quant)
    Y_pred = oracle.predict(X_test)
    mse = evaluate_mse(Y_test_params, Y_pred)
    print(f"  Oracle Test MSE (epochs={epochs}): {mse:.6f}")

print("\n" + "=" * 80)
print("✓ Epoch test complete")
