import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'simulations_sdr')
from setup1_linear import generate_synthetic_data, OracleReductionFSDRNNWrapper, evaluate_prediction
import numpy as np

# Generate data
X_train, Y_train, z_train, beta, coeffs = generate_synthetic_data(100, seed=42)
X_test, Y_test, z_test, _, _ = generate_synthetic_data(100, seed=1042)

print("Data shapes:")
print(f"  Z_train: {z_train.shape}, Y_train: {Y_train.shape}") 
print(f"  Z_test: {z_test.shape}, Y_test: {Y_test.shape}")

# True relationship: Y = coeffs[:, 0] * z1 + coeffs[:, 1] * z2 + coeffs[:, 2] * z1*z2 + noise
Y_true_linear = np.zeros((100, 8))
for v in range(8):
    Y_true_linear[:, v] = (coeffs[v, 0] * z_test[:, 0] + 
                           coeffs[v, 1] * z_test[:, 1] + 
                           coeffs[v, 2] * z_test[:, 0] * z_test[:, 1])
mse_true_perfect = np.mean((Y_test - Y_true_linear) ** 2)
print(f"\nOra MSE with perfect knowledge of relation (ignoring noise): {mse_true_perfect:.6f}")

# Create and train oracle
oracle = OracleReductionFSDRNNWrapper(output_dim=8, latent_dim=2, hidden_dim=32, 
                                     lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False)
oracle.fit(z_train, Y_train)
Y_pred_oracle = oracle.predict(z_test)

# Evaluate
mse_oracle = evaluate_prediction(Y_test, Y_pred_oracle)['mse']
print(f"\nOracle MSE on test set: {mse_oracle:.6f}")

# Check if it at least overfits to training data
Y_pred_train = oracle.predict(z_train)
mse_train = evaluate_prediction(Y_train, Y_pred_train)['mse']
print(f"Oracle MSE on train set: {mse_train:.6f}")

# Sanity check - compute MSE if we just output mean of Y_train
y_train_mean = np.mean(Y_train, axis=0)
Y_pred_mean = np.repeat(y_train_mean[np.newaxis, :], Y_test.shape[0], axis=0)
mse_mean = np.mean((Y_test - Y_pred_mean) ** 2)
print(f"Baseline (mean) MSE: {mse_mean:.6f}")
