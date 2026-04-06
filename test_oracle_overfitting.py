#!/usr/bin/env python3
"""Check for oracle overfitting."""
import sys
import os
workspace = '/Users/taniyamac/Documents/FSU/course materials/Tensor research/FSDRNN/FSDRNN/code_taniya'
sys.path.insert(0, workspace)
sys.path.insert(0, os.path.join(workspace, 'simulations_sdr'))
os.chdir(workspace)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import setup6_wasserstein_distributions as setup6

generate_synthetic_data = setup6.generate_synthetic_data
evaluate_mse = setup6.evaluate_mse

n_train = 50
n_test = 30
seed = 42

print("=" * 80)
print("CHECKING ORACLE OVERFITTING")
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

# Manually train oracle with train/test loss tracking
B_torch = torch.tensor(beta, dtype=torch.float32)
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_params, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_params, dtype=torch.float32)

# Oracle latents
Z_train = X_train_torch @ B_torch
Z_test = X_test_torch @ B_torch

# Create heads
hidden_dim = 32
heads = nn.ModuleList([
    nn.Sequential(
        nn.Linear(2, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 2)
    )
    for _ in range(V)
])

r_max = min(V, 6)
lora_A = nn.Parameter(torch.randn(2, r_max) * 0.01)
lora_B = nn.Parameter(torch.randn(r_max, V * 2) * 0.01)

params = list(heads.parameters()) + [lora_A, lora_B]
optimizer = optim.Adam(params, lr=5e-4)
criterion = nn.MSELoss()

def compute_pred(Z, heads, lora_A, lora_B):
    outputs = []
    for head in heads:
        outputs.append(head(Z))
    pred = torch.stack(outputs, dim=1)
    lora_contrib = Z @ lora_A @ lora_B
    lora_contrib = lora_contrib.reshape(-1, V, 2)
    pred = pred + 0.1 * lora_contrib
    return pred

train_losses = []
test_losses = []
print("\nTraining for 500 epochs (tracking train/test loss)...")
for epoch in range(500):
    optimizer.zero_grad()
    
    pred_train = compute_pred(Z_train, heads, lora_A, lora_B)
    train_loss = criterion(pred_train, Y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    train_losses.append(train_loss.item())
    
    with torch.no_grad():
        pred_test = compute_pred(Z_test, heads, lora_A, lora_B)
        test_loss = criterion(pred_test, Y_test_tensor)
        test_losses.append(test_loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:3d}: Train Loss = {train_loss.item():.6f}, Test Loss = {test_loss.item():.6f}")

print("\n" + "=" * 80)
print("TRAIN/TEST LOSS ANALYSIS")
print("=" * 80)
print(f"Final Train Loss: {train_losses[-1]:.6f}")
print(f"Final Test Loss: {test_losses[-1]:.6f}")
print(f"Min Train Loss: {min(train_losses):.6f} at epoch {np.argmin(train_losses)+1}")
print(f"Min Test Loss: {min(test_losses):.6f} at epoch {np.argmin(test_losses)+1}")

# Find epoch where test loss diverges from train loss
divergence_epoch = None
for i in range(100, len(test_losses)):
    if test_losses[i] > test_losses[i-1] * 1.01:  # 1% increase
        divergence_epoch = i + 1
        break

if divergence_epoch:
    print(f"\nTest loss starts diverging at epoch {divergence_epoch}")
    print(f"  Train loss at epoch {divergence_epoch}: {train_losses[divergence_epoch-1]:.6f}")
    print(f"  Test loss at epoch {divergence_epoch}: {test_losses[divergence_epoch-1]:.6f}")
else:
    print("\nTest loss tracks train loss well (no divergence)")

gap = test_losses[-1] - train_losses[-1]
print(f"\nTrain-Test Gap at end: {gap:.6f}")
if gap > 0.5:
    print("✗ SEVERE OVERFITTING - Gap is large!")
elif gap > 0.2:
    print("⚠ MODERATE OVERFITTING")
else:
    print("✓ Good generalization")
