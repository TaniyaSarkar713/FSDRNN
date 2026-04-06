#!/usr/bin/env python3
"""Check if B_true is actually useful for predicting quantiles."""
import sys, os
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

n_train = 100
n_test = 50
seed = 42

print("=" * 80)
print("DIAGNOSTIC: Is B_true useful for predicting quantiles?")
print("=" * 80)

X_train, Y_train_quant, z_train, beta = generate_synthetic_data(n_train, seed=seed)
X_test, Y_test_quant, z_test, _ = generate_synthetic_data(n_test, seed=seed+1000)

V = Y_train_quant.shape[1]
n_q = Y_train_quant.shape[2]

print(f"\nData shapes:")
print(f"  X_train: {X_train.shape}, Y_train: {Y_train_quant.shape}")
print(f"  X_test: {X_test.shape}, Y_test: {Y_test_quant.shape}")

# Test 1: Oracle encoding (Z_true = X @ B_true)
print("\n" + "-" * 80)
print("Test 1: Oracle encoding")
Z_train_true = X_train @ beta
Z_test_true = X_test @ beta
print(f"  Z_train_true: mean={Z_train_true.mean():.4f}, std={Z_train_true.std():.4f}")
print(f"  Z_test_true: mean={Z_test_true.mean():.4f}, std={Z_test_true.std():.4f}")

# Train simple predictor on Z_true
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
Y_train_torch = torch.tensor(Y_train_quant, dtype=torch.float32)
Y_test_torch = torch.tensor(Y_test_quant, dtype=torch.float32)
Z_train_true_torch = torch.tensor(Z_train_true, dtype=torch.float32)
Z_test_true_torch = torch.tensor(Z_test_true, dtype=torch.float32)

# Simple linear models to predict Y from Z_true
model_z_true = nn.Linear(2, 8*50)
optimizer = optim.Adam(model_z_true.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    pred = model_z_true(Z_train_true_torch).reshape(-1, 8, 50)
    loss = criterion(pred, Y_train_torch)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred_test = model_z_true(Z_test_true_torch).reshape(-1, 8, 50)
    mse_z_true = criterion(pred_test, Y_test_torch).item()
print(f"  MSE predicting Y from Z_true via linear: {mse_z_true:.6f}")

# Test 2: Random encoding
print("\n" + "-" * 80)
print("Test 2: Random encoding (baseline)")
B_random = np.random.randn(X_train.shape[1], 2)
B_random = B_random / np.linalg.norm(B_random, axis=0, keepdims=True)
Z_train_random = X_train @ B_random
Z_train_random_torch = torch.tensor(Z_train_random, dtype=torch.float32)
Z_test_random = X_test @ B_random
Z_test_random_torch = torch.tensor(Z_test_random, dtype=torch.float32)

model_z_random = nn.Linear(2, 8*50)
optimizer = optim.Adam(model_z_random.parameters(), lr=1e-3)

for epoch in range(500):
    optimizer.zero_grad()
    pred = model_z_random(Z_train_random_torch).reshape(-1, 8, 50)
    loss = criterion(pred, Y_train_torch)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred_test = model_z_random(Z_test_random_torch).reshape(-1, 8, 50)
    mse_z_random = criterion(pred_test, Y_test_torch).item()

print(f"  MSE predicting Y from Z_random via linear: {mse_z_random:.6f}")

# Test 3: Learned encoding (from X directly)
print("\n" + "-" * 80)
print("Test 3: Learned encoding (full model X -> Z -> Y)")
model_learned = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 2),  # Learned Z
)
head = nn.Linear(2, 8*50)
full_model = nn.Sequential(*list(model_learned.children()) + [head])

optimizer = optim.Adam(full_model.parameters(), lr=1e-3)
for epoch in range(500):
    optimizer.zero_grad()
    pred = full_model(X_train_torch).reshape(-1, 8, 50)
    loss = criterion(pred, Y_train_torch)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred_test = full_model(X_test_torch).reshape(-1, 8, 50)
    mse_learned = criterion(pred_test, Y_test_torch).item()

print(f"  MSE predicting Y from X via learned encoder: {mse_learned:.6f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"MSE: Z_true = {mse_z_true:.6f}")
print(f"MSE: Z_random = {mse_z_random:.6f}")
print(f"MSE: Z_learned = {mse_learned:.6f}")
print(f"\nRatio (Z_true / Z_learned): {mse_z_true / (mse_learned + 1e-10):.4f}")
if mse_z_true < mse_learned:
    print(f"✓ Z_true is BETTER than learned ({mse_z_true:.4f} < {mse_learned:.4f})")
else:
    print(f"✗ Z_true is WORSE than learned ({mse_z_true:.4f} > {mse_learned:.4f})")
    print(f"  → Suggests learned encoder finds better representation!")
