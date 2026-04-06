#!/usr/bin/env python
"""Debug oracle training in detail."""
import sys
sys.path.insert(0, '/Users/taniyamac/Documents/FSU/course materials/Tensor research/FSDRNN/FSDRNN/code_taniya')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Generate simple data
np.random.seed(42)
torch.manual_seed(42)

n_train = 100
n_test = 50
latent_dim = 2
output_dim = 8

# Ground truth latent factors
z_train = np.random.randn(n_train, latent_dim)
z_test = np.random.randn(n_test, latent_dim)

# Generate Y from z using true coefficients
B_true = np.random.randn(latent_dim, output_dim)
Y_noise = np.random.randn(n_train, output_dim) * 0.3
y_train = z_train @ B_true + Y_noise

Y_noise_test = np.random.randn(n_test, output_dim) * 0.3
y_test = z_test @ B_true + Y_noise_test

print("=" * 80)
print("ORACLE TRAINING DEBUG")
print("=" * 80)
print(f"z_train shape: {z_train.shape}, y_train shape: {y_train.shape}")
print(f"y_train std: {y_train.std():.4f}, y_test std: {y_test.std():.4f}")

# Test: can we learn Y from z with simple linear model?
Z_train_torch = torch.tensor(z_train, dtype=torch.float32)
Y_train_torch = torch.tensor(y_train, dtype=torch.float32)
Z_test_torch = torch.tensor(z_test, dtype=torch.float32)
Y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Method 1: Direct linear regression
W = torch.linalg.lstsq(Z_train_torch, Y_train_torch).solution
pred_train = Z_train_torch @ W
pred_test = Z_test_torch @ W
mse_train_linear = ((pred_train - Y_train_torch) ** 2).mean()
mse_test_linear = ((pred_test - Y_test_torch) ** 2).mean()
print(f"\n1. Linear regression (ground truth):")
print(f"   Train MSE: {mse_train_linear.item():.6f}")
print(f"   Test MSE:  {mse_test_linear.item():.6f}")

# Method 2: Individual heads with LoRA
print(f"\n2. Neural network heads + LoRA:")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

heads = nn.ModuleList([
    nn.Linear(latent_dim, 1) for _ in range(output_dim)
]).to(device)

r_max = 6
lora_A = nn.Parameter(torch.randn(latent_dim, r_max, device=device) * 0.01)
lora_B = nn.Parameter(torch.randn(r_max, output_dim, device=device) * 0.01)

params = list(heads.parameters()) + [lora_A, lora_B]
optimizer = optim.Adam(params, lr=5e-4)
criterion = nn.MSELoss()

Z_train_device = Z_train_torch.to(device)
Y_train_device = Y_train_torch.to(device)
Z_test_device = Z_test_torch.to(device)
Y_test_device = Y_test_torch.to(device)

# Train
best_test_mse = float('inf')
for epoch in range(2000):
    optimizer.zero_grad()
    
    outputs = []
    for head in heads:
        outputs.append(head(Z_train_device))
    
    lora_contrib = Z_train_device @ lora_A @ lora_B
    pred = torch.cat(outputs, dim=1) + 0.1 * lora_contrib
    
    loss = criterion(pred, Y_train_device)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            pred_train = pred
            mse_train = criterion(pred_train, Y_train_device).item()
            
            # Test
            outputs_test = []
            for head in heads:
                outputs_test.append(head(Z_test_device))
            lora_contrib_test = Z_test_device @ lora_A @ lora_B
            pred_test = torch.cat(outputs_test, dim=1) + 0.1 * lora_contrib_test
            mse_test = criterion(pred_test, Y_test_device).item()
            
            print(f"   Epoch {epoch+1:4d} | train loss: {loss.item():.6f} | train MSE: {mse_train:.6f} | test MSE: {mse_test:.6f}")
            
            if mse_test < best_test_mse:
                best_test_mse = mse_test

print(f"\n   Final best test MSE: {best_test_mse:.6f}")

# Compare
print(f"\n" + "=" * 80)
print("COMPARISON:")
print(f"  Linear regression test MSE: {mse_test_linear.item():.6f}")
print(f"  NN + LoRA test MSE:         {best_test_mse:.6f}")
print(f"  Ratio (should be ~1):       {best_test_mse / mse_test_linear.item():.4f}")
print(f"  Y test std:                 {y_test.std():.4f}")
print("=" * 80)
