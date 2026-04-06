#!/usr/bin/env python3
"""Check if setup1 oracle also overfits."""
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
import setup1_linear as setup1

generate_synthetic_data = setup1.generate_synthetic_data
evaluate_mse = setup1.evaluate_mse

n_train = 50
n_test = 30
seed = 42

print("=" * 80)
print("CHECKING SETUP1 ORACLE FOR OVERFITTING")
print("=" * 80)

X_train, Y_train, z_train, beta = generate_synthetic_data(n_train, seed=seed)
X_test, Y_test, z_test, _ = generate_synthetic_data(n_test, seed=seed+1000)

V = Y_train.shape[1]  # 8

# Manually train oracle with train/test loss tracking (similar to setup6)
B_torch = torch.tensor(beta, dtype=torch.float32)
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test[:n_test], dtype=torch.float32)  
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test[:n_test], dtype=torch.float32)

# Oracle latents
Z_train = X_train_torch @ B_torch
Z_test = X_test_torch @ B_torch

# Create heads (same as setup1 oracle)
heads = nn.ModuleList([
    nn.Linear(2, 1) for _ in range(V)
])

r_max = min(V, 6)
lora_A = nn.Parameter(torch.randn(2, r_max) * 0.01)
lora_B = nn.Parameter(torch.randn(r_max, V) * 0.01)

params = list(heads.parameters()) + [lora_A, lora_B]
optimizer = optim.Adam(params, lr=5e-4)
criterion = nn.MSELoss()

def compute_pred(Z, heads, lora_A, lora_B):
    outputs = []
    for head in heads:
        outputs.append(head(Z))
    pred = torch.stack(outputs, dim=1)
    lora_contrib = Z @ lora_A @ lora_B
    pred = pred + 0.1 * lora_contrib
    return pred

train_losses = []
test_losses = []
print(f"\nTraining setup1 oracle for 3000 epochs (tracking train/test loss)...")
for epoch in range(3000):
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
    
    if (epoch + 1) % 300 == 0:
        print(f"  Epoch {epoch+1:4d}: Train Loss = {train_loss.item():.6f}, Test Loss = {test_loss.item():.6f}")

print("\n" + "=" * 80)
print("SETUP1 TRAIN/TEST LOSS ANALYSIS")
print("=" * 80)
print(f"Final Train Loss: {train_losses[-1]:.6f}")
print(f"Final Test Loss: {test_losses[-1]:.6f}")
print(f"Min Train Loss: {min(train_losses):.6f} at epoch {np.argmin(train_losses)+1}")
print(f"Min Test Loss: {min(test_losses):.6f} at epoch {np.argmin(test_losses)+1}")

gap = test_losses[-1] - train_losses[-1]
print(f"\nTrain-Test Gap at end: {gap:.6f}")
if gap > 0.5:
    print("✗ SEVERE OVERFITTING")
elif gap > 0.2:
    print("⚠ MODERATE OVERFITTING")
else:
    print("✓ Good generalization")

# Compare to actual test MSE we got
print("\nNote: Actual MSE values from earlier tests:")
print("  Setup1 FSDRNN MSE: 0.528")
print("  Setup1 Oracle MSE: 0.275")
print(f"  Simulated test loss at end (raw): {test_losses[-1]:.6f}")
print(f"  Setup1 oracle advantage ratio: ~1.92x")
