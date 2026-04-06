#!/usr/bin/env python3
"""Debug oracle loss during training."""
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
OracleFSdrnnWrapper = setup6.OracleFSdrnnWrapper

n_train = 50
seed = 42

print("=" * 80)
print("DEBUGGING ORACLE LOSS DURING TRAINING")
print("=" * 80)

X_train, Y_train_quant, z_train, beta = generate_synthetic_data(n_train, seed=seed)
_, V, n_q = Y_train_quant.shape

# Extract parameters
Y_train_params = np.zeros((n_train, V, 2))
for i in range(n_train):
    for v in range(V):
        Y_train_params[i, v, 0] = np.median(Y_train_quant[i, v, :])
        Y_train_params[i, v, 1] = np.std(Y_train_quant[i, v, :]) + 0.1

# Manually train oracle with loss tracking
print(f"\nTraining oracle with loss tracking ({n_train} samples)...")
B_torch = torch.tensor(beta, dtype=torch.float32)
X_torch = torch.tensor(X_train, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train_params, dtype=torch.float32)

# Oracle latent
Z_oracle = X_torch @ B_torch  # Perfect encoding
print(f"Oracle Z shape: {Z_oracle.shape}")
print(f"Oracle Z mean: {Z_oracle.mean():.4f}, std: {Z_oracle.std():.4f}")
print(f"Y_params mean: {Y_tensor.mean():.4f}, std: {Y_tensor.std():.4f}")

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

# Track losses
train_losses = []
print("\nTraining for 200 epochs (tracking loss)...")
for epoch in range(200):
    optimizer.zero_grad()
    
    outputs = []
    for head in heads:
        outputs.append(head(Z_oracle))
    
    pred_params = torch.stack(outputs, dim=1)
    lora_contrib = Z_oracle @ lora_A @ lora_B
    lora_contrib = lora_contrib.reshape(-1, V, 2)
    pred_params = pred_params + 0.1 * lora_contrib
    
    loss = criterion(pred_params, Y_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")

print("\n" + "=" * 80)
print("LOSS ANALYSIS")
print("=" * 80)
print(f"Final loss: {train_losses[-1]:.6f}")
print(f"Initial loss: {train_losses[0]:.6f}")
print(f"Min loss: {min(train_losses):.6f} at epoch {np.argmin(train_losses)+1}")
print(f"Max loss: {max(train_losses):.6f} at epoch {np.argmax(train_losses)+1}")

if train_losses[-1] > train_losses[0]:
    print("\n✗ LOSS DIVERGING - Oracle heads not converging properly!")
elif train_losses[-1] > min(train_losses) * 1.1:
    print("\n⚠ LOSS OSCILLATING - Oracle training unstable, but not diverging")
else:
    print("\n✓ Loss converged normally")

# Check gradient magnitudes
print(f"\nParameter inspection:")
for i, head in enumerate(heads):
    if i == 0:  # Just check first head
        for name, param in head.named_parameters():
            if param.grad is not None:
                print(f"  Head 0 {name}: grad_mean={param.grad.mean().item():.6e}, grad_std={param.grad.std().item():.6e}")

print(f"  LoRA A: grad_mean={lora_A.grad.mean().item():.6e}, grad_std={lora_A.grad.std().item():.6e}")
print(f"  LoRA B: grad_mean={lora_B.grad.mean().item():.6e}, grad_std={lora_B.grad.std().item():.6e}")
