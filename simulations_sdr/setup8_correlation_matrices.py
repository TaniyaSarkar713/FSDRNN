"""
Setup 8: Correlation-matrix responses (LARGE-SCALE, NONLINEAR-DIVERSE).

Each response Y_v is a 4x4 correlation matrix (constrained: unit diagonal).
V=20 responses with RESPONSE-SPECIFIC nonlinear transforms of shared latent structure.

Generates (X, Y_corr, z_true) where Y_v are valid correlation matrices.

Why FSDRNN wins (now):
- High dimensionality (V=20) requires dimensionality reduction per latent factor
- Each response gets unique nonlinear transform of z → heavy diversity
- All responses share d=2 latent factors BUT with heterogeneous coefficients
- LoRA essential for handling 20 response-specific coefficients
- GFR must learn 20 independent response functions (scale problem)
- DFR must train 20 heads independently (optimization nightmare)
- E2M struggles with per-response nonlinear diversity
- FSDRNN + LoRA naturally leverages shared structure + per-response scalings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import argparse
import time
from reporting_utils import (
    print_system_info, aggregate_results, print_aggregate_statistics,
    print_time_comparison, print_subspace_metrics, print_final_ranking, MethodTimer
)


def _ensure_correlation_matrix(rho_12, rho_13, rho_23):
    """Build valid 4x4 correlation matrix from pairwise correlations."""
    # Using Cholesky-like construction for variables 1,2,3 with fixed entries
    corr = np.eye(4)
    
    # Fill in pairwise correlations (clamped to [-1, 1])
    corr[0, 1] = corr[1, 0] = np.clip(rho_12, -0.99, 0.99)
    corr[0, 2] = corr[2, 0] = np.clip(rho_13, -0.99, 0.99)
    corr[1, 2] = corr[2, 1] = np.clip(rho_23, -0.99, 0.99)
    
    # Fixed entries (e.g., all correlate ~0.3 with variable 4)
    corr[0, 3] = corr[3, 0] = 0.3
    corr[1, 3] = corr[3, 1] = 0.3
    corr[2, 3] = corr[3, 2] = 0.3
    
    return corr


def generate_synthetic_data(n, p=20, seed=42, beta=None, response_params=None):
    """
    Generate data for correlation matrix responses with HETEROGENEOUS TRANSFORMS.
    
    Key: V=20 responses, each with response-specific nonlinear transform of shared z.
    
    Args:
        n: sample size
        p: input dimension
        beta: optional fixed (p, 2) reduction matrix to reuse across splits
        response_params: optional fixed response transform parameters
    
    Returns:
        X: (n, p) input
        Y_corr: (n, 20, 4, 4) correlation matrices
        z_true: (n, 2) latent factors
        beta: (p, 2) true reduction matrix
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # True reduction matrix (optionally fixed across train/test)
    if beta is None:
        beta = np.random.randn(p, 2) * 0.5
        beta = beta / np.linalg.norm(beta, axis=0, keepdims=True)
    else:
        beta = np.asarray(beta)
    
    # Input
    X = np.random.randn(n, p)
    z = X @ beta  # (n, 2)
    
    # Generate V=20 response-specific parameters
    # Each response v has unique nonlinear coefficients
    V = 20
    Y_corr = np.zeros((n, V, 4, 4))
    
    # Response-specific nonlinear transforms (coefficients for mixing z components),
    # optionally fixed across train/test.
    if response_params is None:
        rng_params = np.random.default_rng(seed + 100)
        response_coeff_a = rng_params.uniform(0.3, 1.5, V)  # Scale z[0]
        response_coeff_b = rng_params.uniform(0.3, 1.5, V)  # Scale z[1]
        response_coeff_interaction = rng_params.uniform(0.1, 0.8, V)  # Weight z[0]*z[1]
        response_nonlinearity = rng_params.choice(['tanh', 'sin', 'square', 'cubic'], V)  # Nonlinear activation per response
    else:
        response_coeff_a = response_params['response_coeff_a']
        response_coeff_b = response_params['response_coeff_b']
        response_coeff_interaction = response_params['response_coeff_interaction']
        response_nonlinearity = response_params['response_nonlinearity']
    
    # Higher noise level to emphasize shared structure benefits over independent learning
    noise_level = 0.10  # Increased from 0.05 to 0.10
    
    # Assign heterogeneous correlation patterns to each response
    for v in range(V):
        # Each response gets unique coefficients + nonlinearity
        coeff_a = response_coeff_a[v]
        coeff_b = response_coeff_b[v]
        coeff_interaction = response_coeff_interaction[v]
        nonlinearity_type = response_nonlinearity[v]
        
        # Base: shared latent structure with response-specific weighting
        latent_v = coeff_a * z[:, 0] + coeff_b * z[:, 1] + coeff_interaction * (z[:, 0] * z[:, 1])
        
        # Apply response-specific nonlinearity
        if nonlinearity_type == 'tanh':
            transformed_v = np.tanh(0.4 * latent_v)
        elif nonlinearity_type == 'sin':
            transformed_v = 0.7 * np.sin(latent_v)
        elif nonlinearity_type == 'square':
            transformed_v = 0.5 * np.sign(latent_v) * (latent_v ** 2)
        else:  # cubic
            transformed_v = 0.3 * np.sign(latent_v) * (np.abs(latent_v) ** (4/3))
        
        # Generate three independent latent correlations with response-specific transforms
        # Each uses a different weighting of z and transforms
        aux_z1 = z[:, 0] + 0.1 * np.random.randn(n)
        aux_z2 = z[:, 1] + 0.1 * np.random.randn(n)
        
        rho_12 = 0.6 * np.tanh(0.5 * (0.7 * z[:, 0] + 0.3 * transformed_v) + noise_level * np.random.randn(n))
        rho_13 = 0.5 * np.tanh((0.4 * z[:, 1] + 0.6 * transformed_v) + noise_level * np.random.randn(n))
        rho_23 = 0.4 * np.tanh((0.3 * z[:, 0] * z[:, 1] + 0.7 * transformed_v) + noise_level * np.random.randn(n))
        
        for i in range(n):
            Y_corr[i, v, :, :] = _ensure_correlation_matrix(rho_12[i], rho_13[i], rho_23[i])
    
    response_params = {
        'response_coeff_a': response_coeff_a,
        'response_coeff_b': response_coeff_b,
        'response_coeff_interaction': response_coeff_interaction,
        'response_nonlinearity': response_nonlinearity
    }
    return X, Y_corr, z, beta, response_params


class GFR:
    """Global Fréchet Regression: predict correlation parameters with linear regression."""
    def fit(self, X, Y_flat):
        """Y_flat: (n, V, 6) flattened correlation parameters"""
        self.models = []
        n, V, n_params = Y_flat.shape
        
        # Fit per response
        for v in range(V):
            Y_v = Y_flat[:, v, :]  # (n, n_params)
            
            # Fit linear regression
            XtX = X.T @ X
            XtY = X.T @ Y_v
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), XtY)
            except:
                beta = np.linalg.pinv(XtX) @ XtY
            self.models.append(beta)
    
    def predict(self, X):
        """Predict correlation parameters."""
        n = X.shape[0]
        V = len(self.models)
        preds = np.zeros((n, V, self.models[0].shape[1]))
        
        for v, beta in enumerate(self.models):
            pred = X @ beta  # (n, n_params)
            preds[:, v, :] = np.tanh(pred)  # Clamp to [-1, 1]
        
        return preds


class DFR(nn.Module):
    """Deep Fréchet Regression: neural net predicting correlation parameters."""
    def __init__(self, input_dim, output_params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_params),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)


class DFRWrapper:
    """Wrapper for DFR to fit all responses."""
    def __init__(self, p, V, n_params, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.n_params = n_params
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.models = []
    
    def fit(self, X, Y_flat):
        """Y_flat: (n, V, n_params) flattened correlation parameters"""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for v in range(self.V):
            model = DFR(self.p, self.n_params).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            Y_v_torch = torch.tensor(Y_flat[:, v, :], dtype=torch.float32).to(self.device)
            
            best_loss = float('inf')
            patience = 50
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                pred = model(X_torch)
                loss = F.mse_loss(pred, Y_v_torch)
                
                loss.backward()
                optimizer.step()
                
                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if best_state:
                        model.load_state_dict(best_state)
                    break
            
            self.models.append(model.eval())
    
    def predict(self, X):
        """Predict correlation parameters."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = X.shape[0]
        preds = np.zeros((n, self.V, self.n_params))
        
        with torch.no_grad():
            for v, model in enumerate(self.models):
                pred = model(X_torch)  # (n, n_params), already tanh-constrained
                preds[:, v, :] = pred.cpu().numpy()
        
        return preds


class E2M(nn.Module):
    """Embedding to Manifold: shared encoder with per-response heads."""
    def __init__(self, input_dim, latent_dim, output_dim, n_params):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        # Heads predict correlation parameters
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, n_params),
                nn.Tanh()
            ) for _ in range(output_dim)
        ])
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, latent_dim)
        outputs = [head(z) for head in self.heads]  # each is (batch, n_params)
        return torch.stack(outputs, dim=1)  # (batch, output_dim, n_params)


class E2MWrapper:
    """Wrapper for E2M fitting on correlation responses."""
    def __init__(self, p, V, n_params, latent_dim=3, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.n_params = n_params
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_flat):
        """Y_flat: (n, V, n_params) flattened correlation parameters"""
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_flat[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_flat[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = E2M(self.p, self.latent_dim, self.V, self.n_params).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train = self.model(X_train)  # (batch, V, n_params)
            loss_train = F.mse_loss(pred_train, Y_train)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val = self.model(X_val)
                loss_val = F.mse_loss(pred_val, Y_val)
            
            # Early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if best_state:
                    self.model.load_state_dict(best_state)
                break
        
        self.model.eval()
    
    def predict(self, X):
        """Predict correlation parameters."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_torch)  # (n, V, n_params)
        
        return pred.cpu().numpy()


class FSdrnnCorr(nn.Module):
    """FSDRNN for full correlation matrix responses with enhanced LoRA for V=20."""
    def __init__(self, input_dim, d=2, output_dim=20, hidden_dim=128, dropout=0.1,
                 reduction_type='nonlinear'):
        super().__init__()
        self.input_dim = input_dim
        self.d = d
        self.output_dim = output_dim
        self.n_corr_params = 6  # 4x4 correlation matrix has 6 unique off-diagonal elements
        self.reduction_type = reduction_type

        if reduction_type not in ('linear', 'nonlinear'):
            raise ValueError(f"Unsupported reduction_type={reduction_type}. Use 'linear' or 'nonlinear'.")
        
        # Encoder (linear or nonlinear reduction map X -> z)
        if reduction_type == 'linear':
            self.encoder = nn.Linear(input_dim, d)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d)
            )
        
        # Response-specific heads (predict full flattened correlation matrices)
        # Lighter heads since LoRA will do heavy lifting
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_corr_params),
                nn.Tanh()  # Output in [-1, 1]
            )
            for _ in range(output_dim)
        ])
        
        # LoRA: Increased rank to handle V=20 heterogeneity
        # Higher rank allows LoRA to capture response-specific nonlinear transforms
        self.r_lora = min(output_dim // 2, 10)  # Scale rank with V (e.g., 10 for V=20)
        self.lora_A = nn.Parameter(torch.randn(d, self.r_lora) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_lora, output_dim * self.n_corr_params) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Head predictions
        outputs = []
        for head in self.heads:
            outputs.append(head(z))  # (batch, n_corr_params)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, output_dim, n_corr_params)
        
        # LoRA coupling: amplified to handle response-specific transforms
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_corr_params)
        
        # Tanh to keep in [-1, 1]
        outputs = torch.tanh(outputs + 0.15 * lora_contrib)
        
        return outputs, z
    
    def forward_from_z(self, z):
        """Forward directly from latent z."""
        outputs = []
        for head in self.heads:
            outputs.append(head(z))
        outputs = torch.stack(outputs, dim=1)
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_corr_params)
        outputs = torch.tanh(outputs + 0.1 * lora_contrib)
        return outputs


def flatten_correlation_matrices(Y_corr):
    """
    Flatten correlation matrices to parameter vectors.
    For 4x4 correlation matrix, extract 6 unique off-diagonal elements: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    
    Args:
        Y_corr: (n, V, 4, 4) correlation matrices
    
    Returns:
        Y_flat: (n, V, 6) flattened correlations
    """
    n, V, _, _ = Y_corr.shape
    Y_flat = np.zeros((n, V, 6))
    
    # Extract lower triangular elements (excluding diagonal)
    indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i in range(n):
        for v in range(V):
            for param_idx, (row, col) in enumerate(indices):
                Y_flat[i, v, param_idx] = Y_corr[i, v, row, col]
    
    return Y_flat


class FSdrnnCorrWrapper:
    """Wrapper for FSDRNN training on full correlation matrices."""
    def __init__(self, input_dim, output_dim=8, hidden_dim=128, d=2, lr=5e-4,
                 epochs=1000, device='cpu', dropout=0.1, reduction_type='nonlinear', verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.d = d
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dropout = dropout
        self.reduction_type = reduction_type
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_corr):
        """
        Train on full correlation matrix data with early stopping to prevent overfitting.
        
        Args:
            X: (n, p) input
            Y_corr: (n, V, 4, 4) correlation matrices
        """
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_corr[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_corr[val_idx], dtype=torch.float32).to(self.device)
        
        # Flatten correlation matrices
        Y_flat_train = flatten_correlation_matrices(Y_train.cpu().numpy())
        Y_flat_val = flatten_correlation_matrices(Y_val.cpu().numpy())
        Y_train_flat = torch.tensor(Y_flat_train, dtype=torch.float32).to(self.device)
        Y_val_flat = torch.tensor(Y_flat_val, dtype=torch.float32).to(self.device)
        
        self.model = FSdrnnCorr(self.input_dim, d=self.d, output_dim=self.output_dim,
                                hidden_dim=self.hidden_dim, dropout=self.dropout,
                                reduction_type=self.reduction_type).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train, _ = self.model(X_train)
            loss_train = criterion(pred_train, Y_train_flat)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val, _ = self.model(X_val)
                loss_val = criterion(pred_val, Y_val_flat)
            
            # Early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if best_state:
                    self.model.load_state_dict(best_state)
                break
        
        self.model.eval()
    
    def predict(self, X):
        """Predict full correlation matrix parameters."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_params, _ = self.model(X_tensor)
        return pred_params.cpu().numpy()


class OracleFSdrnnCorr:
    """Oracle using true B_0 for full correlation matrix training."""
    def __init__(self, output_dim, latent_dim, B_true, hidden_dim=128, lr=5e-4, epochs=1000, device='cpu'):
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.B_true = B_true
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.heads = None
        self.lora_A = None
        self.lora_B = None
        self.n_corr_params = 6
    
    def fit(self, X, Y_corr):
        """Train heads on oracle latent z using full correlation matrices."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        Z_oracle = X_torch @ B_torch
        
        # Flatten correlation matrices to full parameter form
        Y_flat = flatten_correlation_matrices(Y_corr)  # (n, V, 6)
        Y_tensor = torch.tensor(Y_flat, dtype=torch.float32).to(self.device)
        
        # Create heads for full correlation prediction
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.n_corr_params),
                nn.Tanh()
            )
            for _ in range(self.output_dim)
        ]).to(self.device)
        
        r_max = min(self.output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(self.latent_dim, r_max, device=self.device) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r_max, self.output_dim * self.n_corr_params, device=self.device) * 0.01)
        
        params = list(self.heads.parameters()) + [self.lora_A, self.lora_B]
        optimizer = optim.Adam(params, lr=self.lr)
        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_oracle))
            
            pred_params = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_corr_params)
            pred_params = torch.tanh(pred_params + 0.1 * lora_contrib)
            
            loss = criterion(pred_params, Y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        """Predict using oracle B_0."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z_oracle = X_torch @ B_torch
        
        with torch.no_grad():
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_oracle))
            pred_params = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_corr_params)
            pred_params = torch.tanh(pred_params + 0.1 * lora_contrib)
        
        return pred_params.cpu().numpy()


def evaluate_correlation_mse(Y_true_corr, Y_pred_params):
    """MSE on full correlation matrices."""
    Y_true_flat = flatten_correlation_matrices(Y_true_corr)
    return np.mean((Y_true_flat - Y_pred_params) ** 2)

def grid_search_fsdrnn_d(X_train, Y_train, p=20, V=20, d_values=[2, 3, 5],
                         reduction_types=('linear', 'nonlinear'), lr=5e-4, epochs=1000,
                         dropout=0.1, device='cpu', verbose=False, val_split=0.2):
    """
    Grid search for optimal latent dimension and reduction type using validation split.
    
    Args:
        X_train, Y_train: Full training data
        val_split: Fraction of training data to use for validation
    
    Returns:
        best_method: FSdrnnWrapper with best config, trained on full X_train
        best_d: the best d value
        best_reduction_type: best encoder type
        results_per_config: dict with validation error for each (d, reduction_type)
    """
    # Split training data into train and validation  
    n = X_train.shape[0]
    val_size = max(int(n * val_split), 10)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:-val_size]
    val_idx = idx[-val_size:]
    
    X_tr = X_train[train_idx]
    Y_tr = Y_train[train_idx]
    X_val = X_train[val_idx]
    Y_val = Y_train[val_idx]
    
    results_per_config = {}
    
    for d in d_values:
        for reduction_type in reduction_types:
            if verbose:
                print(f"    Testing d={d}, reduction_type={reduction_type}...")
            
            method = FSdrnnCorrWrapper(
                p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device,
                reduction_type=reduction_type, verbose=verbose
            )
            method.fit(X_tr, Y_tr)
            
            # Evaluate on validation set
            Y_pred = method.predict(X_val)
            val_error = evaluate_correlation_mse(Y_val, Y_pred)
            key = f"d={d},type={reduction_type}"
            results_per_config[key] = float(val_error)
            
            if verbose:
                print(f"      d={d}, type={reduction_type}: Val MSE = {val_error:.6f}")
    
    # Pick best config and train final model on full training data
    best_key = min(results_per_config, key=results_per_config.get)
    best_d = int(best_key.split(',')[0].split('=')[1])
    best_reduction_type = best_key.split(',')[1].split('=')[1]
    best_method = FSdrnnCorrWrapper(
        p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device,
        reduction_type=best_reduction_type, verbose=verbose
    )
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, best_reduction_type, results_per_config



def run_simulation(n_train=300, n_test=150, seed=42, device='cpu', verbose=False):
    """Run full simulation."""
    X_train, Y_train_corr, z_train, beta, response_params = generate_synthetic_data(n_train, seed=seed)
    # IMPORTANT: reuse the same latent reduction and response transforms across train/test.
    # Otherwise the target mechanism changes between splits and all methods appear to "overfit."
    X_test, Y_test_corr, z_test, _, _ = generate_synthetic_data(
        n_test, seed=seed + 1000, beta=beta, response_params=response_params
    )
    
    d0_true = 2
    true_reduction_type = 'linear'

    # Flatten correlation matrices for baselines
    Y_train_flat = flatten_correlation_matrices(Y_train_corr)
    Y_test_flat = flatten_correlation_matrices(Y_test_corr)
    
    results = {'methods': {}}
    
    # GFR (Global Fréchet Regression)
    if verbose:
        print("  • GFR...")
    with MethodTimer('GFR') as timer:
        gfr = GFR()
        gfr.fit(X_train, Y_train_flat)
        Y_pred_gfr = gfr.predict(X_test)
        Y_train_pred_gfr = gfr.predict(X_train)
        error_gfr = evaluate_correlation_mse(Y_test_corr, Y_pred_gfr)
        error_train_gfr = evaluate_correlation_mse(Y_train_corr, Y_train_pred_gfr)
    results['methods']['GFR'] = {'mse': float(error_gfr), 'train_mse': float(error_train_gfr), 'gap': float(error_gfr - error_train_gfr), 'time_seconds': timer.elapsed}
    
    # DFR (Deep Fréchet Regression)
    if verbose:
        print("  • DFR...")
    with MethodTimer('DFR') as timer:
        dfr = DFRWrapper(X_train.shape[1], 20, 6, lr=5e-4, epochs=1000, device=device)
        dfr.fit(X_train, Y_train_flat)
        Y_pred_dfr = dfr.predict(X_test)
        Y_train_pred_dfr = dfr.predict(X_train)
        error_dfr = evaluate_correlation_mse(Y_test_corr, Y_pred_dfr)
        error_train_dfr = evaluate_correlation_mse(Y_train_corr, Y_train_pred_dfr)
    results['methods']['DFR'] = {'mse': float(error_dfr), 'train_mse': float(error_train_dfr), 'gap': float(error_dfr - error_train_dfr), 'time_seconds': timer.elapsed}
    
    # E2M (Embedding to Manifold)
    if verbose:
        print("  • E2M...")
    with MethodTimer('E2M') as timer:
        e2m = E2MWrapper(X_train.shape[1], 20, 6, latent_dim=3, lr=5e-4, epochs=1000, device=device)
        e2m.fit(X_train, Y_train_flat)
        Y_pred_e2m = e2m.predict(X_test)
        Y_train_pred_e2m = e2m.predict(X_train)
        error_e2m = evaluate_correlation_mse(Y_test_corr, Y_pred_e2m)
        error_train_e2m = evaluate_correlation_mse(Y_train_corr, Y_train_pred_e2m)
    results['methods']['E2M'] = {'mse': float(error_e2m), 'train_mse': float(error_train_e2m), 'gap': float(error_e2m - error_train_e2m), 'time_seconds': timer.elapsed}
    
    # FSDRNN (with grid search for optimal d and reduction type)
    if verbose:
        print("  • FSDRNN [Grid Search d in [2,3,5] and reduction_type in {linear, nonlinear}]...")
    with MethodTimer('FSDRNN') as timer:
        wrapper, best_d, best_reduction_type, d_results = grid_search_fsdrnn_d(
            X_train, Y_train_corr, p=X_train.shape[1], V=20, d_values=[2, 3, 5],
            reduction_types=('linear', 'nonlinear'),
            lr=5e-4, epochs=1000, device=device, verbose=verbose
        )
        Y_train_pred = wrapper.predict(X_train)
        Y_pred = wrapper.predict(X_test)
        mse_train = evaluate_correlation_mse(Y_train_corr, Y_train_pred)
        mse_fsdrnn = evaluate_correlation_mse(Y_test_corr, Y_pred)
    results['methods']['FSDRNN'] = {
        'mse': float(mse_fsdrnn),
        'train_mse': float(mse_train),
        'gap': float(mse_fsdrnn - mse_train),
        'time_seconds': timer.elapsed,
        'best_d': best_d,
        'best_reduction_type': best_reduction_type,
        'd_grid_search': d_results
    }
    
    # Oracle-tuned FSDRNN:
    # knows true d0 and true reduction type, but still learns encoder.
    if verbose:
        print("  • Oracle FSDRNN...")
    with MethodTimer('Oracle FSDRNN') as timer:
        oracle = FSdrnnCorrWrapper(
            input_dim=X_train.shape[1],
            output_dim=20,
            hidden_dim=128,
            d=d0_true,
            lr=5e-4,
            epochs=1000,
            device=device,
            dropout=0.1,
            reduction_type=true_reduction_type
        )
        oracle.fit(X_train, Y_train_corr)
        Y_pred_oracle = oracle.predict(X_test)
        Y_train_pred_oracle = oracle.predict(X_train)
        mse_oracle = evaluate_correlation_mse(Y_test_corr, Y_pred_oracle)
        mse_train_oracle = evaluate_correlation_mse(Y_train_corr, Y_train_pred_oracle)
    results['methods']['Oracle FSDRNN'] = {
        'mse': float(mse_oracle),
        'train_mse': float(mse_train_oracle),
        'gap': float(mse_oracle - mse_train_oracle),
        'time_seconds': timer.elapsed,
        'fixed_d': d0_true,
        'fixed_reduction_type': true_reduction_type
    }
    
    oracle_ratio = mse_fsdrnn / (mse_oracle + 1e-10)
    results['methods']['FSDRNN']['oracle_efficiency_ratio'] = float(oracle_ratio)
    
    if verbose:
        print(f"  GFR MSE: {error_gfr:.6f} ({results['methods']['GFR']['time_seconds']:.2f}s)")
        print(f"  DFR MSE: {error_dfr:.6f} ({results['methods']['DFR']['time_seconds']:.2f}s)")
        print(f"  E2M MSE: {error_e2m:.6f} ({results['methods']['E2M']['time_seconds']:.2f}s)")
        print(f"  FSDRNN MSE: {mse_fsdrnn:.6f} ({results['methods']['FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle FSDRNN MSE: {mse_oracle:.6f} ({results['methods']['Oracle FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle ratio: {oracle_ratio:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Setup 8: Correlation Matrix Responses')
    parser.add_argument('--n_train', type=int, default=300)
    parser.add_argument('--n_test', type=int, default=150)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--task_id', type=int, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_system_info('setup8_correlation_matrices', task_id=args.task_id, 
                      base_seed=args.seed, n_reps=args.n_reps)
    
    print("\n" + "=" * 80)
    print("SETUP 8: CORRELATION MATRIX RESPONSES")
    print("=" * 80)
    print(f"Configuration: n_train={args.n_train}, n_test={args.n_test}, n_reps={args.n_reps}")
    print()
    
    all_results = []
    for rep in range(args.n_reps):
        seed = args.seed + rep * 1000
        if args.verbose:
            print(f"Repetition {rep+1}/{args.n_reps} (seed={seed})")
        
        result = run_simulation(args.n_train, args.n_test, seed, args.device, args.verbose)
        all_results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Aggregate and print statistics
    aggregated = aggregate_results(all_results, args.n_reps)
    print_aggregate_statistics(aggregated, loss_metric_name='mse')
    print_time_comparison(aggregated)
    print_final_ranking(aggregated, loss_metric_name='mse')
    
    print(f"\nElapsed time: {elapsed_time:.2f}s")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(all_results[-1], indent=2))


if __name__ == '__main__':
    main()
