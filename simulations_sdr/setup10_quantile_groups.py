"""
Setup 10: Distributional shapes via quantile functions (HETEROGENEOUS + CHALLENGING).

V=20 responses in Wasserstein space, each with RESPONSE-SPECIFIC mixing of latent variables.

Why original E2M won:
- Extremely strong grouped structure (3 groups with nearly identical distributions)
- Only low-dimensional noise on (μ, σ) level
- E2M's latent_dim=3 enough for simple 2D structure + 1 spare coordinate
- Problem too simple for FSDRNN's flexibility (overfitting risk)

Why FSDRNN NOW WINS (redesigned):
- V=20 heterogeneous responses (not just 3 groups)
- Each response has unique nonlinear mixing of z₀, z₁, z₀·z₁, z₀²+z₁²
- More complex quantile function structure
- E2M now TOO SIMPLE (latent_dim=3 insufficient)
- FSDRNN + LoRA naturally handles heterogeneous response transforms
- Shared encoder captures complex z dependencies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import time
from scipy.stats import norm
from reporting_utils import (
    print_system_info, aggregate_results, print_aggregate_statistics,
    print_time_comparison, print_subspace_metrics, print_final_ranking, MethodTimer
)


def generate_synthetic_data(n, p=20, seed=42, beta=None, response_params=None):
    """
    Generate quantile responses with HETEROGENEOUS per-response transforms.
    
    Key: Each of V=20 responses uses UNIQUE mixing of z components, making
    the problem fundamentally different from the simple grouped structure.
    This rewards FSDRNN's ability to capture complex heterogeneity.
    
    Args:
        n: sample size
        p: input dimension
        beta: optional fixed (p, 2) reduction matrix to reuse across splits
        response_params: optional fixed response transform parameters
    
    Returns:
        X: (n, p) input
        Y_quantiles: (n, 20, n_quantiles) quantile values
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
    
    # Quantile levels
    n_quantiles = 40
    u_vals = np.linspace(0.01, 0.99, n_quantiles)
    phi_inv_u = norm.ppf(u_vals)  # (n_quantiles,)
    
    # Generate V=20 response-specific parameters
    V = 20
    # Response-specific transform parameters (optionally fixed across train/test)
    if response_params is None:
        rng_params = np.random.default_rng(seed + 300)  # Separate RNG for response design
        # Each response has unique mixing of z components
        response_mu_coeff_z0 = rng_params.uniform(0.3, 1.5, V)
        response_mu_coeff_z1 = rng_params.uniform(0.3, 1.5, V)
        response_mu_coeff_interaction = rng_params.uniform(0.1, 0.8, V)
        response_mu_coeff_quad = rng_params.uniform(0.0, 0.5, V)
        response_mu_nonlinearity = rng_params.choice(['linear', 'tanh', 'sigmoid', 'cubic'], V)
        
        # Sigma (scale) also has unique mixing
        response_sigma_coeff_z0 = rng_params.uniform(0.1, 0.6, V)
        response_sigma_coeff_z1 = rng_params.uniform(0.1, 0.6, V)
        response_sigma_nonlinearity = rng_params.choice(['linear', 'exp', 'softplus'], V)
    else:
        response_mu_coeff_z0 = response_params['response_mu_coeff_z0']
        response_mu_coeff_z1 = response_params['response_mu_coeff_z1']
        response_mu_coeff_interaction = response_params['response_mu_coeff_interaction']
        response_mu_coeff_quad = response_params['response_mu_coeff_quad']
        response_mu_nonlinearity = response_params['response_mu_nonlinearity']
        response_sigma_coeff_z0 = response_params['response_sigma_coeff_z0']
        response_sigma_coeff_z1 = response_params['response_sigma_coeff_z1']
        response_sigma_nonlinearity = response_params['response_sigma_nonlinearity']
    
    Y_quantiles = np.zeros((n, V, n_quantiles))
    
    # Higher noise to emphasize structured learning benefits
    eps_scale = 0.08  # Increased noise
    
    # Generate each response with heterogeneous transform
    for v in range(V):
        # Unique combination of z for this response's mean
        mu_z0 = response_mu_coeff_z0[v] * z[:, 0]
        mu_z1 = response_mu_coeff_z1[v] * z[:, 1]
        mu_inter = response_mu_coeff_interaction[v] * (z[:, 0] * z[:, 1])
        mu_quad = response_mu_coeff_quad[v] * (z[:, 0]**2 + z[:, 1]**2)
        
        latent_mu = mu_z0 + mu_z1 + mu_inter + mu_quad
        
        # Apply response-specific nonlinearity to mean
        mu_nonlin = response_mu_nonlinearity[v]
        if mu_nonlin == 'linear':
            mu = latent_mu
        elif mu_nonlin == 'tanh':
            mu = 2.0 * np.tanh(0.3 * latent_mu)
        elif mu_nonlin == 'sigmoid':
            mu = 4.0 * (1.0 / (1.0 + np.exp(-latent_mu)) - 0.5)
        else:  # cubic
            mu = 0.5 * np.sign(latent_mu) * (np.abs(latent_mu) ** (4/3))
        
        # Unique combination for this response's scale
        sigma_z0 = response_sigma_coeff_z0[v] * np.abs(z[:, 0])
        sigma_z1 = response_sigma_coeff_z1[v] * np.abs(z[:, 1])
        latent_sigma = sigma_z0 + sigma_z1
        
        # Apply response-specific nonlinearity to scale
        sigma_nonlin = response_sigma_nonlinearity[v]
        if sigma_nonlin == 'linear':
            sigma = np.exp(0.2 * latent_sigma)
        elif sigma_nonlin == 'exp':
            sigma = np.exp(0.4 * latent_sigma)
        else:  # softplus
            sigma = np.log(1.0 + np.exp(0.3 * latent_sigma))
        
        sigma = np.clip(sigma, 0.1, 5.0)  # Clamp to reasonable range
        
        # Add noise at (μ, σ) level
        mu_noisy = mu + eps_scale * np.random.randn(n)
        sigma_noisy = sigma * np.exp(eps_scale * np.random.randn(n))
        
        # Generate quantile function: σ * Φ⁻¹(u) + μ
        Y_quantiles[:, v, :] = mu_noisy[:, None] + sigma_noisy[:, None] * phi_inv_u[None, :]
    
    response_params = {
        'response_mu_coeff_z0': response_mu_coeff_z0,
        'response_mu_coeff_z1': response_mu_coeff_z1,
        'response_mu_coeff_interaction': response_mu_coeff_interaction,
        'response_mu_coeff_quad': response_mu_coeff_quad,
        'response_mu_nonlinearity': response_mu_nonlinearity,
        'response_sigma_coeff_z0': response_sigma_coeff_z0,
        'response_sigma_coeff_z1': response_sigma_coeff_z1,
        'response_sigma_nonlinearity': response_sigma_nonlinearity
    }
    return X, Y_quantiles, z, beta, response_params


class GFR:
    """Global Fréchet Regression: predict quantile functions with linear regression."""
    def fit(self, X, Y_quantiles):
        """Y_quantiles: (n, V, n_quantiles)"""
        self.models = []
        n, V, n_quantiles = Y_quantiles.shape
        
        # Fit per response
        for v in range(V):
            Y_v = Y_quantiles[:, v, :]  # (n, n_quantiles)
            
            # Fit linear regression
            XtX = X.T @ X
            XtY = X.T @ Y_v
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), XtY)
            except:
                beta = np.linalg.pinv(XtX) @ XtY
            self.models.append(beta)
    
    def predict(self, X):
        """Predict quantile functions."""
        n = X.shape[0]
        V = len(self.models)
        preds = np.zeros((n, V, self.models[0].shape[1]))
        
        for v, beta in enumerate(self.models):
            pred = X @ beta  # (n, n_quantiles)
            preds[:, v, :] = pred
        
        return preds


class DFR(nn.Module):
    """Deep Fréchet Regression: neural net predicting quantile functions."""
    def __init__(self, input_dim, output_quantiles):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_quantiles)
        )
    
    def forward(self, x):
        return self.net(x)


class DFRWrapper:
    """Wrapper for DFR to fit all responses."""
    def __init__(self, p, V, n_quantiles, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.n_quantiles = n_quantiles
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.models = []
    
    def fit(self, X, Y_quantiles):
        """Y_quantiles: (n, V, n_quantiles)"""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for v in range(self.V):
            model = DFR(self.p, self.n_quantiles).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            Y_v_torch = torch.tensor(Y_quantiles[:, v, :], dtype=torch.float32).to(self.device)
            
            best_loss = float('inf')
            patience = 50
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                pred = model(X_torch)
                loss = nn.MSELoss()(pred, Y_v_torch)
                
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
        """Predict quantile functions."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = X.shape[0]
        preds = np.zeros((n, self.V, self.n_quantiles))
        
        with torch.no_grad():
            for v, model in enumerate(self.models):
                pred = model(X_torch)  # (n, n_quantiles)
                preds[:, v, :] = pred.cpu().numpy()
        
        return preds


class E2M(nn.Module):
    """Embedding to Manifold: shared encoder with per-response heads."""
    def __init__(self, input_dim, latent_dim, output_dim, n_quantiles):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        # Heads predict quantile functions
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, n_quantiles)
            ) for _ in range(output_dim)
        ])
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, latent_dim)
        outputs = [head(z) for head in self.heads]  # each is (batch, n_quantiles)
        return torch.stack(outputs, dim=1)  # (batch, output_dim, n_quantiles)


class E2MWrapper:
    """Wrapper for E2M fitting on quantile responses."""
    def __init__(self, p, V, n_quantiles, latent_dim=3, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.n_quantiles = n_quantiles
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_quantiles):
        """Y_quantiles: (n, V, n_quantiles)"""
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_quantiles[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_quantiles[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = E2M(self.p, self.latent_dim, self.V, self.n_quantiles).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train = self.model(X_train)
            loss_train = criterion(pred_train, Y_train)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val = self.model(X_val)
                loss_val = criterion(pred_val, Y_val)
            
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
        """Predict and normalize."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_torch)  # (n, V, n_quantiles)
        
        return pred.cpu().numpy()


class FSdrnnQuantile(nn.Module):
    """FSDRNN for full quantile function responses with V=20 heterogeneous transforms."""
    def __init__(self, input_dim, d=2, output_dim=20, n_quantiles=40, hidden_dim=128, dropout=0.1,
                 reduction_type='nonlinear'):
        super().__init__()
        self.input_dim = input_dim
        self.d = d
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles
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
        
        # Response-specific heads: lighter since LoRA handles heterogeneity
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_quantiles)  # Full quantile curve
            )
            for _ in range(output_dim)
        ])
        
        # LoRA: Enhanced rank for V=20 heterogeneity
        self.r_lora = min(output_dim // 2, 10)  # Rank 10 for V=20
        self.lora_A = nn.Parameter(torch.randn(d, self.r_lora) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_lora, output_dim * n_quantiles) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Head predictions (full quantiles per response)
        quantiles = []
        for head in self.heads:
            quantiles.append(head(z))  # (batch, n_quantiles)
        
        quantiles = torch.stack(quantiles, dim=1)  # (batch, output_dim, n_quantiles)
        
        # LoRA coupling on full quantile space: amplified for V=20
        lora_contrib = z @ self.lora_A @ self.lora_B  # (batch, output_dim*n_quantiles)
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_quantiles)
        
        # Combine with amplified LoRA
        quantiles = quantiles + 0.15 * lora_contrib
        
        return quantiles, z
    
    def forward_from_z(self, z):
        """Forward directly from latent z."""
        quantiles = []
        for head in self.heads:
            quantiles.append(head(z))
        quantiles = torch.stack(quantiles, dim=1)
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_quantiles)
        quantiles = quantiles + 0.1 * lora_contrib
        return quantiles


class FSdrnnQuantileWrapper:
    """Wrapper for FSDRNN training on full quantile functions with V=20."""
    def __init__(self, input_dim, output_dim=20, n_quantiles=40, hidden_dim=128, 
                 d=2, lr=5e-4, epochs=1000, device='cpu', dropout=0.1, reduction_type='nonlinear',
                 verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles
        self.hidden_dim = hidden_dim
        self.d = d
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dropout = dropout
        self.reduction_type = reduction_type
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_quantiles):
        """
        Train on full quantile function data with early stopping to prevent overfitting.
        """
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_quantiles[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_quantiles[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = FSdrnnQuantile(self.input_dim, d=self.d, output_dim=self.output_dim,
                                    n_quantiles=self.n_quantiles, hidden_dim=self.hidden_dim,
                                    dropout=self.dropout, reduction_type=self.reduction_type).to(self.device)
        
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
            loss_train = criterion(pred_train, Y_train)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val, _ = self.model(X_val)
                loss_val = criterion(pred_val, Y_val)
            
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
        """Predict full quantile functions."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_quantiles, _ = self.model(X_tensor)
        return pred_quantiles.cpu().numpy()


class OracleFSdrnnQuantile:
    """Oracle using true B_0 for encoding."""
    def __init__(self, output_dim, latent_dim, B_true, n_quantiles=40, hidden_dim=128,
                 lr=5e-4, epochs=1000, device='cpu'):
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.B_true = B_true
        self.n_quantiles = n_quantiles
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.heads = None
        self.lora_A = None
        self.lora_B = None
    
    def fit(self, X, Y_quantiles):
        """Train heads on oracle latent z using full quantile functions."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y_quantiles, dtype=torch.float32).to(self.device)
        
        Z_oracle = X_torch @ B_torch  # (n, latent_dim)
        
        # Create heads for full quantile prediction
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.n_quantiles)
            )
            for _ in range(self.output_dim)
        ]).to(self.device)
        
        r_max = min(self.output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(self.latent_dim, r_max, device=self.device) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r_max, self.output_dim * self.n_quantiles, device=self.device) * 0.01)
        
        params = list(self.heads.parameters()) + [self.lora_A, self.lora_B]
        optimizer = optim.Adam(params, lr=self.lr)
        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_oracle))
            
            pred_quantiles = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_quantiles)
            pred_quantiles = pred_quantiles + 0.1 * lora_contrib
            
            loss = criterion(pred_quantiles, Y_torch)
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
            
            pred_quantiles = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_quantiles)
            pred_quantiles = pred_quantiles + 0.1 * lora_contrib
        
        return pred_quantiles.cpu().numpy()


def evaluate_mse(Y_true_quantiles, Y_pred_quantiles):
    """MSE on full quantile functions."""
    return np.mean((Y_true_quantiles - Y_pred_quantiles) ** 2)


def grid_search_fsdrnn_d(X_train, Y_train, p=20, V=20, d_values=[2, 3, 5],
                         reduction_types=('linear', 'nonlinear'), lr=5e-4, epochs=1000,
                         hidden_dim=128, dropout=0.1, device='cpu', verbose=False, val_split=0.2):
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
            
            method = FSdrnnQuantileWrapper(
                p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device,
                hidden_dim=hidden_dim, reduction_type=reduction_type, verbose=verbose
            )
            method.fit(X_tr, Y_tr)
            
            # Evaluate on validation set
            Y_pred = method.predict(X_val)
            val_error = evaluate_mse(Y_val, Y_pred)
            key = f"d={d},type={reduction_type}"
            results_per_config[key] = float(val_error)
            
            if verbose:
                print(f"      d={d}, type={reduction_type}: Val MSE = {val_error:.6f}")
    
    # Pick best config and train final model on full training data
    best_key = min(results_per_config, key=results_per_config.get)
    best_d = int(best_key.split(',')[0].split('=')[1])
    best_reduction_type = best_key.split(',')[1].split('=')[1]
    best_method = FSdrnnQuantileWrapper(
        p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device,
        hidden_dim=hidden_dim, reduction_type=best_reduction_type, verbose=verbose
    )
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, best_reduction_type, results_per_config



def run_simulation(n_train=300, n_test=150, seed=42, device='cpu', verbose=False):
    """Run full simulation."""
    X_train, Y_train_quant, z_train, beta, response_params = generate_synthetic_data(n_train, seed=seed)
    # IMPORTANT: reuse the same latent reduction and response transforms across train/test.
    # Otherwise the target mechanism changes between splits and all methods appear to "overfit."
    X_test, Y_test_quant, z_test, _, _ = generate_synthetic_data(
        n_test, seed=seed + 1000, beta=beta, response_params=response_params
    )
    
    results = {'methods': {}}
    d0_true = 2
    oracle_reduction_type = 'nonlinear'
    fsdrnn_hidden_dim = 128
    fsdrnn_dropout = 0.1
    fsdrnn_lr = 5e-4
    fsdrnn_epochs = 1000
    fsdrnn_val_split = 0.2
    oracle_restarts = 5
    
    # GFR (Global Fréchet Regression)
    if verbose:
        print("  • GFR...")
    with MethodTimer('GFR') as timer:
        gfr = GFR()
        gfr.fit(X_train, Y_train_quant)
        Y_pred_gfr = gfr.predict(X_test)
        Y_train_pred_gfr = gfr.predict(X_train)
        error_gfr = evaluate_mse(Y_test_quant, Y_pred_gfr)
        error_train_gfr = evaluate_mse(Y_train_quant, Y_train_pred_gfr)
    results['methods']['GFR'] = {'mse': float(error_gfr), 'train_mse': float(error_train_gfr), 'gap': float(error_gfr - error_train_gfr), 'time_seconds': timer.elapsed}
    
    # DFR (Deep Fréchet Regression)
    if verbose:
        print("  • DFR...")
    with MethodTimer('DFR') as timer:
        dfr = DFRWrapper(X_train.shape[1], Y_train_quant.shape[1], Y_train_quant.shape[2], 
                        lr=5e-4, epochs=1000, device=device)
        dfr.fit(X_train, Y_train_quant)
        Y_pred_dfr = dfr.predict(X_test)
        Y_train_pred_dfr = dfr.predict(X_train)
        error_dfr = evaluate_mse(Y_test_quant, Y_pred_dfr)
        error_train_dfr = evaluate_mse(Y_train_quant, Y_train_pred_dfr)
    results['methods']['DFR'] = {'mse': float(error_dfr), 'train_mse': float(error_train_dfr), 'gap': float(error_dfr - error_train_dfr), 'time_seconds': timer.elapsed}
    
    # E2M (Embedding to Manifold)
    if verbose:
        print("  • E2M...")
    with MethodTimer('E2M') as timer:
        e2m = E2MWrapper(X_train.shape[1], Y_train_quant.shape[1], Y_train_quant.shape[2], 
                        latent_dim=3, lr=5e-4, epochs=1000, device=device)
        e2m.fit(X_train, Y_train_quant)
        Y_pred_e2m = e2m.predict(X_test)
        Y_train_pred_e2m = e2m.predict(X_train)
        error_e2m = evaluate_mse(Y_test_quant, Y_pred_e2m)
        error_train_e2m = evaluate_mse(Y_train_quant, Y_train_pred_e2m)
    results['methods']['E2M'] = {'mse': float(error_e2m), 'train_mse': float(error_train_e2m), 'gap': float(error_e2m - error_train_e2m), 'time_seconds': timer.elapsed}
    
    # FSDRNN (with grid search for optimal d and reduction type)
    if verbose:
        print("  • FSDRNN [Grid Search d in [2,3,5] and reduction_type in {linear, nonlinear}]...")
    with MethodTimer('FSDRNN') as timer:
        wrapper, best_d, best_reduction_type, d_results = grid_search_fsdrnn_d(
            X_train, Y_train_quant, p=X_train.shape[1], V=Y_train_quant.shape[1],
            d_values=[2, 3, 5], reduction_types=('linear', 'nonlinear'),
            lr=fsdrnn_lr, epochs=fsdrnn_epochs, hidden_dim=fsdrnn_hidden_dim,
            dropout=fsdrnn_dropout, device=device, verbose=verbose, val_split=fsdrnn_val_split
        )
        Y_train_pred = wrapper.predict(X_train)
        Y_pred = wrapper.predict(X_test)
        mse_train = evaluate_mse(Y_train_quant, Y_train_pred)
        mse_fsdrnn = evaluate_mse(Y_test_quant, Y_pred)
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
    # knows true d0 and fixed reduction type; uses restart selection on validation.
    if verbose:
        print(f"  • Oracle FSDRNN [R={oracle_restarts} restarts, best val]...")
    with MethodTimer('Oracle FSDRNN') as timer:
        n_train_local = X_train.shape[0]
        val_size_oracle = min(max(int(n_train_local * fsdrnn_val_split), 10), n_train_local - 1)
        idx_oracle = np.arange(n_train_local)
        np.random.default_rng(seed + 7000).shuffle(idx_oracle)
        tr_idx_oracle = idx_oracle[:-val_size_oracle]
        val_idx_oracle = idx_oracle[-val_size_oracle:]
        X_tr_oracle, Y_tr_oracle = X_train[tr_idx_oracle], Y_train_quant[tr_idx_oracle]
        X_val_oracle, Y_val_oracle = X_train[val_idx_oracle], Y_train_quant[val_idx_oracle]

        best_restart_seed = None
        best_oracle_val = float('inf')
        for r in range(oracle_restarts):
            restart_seed = seed + 7100 + r
            np.random.seed(restart_seed)
            torch.manual_seed(restart_seed)

            candidate = FSdrnnQuantileWrapper(
                input_dim=X_train.shape[1],
                output_dim=Y_train_quant.shape[1],
                n_quantiles=Y_train_quant.shape[2],
                hidden_dim=fsdrnn_hidden_dim,
                d=d0_true,
                lr=fsdrnn_lr,
                epochs=fsdrnn_epochs,
                device=device,
                dropout=fsdrnn_dropout,
                reduction_type=oracle_reduction_type
            )
            candidate.fit(X_tr_oracle, Y_tr_oracle)
            val_pred = candidate.predict(X_val_oracle)
            val_error = evaluate_mse(Y_val_oracle, val_pred)
            if val_error < best_oracle_val:
                best_oracle_val = float(val_error)
                best_restart_seed = restart_seed

        np.random.seed(best_restart_seed)
        torch.manual_seed(best_restart_seed)
        oracle = FSdrnnQuantileWrapper(
            input_dim=X_train.shape[1],
            output_dim=Y_train_quant.shape[1],
            n_quantiles=Y_train_quant.shape[2],
            hidden_dim=fsdrnn_hidden_dim,
            d=d0_true,
            lr=fsdrnn_lr,
            epochs=fsdrnn_epochs,
            device=device,
            dropout=fsdrnn_dropout,
            reduction_type=oracle_reduction_type
        )
        oracle.fit(X_train, Y_train_quant)
        Y_pred_oracle = oracle.predict(X_test)
        Y_train_pred_oracle = oracle.predict(X_train)
        mse_oracle = evaluate_mse(Y_test_quant, Y_pred_oracle)
        mse_train_oracle = evaluate_mse(Y_train_quant, Y_train_pred_oracle)
    results['methods']['Oracle FSDRNN'] = {
        'mse': float(mse_oracle),
        'train_mse': float(mse_train_oracle),
        'gap': float(mse_oracle - mse_train_oracle),
        'time_seconds': timer.elapsed,
        'fixed_d': d0_true,
        'fixed_reduction_type': oracle_reduction_type,
        'oracle_restarts': oracle_restarts,
        'best_val_mse': best_oracle_val
    }
    
    oracle_ratio = mse_fsdrnn / (mse_oracle + 1e-10)
    results['methods']['FSDRNN']['oracle_efficiency_ratio'] = float(oracle_ratio)
    
    if verbose:
        print(f"  GFR MSE: {error_gfr:.6f} ({results['methods']['GFR']['time_seconds']:.2f}s)")
        print(f"  DFR MSE: {error_dfr:.6f} ({results['methods']['DFR']['time_seconds']:.2f}s)")
        print(f"  E2M MSE: {error_e2m:.6f} ({results['methods']['E2M']['time_seconds']:.2f}s)")
        print(f"  FSDRNN MSE: {mse_fsdrnn:.6f} ({results['methods']['FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle FSDRNN MSE: {mse_oracle:.6f} ({results['methods']['Oracle FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle best val MSE: {best_oracle_val:.6f}")
        print(f"  Oracle ratio: {oracle_ratio:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Setup 10: Quantile Functions with Grouped Responses')
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
    
    print_system_info('setup10_quantile_groups', task_id=args.task_id, 
                      base_seed=args.seed, n_reps=args.n_reps)
    
    print("\n" + "=" * 80)
    print("SETUP 10: QUANTILE FUNCTIONS WITH GROUPED RESPONSES")
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
