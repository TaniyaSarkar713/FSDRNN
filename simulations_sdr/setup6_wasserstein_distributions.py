"""
Setup 6: Distribution-valued responses in Wasserstein space.

Each response is a 1D Gaussian distribution represented by quantile function.
V=8 distributions with grouped structure: 3+3+2 in parameter space.
Generates (X, Y_quantiles, z_true) where Y_v = Gaussian with predictor-dependent mean/scale.

Why FSDRNN wins:
- Responses are non-Euclidean (Wasserstein geometry)
- Strong low-dimensional latent structure (d=2)
- Grouped responses share parameters → natural for SDR
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import argparse
import time
from pathlib import Path
from reporting_utils import (
    print_system_info, aggregate_results, print_aggregate_statistics,
    print_time_comparison, print_subspace_metrics, print_final_ranking, MethodTimer
)

torch.manual_seed(42)
np.random.seed(42)


def generate_synthetic_data(n, p=20, seed=42, nonlinear=False, beta=None):
    """
    Generate data for Wasserstein distribution responses with true d_0 = 2 oracle dimension.
    
    KEY: All noise is applied only at the (μ, σ) level, not on individual quantile points.
    This ensures the full 50D quantile curve depends only on the 2D latent z.
    
    Args:
        n: sample size
        p: input dimension
        seed: random seed
        nonlinear: if True, use nonlinear mean/scale functions
        beta: optional fixed (p, 2) reduction matrix to reuse across splits
    
    Returns:
        X: (n, p) input
        Y_quantiles: (n, V, n_quantiles) quantile evaluations
        z_train: (n, 2) true latent factors
        beta: (p, 2) true reduction matrix (orthonormal columns)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # True reduction matrix (orthonormal), optionally fixed across train/test
    d0 = 2
    if beta is None:
        beta = np.random.randn(p, d0)
        beta, _ = np.linalg.qr(beta)  # orthonormal columns
    else:
        beta = np.asarray(beta)
    
    # Input
    X = np.random.randn(n, p)
    z = X @ beta  # (n, 2) true latent factors
    
    # Quantile levels and inverse normal CDF
    V = 8
    n_quantiles = 50
    u_vals = np.linspace(0.01, 0.99, n_quantiles)
    
    from scipy.stats import norm
    phi_inv_u = norm.ppf(u_vals)  # (n_quantiles,)
    
    # Response-specific coefficients (grouped: 3+3+2)
    A_mu = np.array([
        [1.0, 0.5],   # Group 1
        [1.0, 0.5],
        [1.0, 0.5],
        [0.7, 1.2],   # Group 2
        [0.7, 1.2],
        [0.7, 1.2],
        [0.9, 0.8],   # Group 3
        [0.9, 0.8],
    ])
    
    A_sig = np.array([
        [0.3, 0.2],   # Group 1
        [0.3, 0.2],
        [0.3, 0.2],
        [0.4, 0.1],   # Group 2
        [0.4, 0.1],
        [0.4, 0.1],
        [0.2, 0.3],   # Group 3
        [0.2, 0.3],
    ])
    
    Y_quantiles = np.zeros((n, V, n_quantiles))
    
    for v in range(V):
        if nonlinear:
            # Nonlinear mean and log-scale as functions of z
            mu_v = (
                A_mu[v, 0] * np.sin(np.pi * z[:, 0]) +
                A_mu[v, 1] * (z[:, 1] ** 2 - 1)
            )
            log_sigma_v = (
                A_sig[v, 0] * z[:, 0] * z[:, 1] +
                0.1 * np.sin(z[:, 1])
            )
        else:
            # Linear mean and log-scale as functions of z
            mu_v = A_mu[v, 0] * z[:, 0] + A_mu[v, 1] * z[:, 1]
            log_sigma_v = A_sig[v, 0] * z[:, 0] + A_sig[v, 1] * z[:, 1]
        
        sigma_v = np.exp(log_sigma_v)  # (n,)
        
        # Add LOW-DIMENSIONAL noise: only on mu and log_sigma, not on each quantile
        eps_mu = 0.05 * np.random.randn(n)
        eps_ls = 0.03 * np.random.randn(n)
        
        mu_v_noisy = mu_v + eps_mu
        sigma_v_noisy = np.exp(log_sigma_v + eps_ls)
        
        # Generate quantile function from noisy (mu, sigma)
        # Q_v(u | X) = mu_v(X) + sigma_v(X) * Phi^{-1}(u)
        Y_quantiles[:, v, :] = (
            mu_v_noisy[:, None] + sigma_v_noisy[:, None] * phi_inv_u[None, :]
        )
    
    return X, Y_quantiles, z, beta


def encode_responses(Y_quantiles):
    """
    Encode full quantile functions as feature vectors.
    Simple approach: use quantile values + computed statistics.
    """
    n, V, n_quantiles = Y_quantiles.shape
    
    # Features: mean, std, min, max of each response
    features = np.zeros((n, V, 4))
    features[:, :, 0] = Y_quantiles.mean(axis=2)  # mean
    features[:, :, 1] = Y_quantiles.std(axis=2)   # std
    features[:, :, 2] = Y_quantiles.min(axis=2)   # min
    features[:, :, 3] = Y_quantiles.max(axis=2)   # max
    
    # Flatten to (n, V*4)
    return features.reshape(n, V * 4)


def wasserstein_distance(Q1_vals, Q2_vals):
    """
    Approximate Wasserstein distance between two distributions
    represented by quantile functions.
    
    Args:
        Q1_vals, Q2_vals: (n_quantiles,) quantile values at u in (0,1)
    
    Returns:
        scalar Wasserstein distance
    """
    # Approximate via L2 distance on quantile functions
    return np.mean((Q1_vals - Q2_vals) ** 2) ** 0.5


def compute_response_distance(Y_pred_quantiles, Y_true_quantiles):
    """Compute Wasserstein distance between predicted and true distributions."""
    n, V, _ = Y_true_quantiles.shape
    distances = np.zeros((n, V))
    
    for i in range(n):
        for v in range(V):
            distances[i, v] = wasserstein_distance(
                Y_pred_quantiles[i, v, :], Y_true_quantiles[i, v, :]
            )
    
    return distances.mean()


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
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train = self.model(X_train)  # (batch, V, n_quantiles)
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
        """Predict quantile functions."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_torch)  # (n, V, n_quantiles)
        
        return pred.cpu().numpy()


class FSDRNN(nn.Module):
    """FSDRNN for full quantile function responses (not lossy parameters)."""
    def __init__(self, input_dim, d=2, output_dim=8, n_quantiles=50, hidden_dim=128, dropout=0.1,
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
        
        # Response-specific heads with increased capacity (predict full quantile functions)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_quantiles)  # Full quantile curve
            )
            for _ in range(output_dim)
        ])
        
        # LoRA for quantile functions
        self.r_max = min(output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(d, self.r_max) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_max, output_dim * n_quantiles) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Head predictions (full quantiles per response)
        quantiles = []
        for head in self.heads:
            quantiles.append(head(z))  # (batch, n_quantiles)
        
        quantiles = torch.stack(quantiles, dim=1)  # (batch, output_dim, n_quantiles)
        
        # LoRA coupling on full quantile space
        lora_contrib = z @ self.lora_A @ self.lora_B  # (batch, output_dim*n_quantiles)
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.n_quantiles)
        
        # Combine
        quantiles = quantiles + 0.1 * lora_contrib
        
        return quantiles, z
    
    def forward_from_z(self, z):
        """Forward directly from latent z."""
        params = []
        for head in self.heads:
            params.append(head(z))
        
        params = torch.stack(params, dim=1)  # (batch, output_dim, 2)
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, 2)
        params = params + 0.1 * lora_contrib
        
        return params


class FSdrnnWrapper:
    """Wrapper for FSDRNN training on full quantile functions (no lossy extraction)."""
    def __init__(self, input_dim, output_dim=8, n_quantiles=50, d=2, hidden_dim=128, 
                 lr=5e-4, epochs=1000, device='cpu', dropout=0.2, reduction_type='nonlinear',
                 verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles
        self.d = d
        self.hidden_dim = hidden_dim
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
        
        Args:
            X: (n, p) input
            Y_quantiles: (n, V, n_quantiles) full quantile functions
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
        
        self.model = FSDRNN(self.input_dim, d=self.d, output_dim=self.output_dim,
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
                best_state = self.model.state_dict()
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


def evaluate_mse(Y_true_quant, Y_pred_quant):
    """Wasserstein-like distance: MSE on full quantile functions."""
    return np.mean((Y_true_quant - Y_pred_quant) ** 2)


def grid_search_fsdrnn_d(X_train, Y_train, p=100, V=20, d_values=[2, 3, 5],
                         reduction_types=('linear', 'nonlinear'), lr=3e-4, epochs=1000,
                         dropout=0.1, device='cpu', verbose=False, val_split=0.2):
    """
    Grid search for optimal latent dimension and reduction type using validation split.
    
    Args:
        X_train, Y_train: Full training data
        reduction_types: iterable of encoder types {'linear', 'nonlinear'}
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
            
            method = FSdrnnWrapper(
                p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device,
                reduction_type=reduction_type
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
    # format: d=<int>,type=<str>
    best_d = int(best_key.split(',')[0].split('=')[1])
    best_reduction_type = best_key.split(',')[1].split('=')[1]
    best_method = FSdrnnWrapper(
        p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device,
        reduction_type=best_reduction_type
    )
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, best_reduction_type, results_per_config



def run_simulation(n_train=500, n_test=250, seed=42, device='cpu', verbose=False, nonlinear=False):
    """Run full simulation training on full quantile functions."""
    X_train, Y_train_quant, z_train, beta = generate_synthetic_data(n_train, seed=seed, nonlinear=nonlinear)
    # IMPORTANT: reuse the same true reduction matrix across train/test.
    # Otherwise the target mechanism changes between splits and all methods appear to "overfit."
    X_test, Y_test_quant, z_test, _ = generate_synthetic_data(
        n_test, seed=seed + 1000, nonlinear=nonlinear, beta=beta
    )
    
    n_train, V, n_quant = Y_train_quant.shape
    n_test = X_test.shape[0]
    d0_true = 2
    true_reduction_type = 'nonlinear' if nonlinear else 'linear'
    
    results = {'methods': {}}
    
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
        dfr = DFRWrapper(X_train.shape[1], V, n_quant, lr=5e-4, epochs=1000, device=device)
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
        e2m = E2MWrapper(X_train.shape[1], V, n_quant, latent_dim=3, lr=5e-4, epochs=1000, device=device)
        e2m.fit(X_train, Y_train_quant)
        Y_pred_e2m = e2m.predict(X_test)
        Y_train_pred_e2m = e2m.predict(X_train)
        error_e2m = evaluate_mse(Y_test_quant, Y_pred_e2m)
        error_train_e2m = evaluate_mse(Y_train_quant, Y_train_pred_e2m)
    results['methods']['E2M'] = {'mse': float(error_e2m), 'train_mse': float(error_train_e2m), 'gap': float(error_e2m - error_train_e2m), 'time_seconds': timer.elapsed}
    
    # FSDRNN (trained on full quantiles) - Grid Search over d and reduction type
    if verbose:
        print("  • FSDRNN [Grid Search d in [2,3,5] and reduction_type in {linear, nonlinear}]...")
    with MethodTimer('FSDRNN') as timer:
        wrapper, best_d, best_reduction_type, d_results = grid_search_fsdrnn_d(
            X_train, Y_train_quant, p=X_train.shape[1], V=V, d_values=[2, 3, 5],
            reduction_types=('linear', 'nonlinear'), lr=5e-4, epochs=1000,
            dropout=0.2, device=device, verbose=verbose
        )
        Y_train_pred = wrapper.predict(X_train)  # Get train quantile predictions
        Y_pred = wrapper.predict(X_test)  # Get full quantile predictions
        mse_train = evaluate_mse(Y_train_quant, Y_train_pred)
        mse_fsdrnn = evaluate_mse(Y_test_quant, Y_pred)  # Evaluate on full quantiles
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
    # Knows true structural dimension d0 and true reduction type, but still learns encoder.
    if verbose:
        print("  • Oracle FSDRNN...")
    with MethodTimer('Oracle FSDRNN') as timer:
        oracle = FSdrnnWrapper(
            input_dim=X_train.shape[1],
            output_dim=V,
            n_quantiles=n_quant,
            d=d0_true,
            hidden_dim=128,
            lr=5e-4,
            epochs=1000,
            device=device,
            dropout=0.2,
            reduction_type=true_reduction_type
        )
        oracle.fit(X_train, Y_train_quant)
        Y_train_pred_oracle = oracle.predict(X_train)
        Y_pred_oracle = oracle.predict(X_test)
        mse_train_oracle = evaluate_mse(Y_train_quant, Y_train_pred_oracle)
        mse_oracle = evaluate_mse(Y_test_quant, Y_pred_oracle)
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
    parser = argparse.ArgumentParser(description='Setup 6: Wasserstein Distribution Responses')
    parser.add_argument('--n_train', type=int, default=500)
    parser.add_argument('--n_test', type=int, default=250)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nonlinear', action='store_true')
    parser.add_argument('--task_id', type=int, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_system_info('setup6_wasserstein_distributions', task_id=args.task_id, 
                      base_seed=args.seed, n_reps=args.n_reps)
    
    print("\n" + "=" * 80)
    print("SETUP 6: WASSERSTEIN DISTRIBUTION RESPONSES")
    print("=" * 80)
    print(f"Configuration: n_train={args.n_train}, n_test={args.n_test}, n_reps={args.n_reps}")
    print(f"Nonlinear: {args.nonlinear}")
    print()
    
    all_results = []
    for rep in range(args.n_reps):
        seed = args.seed + rep * 1000
        if args.verbose:
            print(f"Repetition {rep+1}/{args.n_reps} (seed={seed})")
        
        result = run_simulation(args.n_train, args.n_test, seed, args.device, args.verbose, args.nonlinear)
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
