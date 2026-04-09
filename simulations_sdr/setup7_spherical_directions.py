"""
Setup 7: Directional responses on the sphere.

Each response Y_v lies on the unit sphere S^2 in R^3.
V=8 responses grouped by direction: 3+3+2 near different sphere points.
Generates (X, Y_sphere, z_true) where Y_v are 3D unit vectors.

Why FSDRNN wins:
- Responses are non-Euclidean (spherical manifold geometry)
- Strong low-dimensional latent structure (d=2)
- Grouped responses concentrated near few shared directions
- Natural for SDR via grouped Fisher geometry
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import time
from scipy.spatial.distance import cosine
from reporting_utils import (
    print_system_info, aggregate_results, print_aggregate_statistics,
    print_time_comparison, print_subspace_metrics, print_final_ranking, MethodTimer
)


def generate_synthetic_data(n, p=20, seed=42, beta=None):
    """
    Generate data for spherical responses.
    
    Args:
        n: sample size
        p: input dimension
        beta: optional fixed (p, 2) reduction matrix to reuse across splits
    
    Returns:
        X: (n, p) input
        Y_sphere: (n, 8, 3) unit vectors on sphere
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
    
    # Latent angles
    theta1 = 0.8 * z[:, 0]  # (n,)
    theta2 = 0.8 * z[:, 1]
    
    # Mean direction m_1(X)
    def get_mean_direction(t1, t2):
        """Convert angles to 3D unit vector."""
        m = np.zeros((len(t1), 3))
        m[:, 0] = np.cos(t1) * np.cos(t2)
        m[:, 1] = np.sin(t1) * np.cos(t2)
        m[:, 2] = np.sin(t2)
        return m
    
    m1 = get_mean_direction(theta1, theta2)  # (n, 3)
    
    # Perturbed directions for other groups
    m2 = get_mean_direction(theta1 + 0.5, theta2 - 0.3)
    m3 = get_mean_direction(theta1 - 0.5, theta2 + 0.4)
    
    # Generate responses
    Y_sphere = np.zeros((n, 8, 3))
    sigma = 0.15  # noise level
    
    # Group 1 (v=0,1,2) near m1
    for v in range(3):
        noise = np.random.randn(n, 3) * sigma
        Y_tilde = m1 + noise
        Y_sphere[:, v, :] = Y_tilde / np.linalg.norm(Y_tilde, axis=1, keepdims=True)
    
    # Group 2 (v=3,4,5) near m2
    for v in range(3, 6):
        noise = np.random.randn(n, 3) * sigma
        Y_tilde = m2 + noise
        Y_sphere[:, v, :] = Y_tilde / np.linalg.norm(Y_tilde, axis=1, keepdims=True)
    
    # Group 3 (v=6,7) near m3
    for v in range(6, 8):
        noise = np.random.randn(n, 3) * sigma
        Y_tilde = m3 + noise
        Y_sphere[:, v, :] = Y_tilde / np.linalg.norm(Y_tilde, axis=1, keepdims=True)
    
    return X, Y_sphere, z, beta


class GFR:
    """Global Fréchet Regression: predict 3D vectors per response with linear regression."""
    def fit(self, X, Y_sphere):
        """Y_sphere: (n, V, 3)"""
        self.models = []
        n, V, _ = Y_sphere.shape
        
        # Fit per response
        for v in range(V):
            Y_v = Y_sphere[:, v, :]  # (n, 3)
            
            # Fit 3D linear regression
            XtX = X.T @ X
            XtY = X.T @ Y_v
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), XtY)
            except:
                beta = np.linalg.pinv(XtX) @ XtY
            self.models.append(beta)
    
    def predict(self, X):
        """Predict and normalize to unit sphere."""
        n = X.shape[0]
        V = len(self.models)
        preds = np.zeros((n, V, 3))
        
        for v, beta in enumerate(self.models):
            pred = X @ beta  # (n, 3)
            pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
            preds[:, v, :] = pred
        
        return preds


class DFR(nn.Module):
    """Deep Fréchet Regression: predict 3D vector per response."""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x):
        return self.net(x)  # (batch, 3)


class DFRWrapper:
    """Wrapper for DFR to fit all responses."""
    def __init__(self, p, V, hidden_dim=32, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.models = []
    
    def fit(self, X, Y_sphere):
        """Y_sphere: (n, V, 3)"""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for v in range(self.V):
            model = DFR(self.p, self.hidden_dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            Y_v_torch = torch.tensor(Y_sphere[:, v, :], dtype=torch.float32).to(self.device)
            
            best_loss = float('inf')
            patience = 50
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                pred = model(X_torch)
                pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
                
                # Cosine distance loss
                cosine_sim = (pred_norm * Y_v_torch).sum(dim=1)
                loss = (1 - cosine_sim).mean()
                
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
        """Predict and normalize."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = X.shape[0]
        preds = np.zeros((n, self.V, 3))
        
        with torch.no_grad():
            for v, model in enumerate(self.models):
                pred = model(X_torch)  # (n, 3)
                pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
                preds[:, v, :] = pred_norm.cpu().numpy()
        
        return preds


class E2M(nn.Module):
    """Embedding to Manifold: shared latent encoder with per-response heads."""
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        # Heads predict 3D vectors
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 3)
            ) for _ in range(output_dim)
        ])
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, latent_dim)
        outputs = [head(z) for head in self.heads]  # each is (batch, 3)
        return torch.stack(outputs, dim=1)  # (batch, output_dim, 3)


class E2MWrapper:
    """Wrapper for E2M fitting on spherical responses."""
    def __init__(self, p, V, latent_dim=3, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_sphere):
        """Y_sphere: (n, V, 3)"""
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_sphere[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_sphere[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = E2M(self.p, self.latent_dim, self.V).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train = self.model(X_train)  # (batch, V, 3)
            pred_train = pred_train / (torch.norm(pred_train, dim=2, keepdim=True) + 1e-8)
            cosine_sim_train = (pred_train * Y_train).sum(dim=2)
            loss_train = (1 - cosine_sim_train).mean()
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val = self.model(X_val)
                pred_val = pred_val / (torch.norm(pred_val, dim=2, keepdim=True) + 1e-8)
                cosine_sim_val = (pred_val * Y_val).sum(dim=2)
                loss_val = (1 - cosine_sim_val).mean()
            
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
            pred = self.model(X_torch)  # (n, V, 3)
            pred = pred / (torch.norm(pred, dim=2, keepdim=True) + 1e-8)
        
        return pred.cpu().numpy()


class FSdrnnSphere(nn.Module):
    """FSDRNN for spherical responses."""
    def __init__(self, input_dim, d=2, output_dim=8, hidden_dim=32, dropout=0.1,
                 reduction_type='nonlinear'):
        super().__init__()
        self.input_dim = input_dim
        self.d = d
        self.output_dim = output_dim
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
        
        # Response-specific heads (predict 3D directions)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)
            )
            for _ in range(output_dim)
        ])
        
        # LoRA
        self.r_max = min(output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(d, self.r_max) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_max, output_dim * 3) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Head predictions
        outputs = []
        for head in self.heads:
            outputs.append(head(z))  # (batch, 3)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, output_dim, 3)
        
        # LoRA coupling
        lora_contrib = z @ self.lora_A @ self.lora_B  # (batch, output_dim*3)
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, 3)
        
        # Combine and normalize
        outputs = outputs + 0.1 * lora_contrib
        outputs = outputs / (torch.norm(outputs, dim=2, keepdim=True) + 1e-8)
        
        return outputs, z
    
    def forward_from_z(self, z):
        """Forward directly from latent z for oracle."""
        outputs = []
        for head in self.heads:
            outputs.append(head(z))
        
        outputs = torch.stack(outputs, dim=1)
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, 3)
        outputs = outputs + 0.1 * lora_contrib
        outputs = outputs / (torch.norm(outputs, dim=2, keepdim=True) + 1e-8)
        
        return outputs


class FSdrnnSphereWrapper:
    """Wrapper for FSDRNN training on sphere."""
    def __init__(self, input_dim, output_dim=8, hidden_dim=32, d=2, lr=5e-4,
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
    
    def fit(self, X, Y_sphere):
        """
        Train on sphere data with early stopping to prevent overfitting.
        
        Args:
            X: (n, p) input
            Y_sphere: (n, V, 3) unit vectors
        """
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_sphere[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_sphere[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = FSdrnnSphere(self.input_dim, d=self.d, output_dim=self.output_dim,
                                  hidden_dim=self.hidden_dim, dropout=self.dropout,
                                  reduction_type=self.reduction_type).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train, _ = self.model(X_train)
            pred_train = pred_train / (torch.norm(pred_train, dim=2, keepdim=True) + 1e-8)
            cosine_sim_train = (pred_train * Y_train).sum(dim=2)
            loss_train = (1 - cosine_sim_train).mean()
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val, _ = self.model(X_val)
                pred_val = pred_val / (torch.norm(pred_val, dim=2, keepdim=True) + 1e-8)
                cosine_sim_val = (pred_val * Y_val).sum(dim=2)
                loss_val = (1 - cosine_sim_val).mean()
            
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
        """Predict sphere directions."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(X_tensor)
            pred = pred / (torch.norm(pred, dim=2, keepdim=True) + 1e-8)
        return pred.cpu().numpy()


class OracleFSdrnnSphere:
    """Oracle using true B_0."""
    def __init__(self, output_dim, latent_dim, B_true, hidden_dim=32, lr=5e-4, epochs=1000, device='cpu'):
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
    
    def fit(self, X, Y_sphere):
        """Train heads on oracle latent z."""
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_sphere[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_sphere[val_idx], dtype=torch.float32).to(self.device)
        
        Z_train = X_train @ B_torch  # (n_train, latent_dim)
        Z_val = X_val @ B_torch  # (n_val, latent_dim)
        
        # Create heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 3)
            )
            for _ in range(self.output_dim)
        ]).to(self.device)
        
        r_max = min(self.output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(self.latent_dim, r_max, device=self.device) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r_max, self.output_dim * 3, device=self.device) * 0.01)
        
        params = list(self.heads.parameters()) + [self.lora_A, self.lora_B]
        optimizer = optim.Adam(params, lr=self.lr)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_train))
            
            pred_train = torch.stack(outputs, dim=1)
            lora_contrib = Z_train @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, 3)
            pred_train = pred_train + 0.1 * lora_contrib
            pred_train = pred_train / (torch.norm(pred_train, dim=2, keepdim=True) + 1e-8)
            
            cosine_sim_train = (pred_train * Y_train).sum(dim=2)
            loss_train = (1 - cosine_sim_train).mean()
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                outputs_val = []
                for head in self.heads:
                    outputs_val.append(head(Z_val))
                pred_val = torch.stack(outputs_val, dim=1)
                lora_contrib_val = Z_val @ self.lora_A @ self.lora_B
                lora_contrib_val = lora_contrib_val.reshape(-1, self.output_dim, 3)
                pred_val = pred_val + 0.1 * lora_contrib_val
                pred_val = pred_val / (torch.norm(pred_val, dim=2, keepdim=True) + 1e-8)
                cosine_sim_val = (pred_val * Y_val).sum(dim=2)
                loss_val = (1 - cosine_sim_val).mean()
            
            # Early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                best_state = {
                    'heads': [h.state_dict() for h in self.heads],
                    'lora_A': self.lora_A.data.clone(),
                    'lora_B': self.lora_B.data.clone()
                }
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if best_state:
                    # Restore best state
                    for i, head in enumerate(self.heads):
                        head.load_state_dict(best_state['heads'][i])
                    self.lora_A.data = best_state['lora_A']
                    self.lora_B.data = best_state['lora_B']
                break
    
    def predict(self, X):
        """Predict using oracle B_0."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z_oracle = X_torch @ B_torch
        
        with torch.no_grad():
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_oracle))
            pred = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, 3)
            pred = pred + 0.1 * lora_contrib
            pred = pred / (torch.norm(pred, dim=2, keepdim=True) + 1e-8)
        
        return pred.cpu().numpy()


def evaluate_angular_distance(Y_true, Y_pred):
    """Angular distance on sphere (geodesic distance)."""
    n, V, _ = Y_true.shape
    distances = np.zeros((n, V))
    
    for i in range(n):
        for v in range(V):
            # Cosine distance ensures values in [0, pi]
            dot_prod = np.dot(Y_true[i, v], Y_pred[i, v])
            dot_prod = np.clip(dot_prod, -1, 1)
            distances[i, v] = np.arccos(dot_prod)
    
    return distances.mean()


def grid_search_fsdrnn_d(X_train, Y_train, p=20, V=15, d_values=[2, 3, 5],
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
            
            method = FSdrnnSphereWrapper(
                p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device,
                reduction_type=reduction_type, verbose=verbose
            )
            method.fit(X_tr, Y_tr)
            
            # Evaluate on validation set
            Y_pred = method.predict(X_val)
            val_error = evaluate_angular_distance(Y_val, Y_pred)
            key = f"d={d},type={reduction_type}"
            results_per_config[key] = float(val_error)
            
            if verbose:
                print(f"      d={d}, type={reduction_type}: Val MSE = {val_error:.6f}")
    
    # Pick best config and train final model on full training data
    best_key = min(results_per_config, key=results_per_config.get)
    best_d = int(best_key.split(',')[0].split('=')[1])
    best_reduction_type = best_key.split(',')[1].split('=')[1]
    best_method = FSdrnnSphereWrapper(
        p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device,
        reduction_type=best_reduction_type, verbose=verbose
    )
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, best_reduction_type, results_per_config



def run_simulation(n_train=300, n_test=150, seed=42, device='cpu', verbose=False):
    """Run full simulation."""
    X_train, Y_train, z_train, beta = generate_synthetic_data(n_train, seed=seed)
    # IMPORTANT: reuse the same true reduction matrix across train/test.
    # Otherwise the target mechanism changes between splits and all methods appear to "overfit."
    X_test, Y_test, z_test, _ = generate_synthetic_data(n_test, seed=seed + 1000, beta=beta)
    
    results = {'methods': {}}
    d0_true = 2
    true_reduction_type = 'linear'
    
    # GFR (Global Fréchet Regression)
    if verbose:
        print("  • GFR...")
    with MethodTimer('GFR') as timer:
        gfr = GFR()
        gfr.fit(X_train, Y_train)
        Y_pred_gfr = gfr.predict(X_test)
        Y_train_pred_gfr = gfr.predict(X_train)
        error_gfr = evaluate_angular_distance(Y_test, Y_pred_gfr)
        error_train_gfr = evaluate_angular_distance(Y_train, Y_train_pred_gfr)
    results['methods']['GFR'] = {'angular_error': float(error_gfr), 'train_angular_error': float(error_train_gfr), 'gap': float(error_gfr - error_train_gfr), 'time_seconds': timer.elapsed}
    
    # DFR (Deep Fréchet Regression)
    if verbose:
        print("  • DFR...")
    with MethodTimer('DFR') as timer:
        dfr = DFRWrapper(X_train.shape[1], 8, hidden_dim=32, lr=5e-4, epochs=1000, device=device)
        dfr.fit(X_train, Y_train)
        Y_pred_dfr = dfr.predict(X_test)
        Y_train_pred_dfr = dfr.predict(X_train)
        error_dfr = evaluate_angular_distance(Y_test, Y_pred_dfr)
        error_train_dfr = evaluate_angular_distance(Y_train, Y_train_pred_dfr)
    results['methods']['DFR'] = {'angular_error': float(error_dfr), 'train_angular_error': float(error_train_dfr), 'gap': float(error_dfr - error_train_dfr), 'time_seconds': timer.elapsed}
    
    # E2M (Embedding to Manifold)
    if verbose:
        print("  • E2M...")
    with MethodTimer('E2M') as timer:
        e2m = E2MWrapper(X_train.shape[1], 8, latent_dim=3, lr=5e-4, epochs=1000, device=device)
        e2m.fit(X_train, Y_train)
        Y_pred_e2m = e2m.predict(X_test)
        Y_train_pred_e2m = e2m.predict(X_train)
        error_e2m = evaluate_angular_distance(Y_test, Y_pred_e2m)
        error_train_e2m = evaluate_angular_distance(Y_train, Y_train_pred_e2m)
    results['methods']['E2M'] = {'angular_error': float(error_e2m), 'train_angular_error': float(error_train_e2m), 'gap': float(error_e2m - error_train_e2m), 'time_seconds': timer.elapsed}
    
    # FSDRNN (with grid search for optimal d and reduction type)
    if verbose:
        print("  • FSDRNN [Grid Search d in [2,3,5] and reduction_type in {linear, nonlinear}]...")
    with MethodTimer('FSDRNN') as timer:
        wrapper, best_d, best_reduction_type, d_results = grid_search_fsdrnn_d(
            X_train, Y_train, p=X_train.shape[1], V=8, d_values=[2, 3, 5],
            reduction_types=('linear', 'nonlinear'),
            lr=5e-4, epochs=1000, device=device, verbose=verbose
        )
        Y_train_pred = wrapper.predict(X_train)
        Y_pred = wrapper.predict(X_test)
        error_train = evaluate_angular_distance(Y_train, Y_train_pred)
        error_fsdrnn = evaluate_angular_distance(Y_test, Y_pred)
    results['methods']['FSDRNN'] = {
        'angular_error': float(error_fsdrnn),
        'train_angular_error': float(error_train),
        'gap': float(error_fsdrnn - error_train),
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
        oracle = FSdrnnSphereWrapper(
            input_dim=X_train.shape[1],
            output_dim=8,
            hidden_dim=32,
            d=d0_true,
            lr=5e-4,
            epochs=1000,
            device=device,
            dropout=0.1,
            reduction_type=true_reduction_type
        )
        oracle.fit(X_train, Y_train)
        Y_pred_oracle = oracle.predict(X_test)
        Y_train_pred_oracle = oracle.predict(X_train)
        error_oracle = evaluate_angular_distance(Y_test, Y_pred_oracle)
        error_train_oracle = evaluate_angular_distance(Y_train, Y_train_pred_oracle)
    results['methods']['Oracle FSDRNN'] = {
        'angular_error': float(error_oracle),
        'train_angular_error': float(error_train_oracle),
        'gap': float(error_oracle - error_train_oracle),
        'time_seconds': timer.elapsed,
        'fixed_d': d0_true,
        'fixed_reduction_type': true_reduction_type
    }
    
    oracle_ratio = error_fsdrnn / (error_oracle + 1e-10)
    results['methods']['FSDRNN']['oracle_efficiency_ratio'] = float(oracle_ratio)
    
    if verbose:
        print(f"  GFR angular error: {error_gfr:.6f} ({results['methods']['GFR']['time_seconds']:.2f}s)")
        print(f"  DFR angular error: {error_dfr:.6f} ({results['methods']['DFR']['time_seconds']:.2f}s)")
        print(f"  E2M angular error: {error_e2m:.6f} ({results['methods']['E2M']['time_seconds']:.2f}s)")
        print(f"  FSDRNN angular error: {error_fsdrnn:.6f} ({results['methods']['FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle FSDRNN angular error: {error_oracle:.6f} ({results['methods']['Oracle FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle ratio: {oracle_ratio:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Setup 7: Spherical Directional Responses')
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
    
    print_system_info('setup7_spherical_directions', task_id=args.task_id, 
                      base_seed=args.seed, n_reps=args.n_reps)
    
    print("\n" + "=" * 80)
    print("SETUP 7: SPHERICAL DIRECTIONAL RESPONSES")
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
    print_aggregate_statistics(aggregated, loss_metric_name='angular_error')
    print_time_comparison(aggregated)
    print_final_ranking(aggregated, loss_metric_name='angular_error')
    
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
