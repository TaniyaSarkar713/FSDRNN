"""
Setup 9: Compositional responses on simplex (HETEROGENEOUS MULTI-RESPONSE).

Each response Y_v is a K=4 category composition (probability simplex).
V=20 responses, EACH WITH RESPONSE-SPECIFIC NONLINEAR TRANSFORM OF z.

Generates (X, Y_simplex, z_true) where Y_v are 4D probability vectors.

Why FSDRNN WINS (redesigned):
- Responses depend ONLY on z via heterogeneous latent transforms
- Each response has unique: coefficients + nonlinearity + mixing structure
- Direct X→softmax NO LONGER EFFECTIVE (requires learning all transforms)
- V=20: Independent methods (GFR/DFR) must learn 20 complex response functions
- FSDRNN + LoRA: shared encoder captures z, per-response heads handle heterogeneity
- More noise emphasizes: structured sharing > independent learning

Key fix from user:
  "Redesign so response depends only on XB₀ through a structure
   that really rewards shared multi-response bottlenecks"
   → This does exactly that!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import argparse
import time
from reporting_utils import (
    print_system_info, aggregate_results, print_aggregate_statistics,
    print_time_comparison, print_subspace_metrics, print_final_ranking, MethodTimer
)


def generate_synthetic_data(n, p=20, seed=42, beta=None, response_params=None):
    """
    Generate data for simplex responses with HETEROGENEOUS response-specific transforms.
    
    Key design: Each of V=20 responses depends on z through a unique nonlinear transform.
    This makes it impossible for GFR to learn well (requires memorizing 20 functions),
    while FSDRNN's shared encoder + per-response heads naturally handle it.
    
    Args:
        n: sample size
        p: input dimension
        beta: optional fixed (p, 2) reduction matrix to reuse across splits
        response_params: optional fixed response transform parameters
    
    Returns:
        X: (n, p) input
        Y_simplex: (n, 20, 4) probability vectors (sum to 1)
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
    
    # Generate V=20 response-specific transformation parameters
    V = 20
    # Response-specific transform parameters (optionally fixed across train/test)
    if response_params is None:
        rng_params = np.random.default_rng(seed + 200)  # Separate RNG for response design
        # Each response v has unique: coefficients + nonlinearity + mixing
        response_coeff_z0 = rng_params.uniform(0.5, 1.5, V)  # z[0] coefficient
        response_coeff_z1 = rng_params.uniform(0.5, 1.5, V)  # z[1] coefficient
        response_coeff_interaction = rng_params.uniform(0.2, 0.8, V)  # z[0]*z[1]
        response_coeff_quad = rng_params.uniform(0.1, 0.6, V)  # z[0]² + z[1]²
        response_nonlinearity = rng_params.choice(['tanh', 'sin', 'sigmoid', 'elu'], V)  # per-response
    else:
        response_coeff_z0 = response_params['response_coeff_z0']
        response_coeff_z1 = response_params['response_coeff_z1']
        response_coeff_interaction = response_params['response_coeff_interaction']
        response_coeff_quad = response_params['response_coeff_quad']
        response_nonlinearity = response_params['response_nonlinearity']
    
    Y_simplex = np.zeros((n, V, 4))
    
    # Noise level: increased to emphasize shared structure benefits
    noise_level = 0.12  # Higher noise = more value in shared bottleneck
    
    # Generate each response with response-specific transform
    for v in range(V):
        # Unique combination of z components for this response
        coeff_z0 = response_coeff_z0[v]
        coeff_z1 = response_coeff_z1[v]
        coeff_inter = response_coeff_interaction[v]
        coeff_quad = response_coeff_quad[v]
        nonlin_type = response_nonlinearity[v]
        
        # Latent transform: response-specific mixing of z components
        latent_transform = (coeff_z0 * z[:, 0] + 
                           coeff_z1 * z[:, 1] + 
                           coeff_inter * (z[:, 0] * z[:, 1]) +
                           coeff_quad * (z[:, 0]**2 + z[:, 1]**2))
        
        # Apply response-specific nonlinearity
        if nonlin_type == 'tanh':
            transformed = np.tanh(0.4 * latent_transform)
        elif nonlin_type == 'sin':
            transformed = 0.7 * np.sin(latent_transform)
        elif nonlin_type == 'sigmoid':
            transformed = 2.0 * (1.0 / (1.0 + np.exp(-latent_transform)) - 0.5)
        else:  # elu
            transformed = np.where(latent_transform > 0, 0.3 * latent_transform, 0.3 * (np.exp(latent_transform) - 1))
        
        # Create 4D logit vector: each dimension also mixes z differently
        # This makes it impossible to learn by direct X regression alone
        logit_mix1 = 0.6 * z[:, 0] + 0.3 * transformed + noise_level * np.random.randn(n)
        logit_mix2 = 0.4 * z[:, 1] + 0.5 * transformed + noise_level * np.random.randn(n)
        logit_mix3 = 0.7 * (z[:, 0] * z[:, 1]) + 0.2 * transformed + noise_level * np.random.randn(n)
        logit_mix4 = 0.3 * (z[:, 0]**2 + z[:, 1]**2) + 0.6 * transformed + noise_level * np.random.randn(n)
        
        logits = np.column_stack([logit_mix1, logit_mix2, logit_mix3, logit_mix4])  # (n, 4)
        
        # Convert to probabilities via softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        Y_simplex[:, v, :] = probs
    
    response_params = {
        'response_coeff_z0': response_coeff_z0,
        'response_coeff_z1': response_coeff_z1,
        'response_coeff_interaction': response_coeff_interaction,
        'response_coeff_quad': response_coeff_quad,
        'response_nonlinearity': response_nonlinearity
    }
    return X, Y_simplex, z, beta, response_params


class GFR:
    """Global Fréchet Regression: predict simplex compositions with linear regression."""
    def fit(self, X, Y_simplex):
        """Y_simplex: (n, V, K) probability vectors"""
        self.models = []
        n, V, K = Y_simplex.shape
        
        # Fit per response
        for v in range(V):
            Y_v = Y_simplex[:, v, :]  # (n, K)
            
            # Fit linear regression on probabilities
            XtX = X.T @ X
            XtY = X.T @ Y_v
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), XtY)
            except:
                beta = np.linalg.pinv(XtX) @ XtY
            self.models.append(beta)
    
    def predict(self, X):
        """Predict probability vectors."""
        n = X.shape[0]
        V = len(self.models)
        K = self.models[0].shape[1]
        preds = np.zeros((n, V, K))
        
        for v, beta in enumerate(self.models):
            pred_logits = X @ beta  # (n, K)
            # Convert to probabilities via softmax
            exp_logits = np.exp(pred_logits - pred_logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            preds[:, v, :] = probs
        
        return preds


class DFR(nn.Module):
    """Deep Fréchet Regression: neural net predicting simplex logits."""
    def __init__(self, input_dim, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, K)
        )
    
    def forward(self, x):
        return self.net(x)


class DFRWrapper:
    """Wrapper for DFR to fit all responses."""
    def __init__(self, p, V, K, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.K = K
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.models = []
    
    def fit(self, X, Y_simplex):
        """Y_simplex: (n, V, K) probability vectors"""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for v in range(self.V):
            model = DFR(self.p, self.K).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            Y_v_torch = torch.tensor(Y_simplex[:, v, :], dtype=torch.float32).to(self.device)
            
            best_loss = float('inf')
            patience = 50
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                logits = model(X_torch)
                # KL divergence loss
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, Y_v_torch, reduction='batchmean')
                
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
        """Predict simplex probabilities."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = X.shape[0]
        preds = np.zeros((n, self.V, self.K))
        
        with torch.no_grad():
            for v, model in enumerate(self.models):
                logits = model(X_torch)  # (n, K)
                probs = F.softmax(logits, dim=1)  # (n, K)
                preds[:, v, :] = probs.cpu().numpy()
        
        return preds


class E2M(nn.Module):
    """Embedding to Manifold: shared encoder with per-response heads."""
    def __init__(self, input_dim, latent_dim, output_dim, K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        # Heads predict logits
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, K)
            ) for _ in range(output_dim)
        ])
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, latent_dim)
        outputs = [head(z) for head in self.heads]  # each is (batch, K)
        return torch.stack(outputs, dim=1)  # (batch, output_dim, K)


class E2MWrapper:
    """Wrapper for E2M fitting on simplex responses."""
    def __init__(self, p, V, K, latent_dim=3, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.K = K
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_simplex):
        """Y_simplex: (n, V, K) probability vectors"""
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_simplex[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_simplex[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = E2M(self.p, self.latent_dim, self.V, self.K).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            logits_train = self.model(X_train)  # (batch, V, K)
            log_probs_train = F.log_softmax(logits_train, dim=2)
            loss_train = F.kl_div(log_probs_train.reshape(-1, self.K), Y_train.reshape(-1, self.K), reduction='batchmean')
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                logits_val = self.model(X_val)
                log_probs_val = F.log_softmax(logits_val, dim=2)
                loss_val = F.kl_div(log_probs_val.reshape(-1, self.K), Y_val.reshape(-1, self.K), reduction='batchmean')
            
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
        """Predict simplex probabilities."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_torch)  # (n, V, K)
            probs = F.softmax(logits, dim=2)  # (n, V, K)
        
        return probs.cpu().numpy()


class FSdrnnSimplex(nn.Module):
    """FSDRNN for simplex (compositional) responses with enhanced LoRA for V=20."""
    def __init__(self, input_dim, d=2, output_dim=20, K=4, hidden_dim=32, dropout=0.1,
                 reduction_type='nonlinear'):
        super().__init__()
        self.input_dim = input_dim
        self.d = d
        self.output_dim = output_dim
        self.K = K
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
        
        # Response-specific heads (predict K logits)
        # Lighter heads since LoRA will handle heterogeneity
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, K)
            )
            for _ in range(output_dim)
        ])
        
        # LoRA: Increased rank to handle V=20 response-specific transforms
        self.r_lora = min(output_dim // 2, 10)  # Rank 10 for V=20
        self.lora_A = nn.Parameter(torch.randn(d, self.r_lora) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_lora, output_dim * K) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Head predictions (logits)
        outputs = []
        for head in self.heads:
            outputs.append(head(z))  # (batch, K)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, output_dim, K)
        
        # LoRA coupling: amplified to handle response-specific transforms
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.K)
        
        # Combine logits with amplified LoRA
        logits = outputs + 0.15 * lora_contrib
        
        # Convert to probabilities via softmax
        probs = F.softmax(logits, dim=2)
        
        return probs, z
    
    def forward_from_z(self, z):
        """Forward directly from latent z."""
        outputs = []
        for head in self.heads:
            outputs.append(head(z))
        outputs = torch.stack(outputs, dim=1)
        lora_contrib = z @ self.lora_A @ self.lora_B
        lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.K)
        logits = outputs + 0.1 * lora_contrib
        probs = F.softmax(logits, dim=2)
        return probs


class FSdrnnSimplexWrapper:
    """Wrapper for FSDRNN training on simplex with V=20."""
    def __init__(self, input_dim, output_dim=20, K=4, hidden_dim=32, d=2, lr=5e-4,
                 epochs=1000, device='cpu', dropout=0.1, reduction_type='nonlinear', verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K
        self.hidden_dim = hidden_dim
        self.d = d
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dropout = dropout
        self.reduction_type = reduction_type
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y_simplex):
        """
        Train on simplex data with early stopping to prevent overfitting.
        
        Args:
            X: (n, p) input
            Y_simplex: (n, V, K) probability vectors
        """
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_simplex[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y_simplex[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = FSdrnnSimplex(self.input_dim, d=self.d, output_dim=self.output_dim, K=self.K,
                                   hidden_dim=self.hidden_dim, dropout=self.dropout,
                                   reduction_type=self.reduction_type).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train, _ = self.model(X_train)
            log_pred_train = torch.log(pred_train + 1e-10)
            loss_train = criterion(log_pred_train.reshape(-1, self.K), Y_train.reshape(-1, self.K))
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val, _ = self.model(X_val)
                log_pred_val = torch.log(pred_val + 1e-10)
                loss_val = criterion(log_pred_val.reshape(-1, self.K), Y_val.reshape(-1, self.K))
            
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
        """Predict simplex probabilities."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_probs, _ = self.model(X_tensor)
        return pred_probs.cpu().numpy()


class OracleFSdrnnSimplex:
    """Oracle using true B_0."""
    def __init__(self, output_dim, latent_dim, B_true, K=4, hidden_dim=32, lr=5e-4, epochs=1000, device='cpu'):
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.B_true = B_true
        self.K = K
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.heads = None
        self.lora_A = None
        self.lora_B = None
    
    def fit(self, X, Y_simplex):
        """Train heads on oracle latent z."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y_simplex, dtype=torch.float32).to(self.device)
        
        Z_oracle = X_torch @ B_torch
        
        # Create heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.K)
            )
            for _ in range(self.output_dim)
        ]).to(self.device)
        
        r_max = min(self.output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(self.latent_dim, r_max, device=self.device) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r_max, self.output_dim * self.K, device=self.device) * 0.01)
        
        params = list(self.heads.parameters()) + [self.lora_A, self.lora_B]
        optimizer = optim.Adam(params, lr=self.lr)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_oracle))
            
            logits = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.K)
            logits = logits + 0.1 * lora_contrib
            probs = F.softmax(logits, dim=2)
            
            log_probs = torch.log(probs + 1e-10)
            loss = criterion(log_probs.reshape(-1, self.K), Y_torch.reshape(-1, self.K))
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
            logits = torch.stack(outputs, dim=1)
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            lora_contrib = lora_contrib.reshape(-1, self.output_dim, self.K)
            logits = logits + 0.1 * lora_contrib
            probs = F.softmax(logits, dim=2)
        
        return probs.cpu().numpy()


def evaluate_kl_divergence(Y_true, Y_pred):
    """KL divergence between true and predicted probabilities."""
    Y_true = np.clip(Y_true, 1e-10, 1)
    Y_pred = np.clip(Y_pred, 1e-10, 1)
    return np.mean(np.sum(Y_true * (np.log(Y_true) - np.log(Y_pred)), axis=2))


def grid_search_fsdrnn_d(X_train, Y_train, p=10, V=20, d_values=[2, 3, 5],
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
            
            method = FSdrnnSimplexWrapper(
                p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device,
                reduction_type=reduction_type, verbose=verbose
            )
            method.fit(X_tr, Y_tr)
            
            # Evaluate on validation set
            Y_pred = method.predict(X_val)
            val_error = evaluate_kl_divergence(Y_val, Y_pred)
            key = f"d={d},type={reduction_type}"
            results_per_config[key] = float(val_error)
            
            if verbose:
                print(f"      d={d}, type={reduction_type}: Val MSE = {val_error:.6f}")
    
    # Pick best config and train final model on full training data
    best_key = min(results_per_config, key=results_per_config.get)
    best_d = int(best_key.split(',')[0].split('=')[1])
    best_reduction_type = best_key.split(',')[1].split('=')[1]
    best_method = FSdrnnSimplexWrapper(
        p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device,
        reduction_type=best_reduction_type, verbose=verbose
    )
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, best_reduction_type, results_per_config



def run_simulation(n_train=300, n_test=150, seed=42, device='cpu', verbose=False):
    """Run full simulation."""
    X_train, Y_train, z_train, beta, response_params = generate_synthetic_data(n_train, seed=seed)
    # IMPORTANT: reuse the same latent reduction and response transforms across train/test.
    # Otherwise the target mechanism changes between splits and all methods appear to "overfit."
    X_test, Y_test, z_test, _, _ = generate_synthetic_data(
        n_test, seed=seed + 1000, beta=beta, response_params=response_params
    )
    
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
        error_gfr = evaluate_kl_divergence(Y_test, Y_pred_gfr)
        error_train_gfr = evaluate_kl_divergence(Y_train, Y_train_pred_gfr)
    results['methods']['GFR'] = {'kl_divergence': float(error_gfr), 'train_kl_divergence': float(error_train_gfr), 'gap': float(error_gfr - error_train_gfr), 'time_seconds': timer.elapsed}
    
    # DFR (Deep Fréchet Regression)
    if verbose:
        print("  • DFR...")
    with MethodTimer('DFR') as timer:
        dfr = DFRWrapper(X_train.shape[1], 20, 4, lr=5e-4, epochs=1000, device=device)
        dfr.fit(X_train, Y_train)
        Y_pred_dfr = dfr.predict(X_test)
        Y_train_pred_dfr = dfr.predict(X_train)
        error_dfr = evaluate_kl_divergence(Y_test, Y_pred_dfr)
        error_train_dfr = evaluate_kl_divergence(Y_train, Y_train_pred_dfr)
    results['methods']['DFR'] = {'kl_divergence': float(error_dfr), 'train_kl_divergence': float(error_train_dfr), 'gap': float(error_dfr - error_train_dfr), 'time_seconds': timer.elapsed}
    
    # E2M (Embedding to Manifold)
    if verbose:
        print("  • E2M...")
    with MethodTimer('E2M') as timer:
        e2m = E2MWrapper(X_train.shape[1], 20, 4, latent_dim=3, lr=5e-4, epochs=1000, device=device)
        e2m.fit(X_train, Y_train)
        Y_pred_e2m = e2m.predict(X_test)
        Y_train_pred_e2m = e2m.predict(X_train)
        error_e2m = evaluate_kl_divergence(Y_test, Y_pred_e2m)
        error_train_e2m = evaluate_kl_divergence(Y_train, Y_train_pred_e2m)
    results['methods']['E2M'] = {'kl_divergence': float(error_e2m), 'train_kl_divergence': float(error_train_e2m), 'gap': float(error_e2m - error_train_e2m), 'time_seconds': timer.elapsed}
    
    # FSDRNN (with grid search for optimal d and reduction type)
    if verbose:
        print("  • FSDRNN [Grid Search d in [2,3,5] and reduction_type in {linear, nonlinear}]...")
    with MethodTimer('FSDRNN') as timer:
        wrapper, best_d, best_reduction_type, d_results = grid_search_fsdrnn_d(
            X_train, Y_train, p=X_train.shape[1], V=20, d_values=[2, 3, 5],
            reduction_types=('linear', 'nonlinear'),
            lr=5e-4, epochs=1000, device=device, verbose=verbose
        )
        Y_train_pred = wrapper.predict(X_train)
        Y_pred = wrapper.predict(X_test)
        kl_train = evaluate_kl_divergence(Y_train, Y_train_pred)
        kl_fsdrnn = evaluate_kl_divergence(Y_test, Y_pred)
    results['methods']['FSDRNN'] = {
        'kl_divergence': float(kl_fsdrnn),
        'train_kl_divergence': float(kl_train),
        'gap': float(kl_fsdrnn - kl_train),
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
        oracle = FSdrnnSimplexWrapper(
            input_dim=X_train.shape[1],
            output_dim=20,
            K=4,
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
        kl_oracle = evaluate_kl_divergence(Y_test, Y_pred_oracle)
        kl_train_oracle = evaluate_kl_divergence(Y_train, Y_train_pred_oracle)
    results['methods']['Oracle FSDRNN'] = {
        'kl_divergence': float(kl_oracle),
        'train_kl_divergence': float(kl_train_oracle),
        'gap': float(kl_oracle - kl_train_oracle),
        'time_seconds': timer.elapsed,
        'fixed_d': d0_true,
        'fixed_reduction_type': true_reduction_type
    }
    
    oracle_ratio = kl_fsdrnn / (kl_oracle + 1e-10)
    results['methods']['FSDRNN']['oracle_efficiency_ratio'] = float(oracle_ratio)
    
    if verbose:
        print(f"  GFR KL: {error_gfr:.6f} ({results['methods']['GFR']['time_seconds']:.2f}s)")
        print(f"  DFR KL: {error_dfr:.6f} ({results['methods']['DFR']['time_seconds']:.2f}s)")
        print(f"  E2M KL: {error_e2m:.6f} ({results['methods']['E2M']['time_seconds']:.2f}s)")
        print(f"  FSDRNN KL: {kl_fsdrnn:.6f} ({results['methods']['FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle FSDRNN KL: {kl_oracle:.6f} ({results['methods']['Oracle FSDRNN']['time_seconds']:.2f}s)")
        print(f"  Oracle ratio: {oracle_ratio:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Setup 9: Compositional Simplex Responses')
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
    
    print_system_info('setup9_simplex_compositions', task_id=args.task_id, 
                      base_seed=args.seed, n_reps=args.n_reps)
    
    print("\n" + "=" * 80)
    print("SETUP 9: COMPOSITIONAL SIMPLEX RESPONSES")
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
    print_aggregate_statistics(aggregated, loss_metric_name='kl_divergence')
    print_time_comparison(aggregated)
    print_final_ranking(aggregated, loss_metric_name='kl_divergence')
    
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
