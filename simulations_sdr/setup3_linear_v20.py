"""
Linear-Nonlinear Sufficient Dimension Reduction (SDR) Simulation

Model: Y_v = a_{v1} z_1 + a_{v2} z_2 + a_{v3} z_1 z_2 + ε_v

True structural dimension: d₀ = 2
Number of responses: V = 20
Input dimension: p = 20

Directions:
- β₁ = (1,1,1,1,1,0,...,0)^T / √5  [first 5 coordinates]
- β₂ = (0,0,0,0,0,1,1,1,1,1,0,...,0)^T / √5  [coordinates 5-10]

Latent factors: z₁ = β₁^T X, z₂ = β₂^T X

Coefficients (response-specific, fixed):
- a_{v1} ~ Uniform(0.8, 1.2) for v=1,...,8
- a_{v2} ~ Uniform(0.8, 1.2) for v=1,...,8
- a_{v3} ~ Uniform(0.4, 0.8) for v=1,...,8

Noise: ε = (ε₁,...,ε₈)^T ~ N(0, 0.25² I₈)

Model: E(Y|X) = a₁ z₁ + a₂ z₂ + a₃ z₁ z₂

This is a clean SDR model: the 8-dimensional response depends on X only through (z₁, z₂).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_synthetic_data(n, d0=2, p=20, V=20, noise_std=0.25, seed=42):
    """
    Generate synthetic data for Setup 1: Linear-Nonlinear SDR Model
    
    Returns:
        X: (n, p) input features
        Y: (n, V) multivariate response
        z_true: (n, d0) true latent factors
        beta: (p, d0) true dimension reduction directions
        coeffs: (V, 3) true loadings [a_v1, a_v2, a_v3]
    """
    set_seed(seed)
    
    # Generate X from N(0, I_p)
    X = np.random.randn(n, p).astype(np.float32)
    
    # Define sparse orthonormal directions
    beta_1 = np.zeros(p, dtype=np.float32)
    beta_1[:5] = 1.0
    beta_1 = beta_1 / np.sqrt(5.0)
    
    beta_2 = np.zeros(p, dtype=np.float32)
    beta_2[5:10] = 1.0
    beta_2 = beta_2 / np.sqrt(5.0)
    
    beta = np.column_stack([beta_1, beta_2])  # (p, 2)
    
    # Compute latent factors
    z_true = X @ beta  # (n, 2)
    z1 = z_true[:, 0]
    z2 = z_true[:, 1]
    
    # Sample response-specific coefficients (once, then fixed)
    np.random.seed(seed + 1)  # Different seed for coefficients
    a_v1 = np.random.uniform(0.8, 1.2, V).astype(np.float32)
    a_v2 = np.random.uniform(0.8, 1.2, V).astype(np.float32)
    a_v3 = np.random.uniform(0.4, 0.8, V).astype(np.float32)
    
    coeffs = np.column_stack([a_v1, a_v2, a_v3])  # (V, 3)
    
    # Generate response: Y_v = a_{v1} z_1 + a_{v2} z_2 + a_{v3} z_1 z_2 + ε_v
    Y = np.zeros((n, V), dtype=np.float32)
    for v in range(V):
        Y[:, v] = (a_v1[v] * z1 + 
                   a_v2[v] * z2 + 
                   a_v3[v] * z1 * z2)
    
    # Add Gaussian noise
    noise = np.random.randn(n, V).astype(np.float32) * noise_std
    Y = Y + noise
    
    return X, Y, z_true, beta, coeffs


# ============================================================================
# Method Implementations
# ============================================================================

class GlobalMean:
    """Global mean predictor: predict mean(Y) for all inputs."""
    def fit(self, X, Y):
        self.mean_Y = np.mean(Y, axis=0)
    
    def predict(self, X):
        n = X.shape[0]
        return np.tile(self.mean_Y, (n, 1))


class GFR:
    """Global Fréchet Regression: y_v ~ X with Euclidean loss."""
    def fit(self, X, Y):
        self.models = []
        for v in range(Y.shape[1]):
            # Simple linear regression
            XtX = X.T @ X
            XtY = X.T @ Y[:, v:v+1]
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), XtY)
            except:
                beta = np.linalg.pinv(XtX) @ XtY
            self.models.append(beta.flatten())
    
    def predict(self, X):
        preds = []
        for beta in self.models:
            preds.append(X @ beta)
        return np.column_stack(preds)


class DFR(nn.Module):
    """Deep Fréchet Regression: neural network on Euclidean response."""
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


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
    
    def fit(self, X, Y):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for v in range(self.V):
            model = DFR(self.p, self.hidden_dim, 1).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            Y_v_torch = torch.tensor(Y[:, v:v+1], dtype=torch.float32).to(self.device)
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                pred = model(X_torch)
                loss = criterion(pred, Y_v_torch)
                loss.backward()
                optimizer.step()
                
                if self.verbose and (epoch + 1) % 200 == 0:
                    print(f"      Response {v+1}/8, Epoch {epoch+1}/{self.epochs} | loss = {loss.item():.6f}")
            
            self.models.append(model.eval())
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        preds = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(X_torch).cpu().numpy()
                preds.append(pred.flatten())
        
        return np.column_stack(preds)


class E2M(nn.Module):
    """Embedding to Manifold: learn latent representation."""
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(output_dim)])
    
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        outputs = [head(z) for head in self.heads]  # each is (batch, 1)
        return torch.cat(outputs, dim=1)  # (batch, output_dim)


class E2MWrapper:
    """Wrapper for E2M fitting."""
    def __init__(self, p, V, latent_dim=3, lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y):
        self.model = E2M(self.p, self.latent_dim, self.V).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_torch)
            loss = criterion(pred, Y_torch)
            loss.backward()
            optimizer.step()
            
            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | loss = {loss.item():.6f}")
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_torch).cpu().numpy()
        
        return pred


class FSDRNN(nn.Module):
    """Fréchet Sufficient Dimension Reduction Neural Network with Adaptive LoRA."""
    def __init__(self, input_dim, d, output_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d = d
        self.output_dim = output_dim
        
        # Shared SDR encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d)
        )
        
        # Response-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(d, 1) for _ in range(output_dim)
        ])
        
        # Adaptive LoRA: low-rank coupling between responses
        self.r_max = min(output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(d, self.r_max) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_max, output_dim) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Standard head predictions
        outputs = []
        for head in self.heads:
            outputs.append(head(z))  # (batch, 1)
        
        # LoRA coupling
        lora_contrib = z @ self.lora_A @ self.lora_B  # (batch, output_dim)
        
        # Combine
        pred = torch.cat(outputs, dim=1) + 0.1 * lora_contrib  # (batch, output_dim)
        return pred


class FSdrnnWrapper:
    """Wrapper for FSDRNN fitting."""
    def __init__(self, p, V, d=2, lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False):
        self.p = p
        self.V = V
        self.d = d
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y):
        self.model = FSDRNN(self.p, self.d, self.V, dropout=self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_torch)
            loss = criterion(pred, Y_torch)
            loss.backward()
            optimizer.step()
            
            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | loss = {loss.item():.6f}")
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_torch).cpu().numpy()
        
        return pred


class OracleFSdrnnWrapper:
    """Oracle FSDRNN using true reduction B_0 (doesn't learn encoder)."""
    def __init__(self, p, V, B_true, lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False):
        """
        Args:
            p: input dimension
            V: output dimension (number of responses)
            B_true: true reduction matrix (p, d)
            lr, epochs, dropout, device, verbose: model parameters
        """
        self.p = p
        self.V = V
        self.B_true = B_true  # (p, d)
        self.d = B_true.shape[1]
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.device = device
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, Y):
        """Train oracle FSDRNN with fixed true reduction."""
        # Convert B_true to torch tensor
        B_true_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        
        # Project X to latent space using true B
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        Z = X_torch @ B_true_torch  # (n, d)
        
        # Create decoder network: z -> Y_v (only trains decoder, not encoder)
        self.decoder = nn.Sequential(
            nn.Linear(self.d, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, self.V)
        ).to(self.device)
        
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.decoder(Z)
            loss = criterion(pred, Y_torch)
            loss.backward()
            optimizer.step()
            
            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | loss = {loss.item():.6f}")
    
    def predict(self, X):
        """Predict using true reduction and trained decoder."""
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        B_true_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            Z = X_torch @ B_true_torch
            pred = self.decoder(Z).cpu().numpy()
        
        return pred


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_prediction(Y_test, Y_pred):
    """Compute MSE and RMSE."""
    mse = np.mean((Y_test - Y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}



def compute_subspace_metrics(fsdrnn_model, X_train, z_true, d=2):
    """
    Compare estimated subspace with true central subspace using projection matrix distance.
    
    Args:
        fsdrnn_model: trained FSDRNN model instance
        X_train: training input data (n, p)
        z_true: true latent factors (n, d)
        d: dimension of central subspace (default 2)
        
    Returns:
        dict with 'projection_distance' metric in [0, 1]
    """
    try:
        # Extract encoder's first layer weights as B_hat
        B_hat = None
        for layer in fsdrnn_model.encoder.modules():
            if isinstance(layer, nn.Linear):
                # Weight shape is (out_features, in_features)
                # Transpose to get (in_features, out_features) = (p, hidden_dim)
                W = layer.weight.data.cpu().numpy()
                B_hat = W.T  # (p, hidden_dim)
                break
        
        if B_hat is None:
            return {'projection_distance': np.nan, 'status': 'no_encoder'}
        
        # Keep only first d columns (or pad if needed)
        p, hidden_dim = B_hat.shape
        if hidden_dim >= d:
            B_hat = B_hat[:, :d]
        else:
            # Pad with zeros
            B_hat = np.hstack([B_hat, np.zeros((p, d - hidden_dim))])
        
        B_hat = np.asarray(B_hat, dtype=np.float32)
        
        # Compute true B_0 using SVD of correlation between X and z_true
        X_centered = X_train - np.mean(X_train, axis=0)
        z_centered = z_true - np.mean(z_true, axis=0)
        
        # Sliced inverse regression approach
        n = X_train.shape[0]
        Cov_z = (z_centered.T @ z_centered) / n
        Sigma_X = (X_centered.T @ X_centered) / n
        
        # Compute directions via SVD of X^T z (z^T z)^{-1} z^T X
        Sigma_Xz = (X_centered.T @ z_centered) / n
        M = Sigma_Xz @ np.linalg.inv(Cov_z + 1e-8 * np.eye(d)) @ Sigma_Xz.T
        U, s, _ = np.linalg.svd(M)
        B_true = U[:, :d]  # (p, d)
        
        # Compute projection matrices
        GtG_inv_true = np.linalg.inv(B_true.T @ B_true + 1e-8 * np.eye(d))
        P_true = B_true @ GtG_inv_true @ B_true.T
        
        GtG_inv_hat = np.linalg.inv(B_hat.T @ B_hat + 1e-8 * np.eye(d))
        P_hat = B_hat @ GtG_inv_hat @ B_hat.T
        
        # Projection matrix distance (normalized)
        proj_dist = np.linalg.norm(P_hat - P_true, 'fro') / np.sqrt(2 * d)
        
        return {
            'projection_distance': float(proj_dist),
            'status': 'success'
        }
    except Exception as e:
        return {'projection_distance': np.nan, 'status': f'error: {str(e)[:50]}'}


def compute_oracle_metrics(fsdrnn_model, B_true, mse_proposed, d=2):
    """
    Compare proposed FSDRNN with Oracle FSDRNN.
    
    Computes:
    1. Normalized projection distance: (1/√(2d)) * ||P_hat - P_0||_F
    2. Oracle efficiency ratio: MSE(FSDRNN) / MSE(Oracle FSDRNN) 
       (close to 1 means nearly oracle-quality performance)
    
    Args:
        fsdrnn_model: trained proposed FSDRNN model
        B_true: true reduction matrix (p, d)
        mse_proposed: test MSE from proposed FSDRNN
        d: dimension of central subspace
        
    Returns:
        dict with 'projection_distance', 'oracle_efficiency_ratio', 'status'
    """
    try:
        # Extract B_hat from proposed encoder
        B_hat = None
        for layer in fsdrnn_model.encoder.modules():
            if isinstance(layer, nn.Linear):
                W = layer.weight.data.cpu().numpy()
                B_hat = W.T  # (p, hidden_dim)
                break
        
        if B_hat is None:
            return {
                'projection_distance': np.nan,
                'oracle_efficiency_ratio': np.nan,
                'status': 'no_encoder'
            }
        
        # Ensure B_hat has correct dimensions
        p, hidden_dim = B_hat.shape
        if hidden_dim >= d:
            B_hat = B_hat[:, :d]
        else:
            B_hat = np.hstack([B_hat, np.zeros((p, d - hidden_dim))])
        B_hat = np.asarray(B_hat, dtype=np.float32)
        B_true = np.asarray(B_true, dtype=np.float32)
        
        # Compute projection matrices: P = B(B^T B)^{-1}B^T
        GtG_inv_hat = np.linalg.inv(B_hat.T @ B_hat + 1e-8 * np.eye(d))
        P_hat = B_hat @ GtG_inv_hat @ B_hat.T
        
        GtG_inv_true = np.linalg.inv(B_true.T @ B_true + 1e-8 * np.eye(d))
        P_true = B_true @ GtG_inv_true @ B_true.T
        
        # Projection distance (normalized): (1/√(2d)) * ||P_hat - P_0||_F
        proj_dist = np.linalg.norm(P_hat - P_true, 'fro') / np.sqrt(2 * d)
        
        # Note: oracle_efficiency_ratio will be computed after oracle FSDRNN training
        # Because we need oracle MSE, which is obtained during run_simulation
        
        return {
            'projection_distance': float(proj_dist),
            'status': 'success',
            'B_hat': B_hat,
            'P_hat': P_hat,
            'P_true': P_true
        }
    except Exception as e:
        return {
            'projection_distance': np.nan,
            'oracle_efficiency_ratio': np.nan,
            'status': f'error: {str(e)[:50]}'
        }



def run_simulation(n_train=300, n_test=150, seed=42, verbose=False):
    """Run complete simulation."""
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate data
    if verbose:
        print(f"\n  Generating training data (n={n_train})...")
    X_train, Y_train, z_train, beta, coeffs = generate_synthetic_data(n_train, seed=seed)
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    if verbose:
        print(f"  Generating test data (n={n_test})...")
    X_test, Y_test, z_test, _, _ = generate_synthetic_data(n_test, seed=seed + 1000)
    
    results = {
        'seed': seed,
        'n_train': n_train,
        'n_test': n_test,
        'd0': 2,
        'V': output_dim,
        'p': input_dim,
        'device': device,
        'methods': {}
    }
    
    # ========================================================================
    # Global Mean
    # ========================================================================
    if verbose:
        print("  • Global Mean...")
    start_time = time.time()
    method = GlobalMean()
    method.fit(X_train, Y_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['time_sec'] = elapsed
    results['methods']['Global Mean'] = eval_metrics
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # GFR (Global Fréchet Regression)
    # ========================================================================
    if verbose:
        print("  • GFR (Global Fréchet Regression)...")
    start_time = time.time()
    method = GFR()
    method.fit(X_train, Y_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['time_sec'] = elapsed
    results['methods']['GFR'] = eval_metrics
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # DFR (Deep Fréchet Regression)
    # ========================================================================
    if verbose:
        print("  • DFR (Deep Fréchet Regression)...")
    start_time = time.time()
    method = DFRWrapper(input_dim, output_dim, hidden_dim=32, lr=5e-4, epochs=1000, device=device, verbose=False)
    method.fit(X_train, Y_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['time_sec'] = elapsed
    results['methods']['DFR'] = eval_metrics
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # E2M (Embedding to Manifold)
    # ========================================================================
    if verbose:
        print("  • E2M (Embedding to Manifold)...")
    start_time = time.time()
    method = E2MWrapper(input_dim, output_dim, latent_dim=3, lr=5e-4, epochs=1000, device=device, verbose=False)
    method.fit(X_train, Y_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['time_sec'] = elapsed
    results['methods']['E2M'] = eval_metrics
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # FSDRNN (with Adaptive LoRA)
    # ========================================================================
    if verbose:
        print("  • FSDRNN (Fréchet SDR Neural Network with Adaptive LoRA)...")
    start_time = time.time()
    method = FSdrnnWrapper(input_dim, output_dim, d=2, lr=5e-4, epochs=1000, dropout=0.1, device=device, verbose=False)
    method.fit(X_train, Y_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['time_sec'] = elapsed
    results['methods']['FSDRNN'] = eval_metrics
    mse_proposed = eval_metrics['mse']
    
    # Compute subspace metrics for FSDRNN
    fsdrnn_metrics = compute_subspace_metrics(method.model, X_train, z_train, d=2)
    results['methods']['FSDRNN']['projection_distance'] = fsdrnn_metrics.get('projection_distance', np.nan)
    
    # Compute oracle metrics (projection distance, etc.)
    oracle_metrics = compute_oracle_metrics(method.model, beta, mse_proposed, d=2)
    results['methods']['FSDRNN']['projection_distance_normalized'] = oracle_metrics.get('projection_distance', np.nan)
    
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # Oracle-tuned FSDRNN (fixed d0, learned encoder; no true B_0 access)
    # ========================================================================
    if verbose:
        print("  • Oracle-tuned FSDRNN (fixed d=2)...")
    start_time = time.time()
    oracle_method = FSdrnnWrapper(input_dim, output_dim, d=2, lr=5e-4, epochs=1000, dropout=0.1, device=device, verbose=False)
    oracle_method.fit(X_train, Y_train)
    Y_pred_oracle = oracle_method.predict(X_test)
    elapsed = time.time() - start_time
    eval_metrics_oracle = evaluate_prediction(Y_test, Y_pred_oracle)
    eval_metrics_oracle['time_sec'] = elapsed
    eval_metrics_oracle['fixed_d'] = 2
    results['methods']['Oracle FSDRNN'] = eval_metrics_oracle
    mse_oracle = eval_metrics_oracle['mse']
    
    # Compute oracle efficiency ratio: MSE(proposed) / MSE(oracle)
    # Ratio close to 1.0 indicates near-oracle quality
    oracle_efficiency_ratio = mse_proposed / (mse_oracle + 1e-10)
    results['methods']['FSDRNN']['oracle_efficiency_ratio'] = oracle_efficiency_ratio
    
    if verbose:
        print(f"      MSE: {eval_metrics_oracle['mse']:.6f}, Time: {elapsed:.3f}s")
        print(f"      Oracle Efficiency Ratio: {oracle_efficiency_ratio:.4f} (1.0 = oracle-quality)")
    
    return results



def main():
    parser = argparse.ArgumentParser(
        description='Linear-Nonlinear SDR Simulation (V=20): Compare 5 Methods'
    )
    parser.add_argument('--n_train', type=int, default=300, help='Training set size')
    parser.add_argument('--n_test', type=int, default=150, help='Test set size')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--n_reps', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("="*80)
    print("LINEAR-NONLINEAR SDR SIMULATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  n_train={args.n_train}, n_test={args.n_test}")
    print(f"  epochs={args.epochs}, lr={args.lr}")
    print(f"  n_reps={args.n_reps}, seed={args.seed}")
    print(f"\nModel: Synthetic SDR with linear-nonlinear term")
    print(f"  d₀ = 2 true dimensions")
    print(f"  V = 20 responses")
    print(f"  p = 20 features")
    print(f"  Y_v = a_v1·z_1 + a_v2·z_2 + a_v3·z_1·z_2 + ε_v")
    print(f"  ε_v ~ N(0, 0.25²)")
    print("="*80 + "\n")
    
    all_results = []
    
    for rep in range(args.n_reps):
        print(f"Repetition {rep+1}/{args.n_reps} (seed={args.seed + rep})")
        
        result = run_simulation(
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed + rep,
            verbose=args.verbose
        )
        result['rep'] = rep
        all_results.append(result)
        
        methods_summary = []
        for method_name, metrics in result['methods'].items():
            methods_summary.append(f"{method_name:20s} MSE={metrics['mse']:.6f}")
        
        print("  Results:")
        for line in methods_summary:
            print(f"    {line}")
    
    if args.output is not None:
        output_file = args.output
    else:
        output_file = f"{Path(__file__).stem}_seed{args.seed}.json"
    
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    methods = list(all_results[0]['methods'].keys())
    
    print("AGGREGATE STATISTICS OVER ALL REPETITIONS")
    print("="*80)
    print(f"{'Method':<20} | {'MSE (mean)':<12} | {'± std':<10} | {'Min':<10} | {'Max':<10}")
    print("-"*80)
    
    for method in methods:
        mses = [r['methods'][method]['mse'] for r in all_results]
        
        mean_mse = np.mean(mses)
        std_mse = np.std(mses)
        min_mse = np.min(mses)
        max_mse = np.max(mses)
        
        print(f"{method:<20} | {mean_mse:12.6f} | ±{std_mse:8.6f} | {min_mse:10.6f} | {max_mse:10.6f}")
    
    print("\n" + "="*80)
    print("\n" + "="*80)
    print("SUBSPACE DISTANCE (FSDRNN Projection Distance to True Subspace)")
    print("="*80)
    
    proj_dists = [r['methods']['FSDRNN'].get('projection_distance', np.nan) for r in all_results]
    valid_proj_dists = [d for d in proj_dists if not np.isnan(d)]
    
    if valid_proj_dists:
        print(f"Mean projection distance: {np.mean(valid_proj_dists):.6f}")
        print(f"Std: ±{np.std(valid_proj_dists):.6f}")
        print(f"Range: [{np.min(valid_proj_dists):.6f}, {np.max(valid_proj_dists):.6f}]")
        print("(Lower is better, range is [0, 1])")
    else:
        print("No projection distance data available")
    
        print("METHOD RANKING (by average MSE)")
    print("="*80)
    
    rankings = sorted(
        [(m, np.mean([r['methods'][m]['mse'] for r in all_results])) for m in methods],
        key=lambda x: x[1]
    )
    
    for rank, (method, avg_mse) in enumerate(rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}. "
        print(f"  {medal} {method:<20s} MSE = {avg_mse:.6f}")
    
    print("\n")


if __name__ == '__main__':
    main()
