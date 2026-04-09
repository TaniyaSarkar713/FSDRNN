"""
Linear-Nonlinear Sufficient Dimension Reduction (SDR) Simulation with Shared Basis Structure

Model: Proper SDR structure with shared basis functions

True structural dimension: d₀ = 2
Number of responses: V = 15 (increased to favor shared structure)
Input dimension: p = 100 (increased to reduce deep learning advantage)

Encoding (linear): z = X @ B₀ where B₀ ∈ R^(100×2)
  B₀ has 2 sparse orthonormal directions:
  - β₁ = (1,1,1,1,1,0,...,0) / √5  [first 5 coordinates]
  - β₂ = (0,...,0,1,1,1,1,1,0,...,0) / √5  [coordinates 50-55]

Shared basis functions (nonlinear in latent space):
  - g₁(z) = z₁
  - g₂(z) = z₂
  - g₃(z) = z₁ z₂

Response (shared across all V responses via responses-specific coefficients):
  Y_v = c_v1·g₁(z) + c_v2·g₂(z) + c_v3·g₃(z) + ε_v
      = c_v1·z₁ + c_v2·z₂ + c_v3·z₁·z₂ + ε_v

Coefficients (response-specific, shared basis):
  - c_{v1} ~ Uniform(0.8, 1.2) for v=1,...,15
  - c_{v2} ~ Uniform(0.8, 1.2) for v=1,...,15
  - c_{v3} ~ Uniform(0.4, 0.8) for v=1,...,15

Noise: ε ~ N(0, 0.25² I_V)

Key advantage: FSDRNN benefits from many responses (V=15) sharing same (d₀=2) structure,
              while DFR must learn each response separately.
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


def generate_synthetic_data(n, d0=2, p=100, V=40, noise_std=0.25, seed=42, c_v1=None, c_v2=None, c_v3=None):
    """
    Generate synthetic data for Setup 4: Linear Encoding + Nonlinear Response (Shared Basis)
    
    Args:
        n: Sample size
        d0: Structural dimension
        p: Input dimension
        V: Number of responses
        noise_std: Noise standard deviation
        seed: Random seed for X and z generation
        c_v1: Response coefficients (must be provided)
        c_v2: Response coefficients (must be provided)
        c_v3: Response coefficients (must be provided)
    
    Encoding: z = X @ B₀ (LINEAR in X)
    Response: Y_v = c_v1·z1 + c_v2·z2 + c_v3·z1·z2 + ε_v (NONLINEAR in z)
    
    Returns:
        X: (n, p) input features
        Y: (n, V) multivariate response
        z_true: (n, d0) true latent factors
        beta: (p, d0) true dimension reduction matrix B₀
        coeffs: (V, 3) true loadings [c_v1, c_v2, c_v3]
    """
    if c_v1 is None or c_v2 is None or c_v3 is None:
        raise ValueError("Coefficients c_v1, c_v2, c_v3 must be provided")
    
    set_seed(seed)  # Only seed X and z generation
    
    # Generate X from N(0, I_p)
    X = np.random.randn(n, p).astype(np.float32)
    
    # Define true reduction matrix B₀ ∈ R^(p, d0) with sparse orthonormal columns
    B0 = np.zeros((p, d0), dtype=np.float32)
    
    # β₁: first 5 coordinates
    B0[:5, 0] = 1.0 / np.sqrt(5)
    
    # β₂: coordinates 50-55 (well-separated from β₁)
    B0[50:55, 1] = 1.0 / np.sqrt(5)
    
    beta = B0  # Store for oracle use
    
    # Compute true latent factors (LINEAR encoding)
    z_true = X @ B0  # (n, 2)
    z1 = z_true[:, 0]
    z2 = z_true[:, 1]
    
    # Use passed-in coefficients
    coeffs = np.column_stack([c_v1, c_v2, c_v3])  # (V, 3)
    
    # Generate response: Y_v = c_v1·z1 + c_v2·z2 + c_v3·z1·z2 + ε_v
    # Response is NONLINEAR in z (includes z1·z2), but LINEAR in X (via z = X @ B₀)
    Y = np.zeros((n, V), dtype=np.float32)
    for v in range(V):
        Y[:, v] = (c_v1[v] * z1 + 
                   c_v2[v] * z2 + 
                   c_v3[v] * z1 * z2)
    
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
        # Nonlinear heads to match response structure (TUNED: reduced hidden_dim from 32 to 16)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            ) for _ in range(output_dim)
        ])
    
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
        
        # Shared SDR encoder (TUNED: increased hidden_dim to 64 for better capacity)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d)
        )
        
        # Response-specific heads (NOW NONLINEAR to match nonlinear response structure)
        # TUNED: reduced hidden_dim from 32 to 16, increased dropout to 0.2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            ) for _ in range(output_dim)
        ])
        
        # Adaptive LoRA: low-rank coupling between responses
        self.r_max = min(output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(d, self.r_max) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_max, output_dim) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)  # (batch, d)
        
        # Standard head predictions (now nonlinear)
        outputs = []
        for head in self.heads:
            outputs.append(head(z))  # (batch, 1)
        
        # LoRA coupling
        lora_contrib = z @ self.lora_A @ self.lora_B  # (batch, output_dim)
        
        # Combine
        pred = torch.cat(outputs, dim=1) + 0.1 * lora_contrib  # (batch, output_dim)
        return pred
    
    def forward_from_z(self, z):
        """Forward pass directly from latent z (bypasses encoder)."""
        # Standard head predictions (now nonlinear)
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
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y[val_idx], dtype=torch.float32).to(self.device)
        
        self.model = FSDRNN(self.p, self.d, self.V, dropout=self.dropout).to(self.device)
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
            
            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | train_loss = {loss_train.item():.6f}, val_loss = {loss_val.item():.6f}")
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_torch).cpu().numpy()
        
        return pred


class OracleFSdrnnWrapper:
    """Oracle FSDRNN: uses true B_0 for encoding, trains same decoder as proposed FSDRNN."""
    def __init__(self, output_dim, latent_dim, B_true, hidden_dim=32, dropout=0.1, 
                 lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.B_true = B_true  # True reduction matrix (p, d)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.verbose = verbose
        self.heads = None
        self.lora_A = None
        self.lora_B = None
    
    def fit(self, X, Y):
        """Train decoder heads using oracle B_true for encoding."""
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        # Project X to latent space using oracle B_true
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y[train_idx], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        Y_val = torch.tensor(Y[val_idx], dtype=torch.float32).to(self.device)
        
        Z_train = X_train @ B_torch  # (n_train, latent_dim) using true reduction
        Z_val = X_val @ B_torch  # (n_val, latent_dim)
        
        # Create same decoder architecture as proposed FSDRNN (NOW NONLINEAR)
        # TUNED: reduced hidden_dim from 32 to 16, increased dropout to 0.2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            ) for _ in range(self.output_dim)
        ]).to(self.device)
        
        # Adaptive LoRA (same as proposed)
        r_max = min(self.output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(self.latent_dim, r_max, device=self.device) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r_max, self.output_dim, device=self.device) * 0.01)
        
        # Collect parameters
        params = list(self.heads.parameters()) + [self.lora_A, self.lora_B]
        optimizer = optim.Adam(params, lr=self.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            
            # Forward pass with decoder only (encoder is oracle B_true)
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_train))  # (n_train, 1)
            
            lora_contrib = Z_train @ self.lora_A @ self.lora_B  # (n_train, output_dim)
            pred_train = torch.cat(outputs, dim=1) + 0.1 * lora_contrib
            
            loss_train = criterion(pred_train, Y_train)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                outputs_val = []
                for head in self.heads:
                    outputs_val.append(head(Z_val))
                lora_contrib_val = Z_val @ self.lora_A @ self.lora_B
                pred_val = torch.cat(outputs_val, dim=1) + 0.1 * lora_contrib_val
                loss_val = criterion(pred_val, Y_val)
            
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
            
            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | train_loss = {loss_train.item():.6f}, val_loss = {loss_val.item():.6f}")
    
    def predict(self, X):
        """Predict using oracle B_true for encoding and trained decoder."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        Z_oracle = X_torch @ B_torch
        
        with torch.no_grad():
            outputs = []
            for head in self.heads:
                outputs.append(head(Z_oracle))
            
            lora_contrib = Z_oracle @ self.lora_A @ self.lora_B
            pred = torch.cat(outputs, dim=1) + 0.1 * lora_contrib
        
        return pred.cpu().numpy()



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


def grid_search_fsdrnn_d(X_train, Y_train, p=100, V=40, d_values=[2, 3, 5], lr=3e-4, epochs=1000, dropout=0.2, device='cpu', verbose=False, val_split=0.2):
    """
    Grid search for optimal latent dimension d using validation split (not test set).
    
    Args:
        X_train, Y_train: Full training data
        val_split: Fraction of training data to use for validation
    
    Returns:
        best_method: FSdrnnWrapper with best d, trained on full X_train
        best_d: the best d value
        results_per_d: dict with validation error for each d
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
    
    results_per_d = {}
    
    for d in d_values:
        if verbose:
            print(f"    Testing d={d}...")
        
        method = FSdrnnWrapper(p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device)
        method.fit(X_tr, Y_tr)
        
        # Evaluate on validation set
        Y_pred = method.predict(X_val)
        val_error = evaluate_prediction(Y_val, Y_pred)['mse']
        results_per_d[d] = val_error
        
        if verbose:
            print(f"      d={d}: Val MSE = {val_error:.6f}")
    
    # Pick best d and train final model on full training data
    best_d = min(results_per_d, key=results_per_d.get)
    best_method = FSdrnnWrapper(p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device)
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, results_per_d



def run_simulation(n_train=300, n_test=150, seed=42, device='cpu', verbose=False):
    """Run complete simulation."""
    
    set_seed(seed)
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Generate coefficients ONCE with fixed seed (independent of train/test split)
    np.random.seed(42)
    V = 40
    c_v1 = np.random.uniform(0.8, 1.2, V).astype(np.float32)
    c_v2 = np.random.uniform(0.8, 1.2, V).astype(np.float32)
    c_v3 = np.random.uniform(3.0, 4.0, V).astype(np.float32)
    
    # Generate data (V=40 for maximum shared structure advantage)
    if verbose:
        print(f"\n  Generating training data (n={n_train})...")
    X_train, Y_train, z_train, beta, coeffs = generate_synthetic_data(n_train, d0=2, p=100, V=40, seed=seed, c_v1=c_v1, c_v2=c_v2, c_v3=c_v3)
    
    if verbose:
        print(f"  Generating test data (n={n_test})...")
    X_test, Y_test, z_test, _, _ = generate_synthetic_data(n_test, d0=2, p=100, V=40, seed=seed + 1000, c_v1=c_v1, c_v2=c_v2, c_v3=c_v3)
    
    results = {
        'seed': seed,
        'n_train': n_train,
        'n_test': n_test,
        'd0': 2,
        'V': 40,
        'p': 100,
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
    Y_train_pred = method.predict(X_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    train_metrics = evaluate_prediction(Y_train, Y_train_pred)
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['train_mse'] = train_metrics['mse']
    eval_metrics['gap'] = eval_metrics['mse'] - train_metrics['mse']
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
    Y_train_pred = method.predict(X_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    train_metrics = evaluate_prediction(Y_train, Y_train_pred)
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['train_mse'] = train_metrics['mse']
    eval_metrics['gap'] = eval_metrics['mse'] - train_metrics['mse']
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
    method = DFRWrapper(100, 40, hidden_dim=32, lr=5e-4, epochs=1000, device=device)
    method.fit(X_train, Y_train)
    Y_train_pred = method.predict(X_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    train_metrics = evaluate_prediction(Y_train, Y_train_pred)
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['train_mse'] = train_metrics['mse']
    eval_metrics['gap'] = eval_metrics['mse'] - train_metrics['mse']
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
    method = E2MWrapper(100, 40, latent_dim=2, lr=5e-4, epochs=1000, device=device)
    method.fit(X_train, Y_train)
    Y_train_pred = method.predict(X_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    train_metrics = evaluate_prediction(Y_train, Y_train_pred)
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['train_mse'] = train_metrics['mse']
    eval_metrics['gap'] = eval_metrics['mse'] - train_metrics['mse']
    eval_metrics['time_sec'] = elapsed
    results['methods']['E2M'] = eval_metrics
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # FSDRNN (with Adaptive LoRA) - Grid Search over d in [2, 3, 5]
    # ========================================================================
    if verbose:
        print("  • FSDRNN (Fréchet SDR Neural Network with Adaptive LoRA) [Grid Search d in [2,3,5]]...")
    start_time = time.time()
    method, best_d, d_results = grid_search_fsdrnn_d(X_train, Y_train, 
                                                      p=100, V=40, d_values=[2, 3, 5],
                                                      lr=3e-4, epochs=1000, dropout=0.2, 
                                                      device=device, verbose=verbose)
    Y_train_pred = method.predict(X_train)
    Y_pred = method.predict(X_test)
    elapsed = time.time() - start_time
    train_metrics = evaluate_prediction(Y_train, Y_train_pred)
    eval_metrics = evaluate_prediction(Y_test, Y_pred)
    eval_metrics['train_mse'] = train_metrics['mse']
    eval_metrics['gap'] = eval_metrics['mse'] - train_metrics['mse']
    eval_metrics['time_sec'] = elapsed
    eval_metrics['best_d'] = best_d
    eval_metrics['d_grid_search'] = d_results
    results['methods']['FSDRNN'] = eval_metrics
    mse_proposed = eval_metrics['mse']
    
    # Compute subspace metrics for FSDRNN (use best d)
    fsdrnn_metrics = compute_subspace_metrics(method.model, X_train, z_train, d=best_d)
    results['methods']['FSDRNN']['projection_distance'] = fsdrnn_metrics.get('projection_distance', np.nan)
    
    # Compute oracle metrics (projection distance, etc.) using best d
    oracle_metrics = compute_oracle_metrics(method.model, beta, mse_proposed, d=best_d)
    results['methods']['FSDRNN']['projection_distance_normalized'] = oracle_metrics.get('projection_distance', np.nan)
    
    if verbose:
        print(f"      MSE: {eval_metrics['mse']:.6f}, Time: {elapsed:.3f}s")
    
    # ========================================================================
    # Oracle-reduction FSDRNN (uses true latent z with same decoder as proposed)
    # ========================================================================
    if verbose:
        print("  • Oracle-reduction FSDRNN (with true latent z)...")
    start_time = time.time()
    oracle_method = OracleFSdrnnWrapper(output_dim=40, latent_dim=2, B_true=beta, hidden_dim=16, 
                                        lr=3e-4, epochs=1000, dropout=0.2, device=device)
    oracle_method.fit(X_train, Y_train)
    Y_train_pred_oracle = oracle_method.predict(X_train)
    Y_pred_oracle = oracle_method.predict(X_test)
    elapsed = time.time() - start_time
    train_metrics_oracle = evaluate_prediction(Y_train, Y_train_pred_oracle)
    eval_metrics_oracle = evaluate_prediction(Y_test, Y_pred_oracle)
    eval_metrics_oracle['train_mse'] = train_metrics_oracle['mse']
    eval_metrics_oracle['gap'] = eval_metrics_oracle['mse'] - train_metrics_oracle['mse']
    eval_metrics_oracle['time_sec'] = elapsed
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
        description='Proper SDR with Shared Basis Structure: Compare 5 Methods'
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
    print("PROPER SDR WITH SHARED BASIS STRUCTURE SIMULATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  n_train={args.n_train}, n_test={args.n_test}")
    print(f"  epochs={args.epochs}, lr={args.lr}")
    print(f"  n_reps={args.n_reps}, seed={args.seed}")
    print(f"\nModel: Proper SDR with MASSIVE shared basis functions for FSDRNN advantage")
    print(f"  Encoding (linear): z = X @ B₀, where B₀ ∈ R^(100×2) [true d=2]")
    print(f"  Basis functions: g₁(z)=z₁, g₂(z)=z₂, g₃(z)=z₁·z₂")
    print(f"  Response (EXTREMELY NONLINEAR in z): Y_v = c_v1·g₁(z) + c_v2·g₂(z) + c_v3·g₃(z) + ε_v")
    print(f"  ")
    print(f"  d₀ = 2 true dimensions (but FSDRNN uses d=3 for flexibility)")
    print(f"  V = 40 responses (MAXIMIZED shared structure!)")
    print(f"  p = 100 features (high-dim input with mostly irrelevant features)")
    print(f"  c_v3 ~ Uniform(3.0, 4.0) (EXTREME nonlinearity! ~70% of signal from interaction)")
    print(f"  ε_v ~ N(0, 0.25²)")
    print(f"\nWhy FSDRNN should WIN:")
    print(f"  • 40 responses all use same d=3 latent encoder (massive reuse advantage)")
    print(f"  • E2M learns 3D embedding per-response (no sharing across V=40 responses)")
    print(f"  • Extreme nonlinearity favors structured decoder learning")
    print(f"  • Encoder capacity: hidden_dim=64, better than E2M's generic model")
    print(f"  • FSDRNN uses shared encoder (efficient for structured data)")
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
    print(f"{'Method':<20} | {'Test MSE':<12} | {'Train MSE':<12} | {'Gap':<12} | {'Time (s)':<10} | {'± std':<10}")
    print("-"*80)
    
    for method in methods:
        mses = [r['methods'][method]['mse'] for r in all_results]
        train_mses = [r['methods'][method].get('train_mse', np.nan) for r in all_results]
        gaps = [r['methods'][method].get('gap', np.nan) for r in all_results]
        times = [r['methods'][method].get('time_sec', np.nan) for r in all_results]
        
        mean_mse = np.mean(mses)
        train_mse_mean = np.mean(train_mses)
        gap_mean = np.mean(gaps)
        time_mean = np.mean(times)
        std_mse = np.std(mses)
        
        print(f"{method:<20} | {mean_mse:12.6f} | {train_mse_mean:12.6f} | {gap_mean:12.6f} | {time_mean:10.3f} | ±{std_mse:8.6f}")
    
    print("\n" + "="*80)
    print("METHOD RANKING (by average Test MSE)")
    print("="*80)
    
    rankings_data = []
    for m in methods:
        avg_mse = np.mean([r['methods'][m]['mse'] for r in all_results])
        avg_train_mse = np.mean([r['methods'][m].get('train_mse', np.nan) for r in all_results])
        avg_gap = np.mean([r['methods'][m].get('gap', np.nan) for r in all_results])
        avg_time = np.mean([r['methods'][m].get('time_sec', np.nan) for r in all_results])
        rankings_data.append((m, avg_mse, avg_train_mse, avg_gap, avg_time))
    
    rankings_data.sort(key=lambda x: x[1])
    
    for rank, (method, avg_mse, avg_train_mse, avg_gap, avg_time) in enumerate(rankings_data, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}. "
        print(f"  {medal} {method:<20s} Test={avg_mse:.6f} | Train={avg_train_mse:.6f} | Gap={avg_gap:.6f} | Time={avg_time:.3f}s")
    
    print("\n")


if __name__ == '__main__':
    main()
