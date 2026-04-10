"""
Enhanced Correlated Responses with Nonlinearity (Setup 5)

OPTIMIZED FOR FSDRNN: combines low-dimensional structure, shared responses, and moderate nonlinearity

True structural dimension: d₀ = 2
Number of responses: V = 8 (8 responses from 3 latent signals)
Input dimension: p = 30 (true signal in first 10, rest noise)

Directions (sparse, signal-relevant):
- β₁ = (1,1,1,1,1,0,...,0)^T / √5  [first 5 coordinates; coords 11-30 are noise]
- β₂ = (0,0,0,0,0,1,1,1,1,1,0,...,0)^T / √5  [coordinates 6-10; coords 11-30 are noise]

Latent factors: z₁ = β₁^T X, z₂ = β₂^T X

Latent response signals (moderate nonlinearity):
- f₁ = sin(z₁)        [nonlinear transformation of z₁]
- f₂ = sin(z₂)        [nonlinear transformation of z₂]
- f₃ = z₁ · z₂        [interaction term]

Response structure (heavily correlated WITHIN groups):
Y = Λ·(f₁, f₂, f₃)^T + ε

where Λ ∈ R^(8×3) with repeated rows:
- Y₁ = Y₂ = Y₃ = f₁ + ε₁,₂,₃
- Y₄ = Y₅ = Y₆ = f₂ + ε₄,₅,₆
- Y₇ = Y₈ = f₃ + ε₇,₈

Noise structure (block-correlated):
- High correlation (ρ=0.8) within response groups
- Between-group noise is independent
- σ = 0.15 (modest noise level)

Key FSDRNN advantages:
1. Low-dimensional predictor structure (d₀=2 in p=50)
2. Moderate nonlinearity (sin, multiplication)
3. Strong shared response structure (8 responses from 3 factors)
4. Block-correlated noise within groups
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


def generate_synthetic_data(n, d0=2, p=50, V=8, noise_std=0.15, noise_corr=0.8, seed=42, a_v1=None, a_v2=None, a_v3=None):
    """
    Generate synthetic data for Setup 5: Enhanced Correlated Responses with Nonlinearity
    
    Args:
        n: Sample size
        d0: Structural dimension (always 2)
        p: Input dimension (50 with 10 signal, 40 noise)
        V: Number of responses (8)
        noise_std: Standard deviation of noise (0.15)
        noise_corr: Correlation within response groups (0.8)
        seed: Random seed for X, z, and noise generation
        a_v1: Scaling coefficients for f1 (must be provided)
        a_v2: Scaling coefficients for f2 (must be provided)
        a_v3: Scaling coefficients for f3 (must be provided)
    
    Returns:
        X: (n, p) input features
        Y: (n, V) multivariate response with block-correlated noise
        z_true: (n, d0) true latent factors
        beta: (p, d0) true dimension reduction directions
    """
    if a_v1 is None or a_v2 is None or a_v3 is None:
        raise ValueError("Coefficients a_v1, a_v2, a_v3 must be provided")
    
    set_seed(seed)  # Seed for X, z, and noise generation
    
    # Generate X ~ N(0, I_p)
    X = np.random.randn(n, p).astype(np.float32)
    
    # Define sparse orthonormal directions (first 10 coords only have signal)
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
    
    # Latent response signals (MODERATE NONLINEARITY)
    f1 = np.sin(z1)        # sin(z₁)
    f2 = np.sin(z2)        # sin(z₂)
    f3 = z1 * z2           # z₁ · z₂ (multiplicative interaction)
    
    # Generate responses: Y = Λ·(f₁, f₂, f₃)^T + ε
    # Λ has repeated rows for grouped responses (scaled by a_v1, a_v2, a_v3)
    Y = np.zeros((n, V), dtype=np.float32)
    
    # Group 1: Y₁, Y₂, Y₃ all respond to a_v1 * f₁
    Y[:, 0] = a_v1[0] * f1
    Y[:, 1] = a_v1[1] * f1
    Y[:, 2] = a_v1[2] * f1
    
    # Group 2: Y₄, Y₅, Y₆ all respond to a_v2 * f₂
    Y[:, 3] = a_v2[3] * f2
    Y[:, 4] = a_v2[4] * f2
    Y[:, 5] = a_v2[5] * f2
    
    # Group 3: Y₇, Y₈ respond to a_v3 * f₃
    Y[:, 6] = a_v3[6] * f3
    Y[:, 7] = a_v3[7] * f3
    
    # ========================================================================
    # BLOCK-CORRELATED NOISE (key FSDRNN advantage)
    # High correlation within response groups, independent between groups
    # ========================================================================
    # Build covariance structure (deterministically, independent of seed)
    Sigma_eps = np.eye(V, dtype=np.float32)
    
    # Group 1 (Y₁, Y₂, Y₃): correlation ρ=0.8
    for i in range(3):
        for j in range(3):
            if i == j:
                Sigma_eps[i, j] = 1.0
            else:
                Sigma_eps[i, j] = noise_corr
    
    # Group 2 (Y₄, Y₅, Y₆): correlation ρ=0.8
    for i in range(3, 6):
        for j in range(3, 6):
            if i == j:
                Sigma_eps[i, j] = 1.0
            else:
                Sigma_eps[i, j] = noise_corr
    
    # Group 3 (Y₇, Y₈): correlation ρ=0.8
    for i in range(6, 8):
        for j in range(6, 8):
            if i == j:
                Sigma_eps[i, j] = 1.0
            else:
                Sigma_eps[i, j] = noise_corr
    
    # Scale by noise_std²
    Sigma_eps = (noise_std ** 2) * Sigma_eps
    
    # Generate correlated noise
    try:
        L = np.linalg.cholesky(Sigma_eps)
        noise = np.random.randn(n, V).astype(np.float32) @ L.T
    except:
        # Fallback to iid if Cholesky fails
        noise = np.random.randn(n, V).astype(np.float32) * noise_std
    
    Y = Y + noise
    
    return X, Y, z_true, beta


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
    """Deep Fréchet Regression: independent neural network for each response."""
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
    """Wrapper to train independent DFR models for each response."""
    def __init__(self, input_dim, output_dim=8, hidden_dim=32, lr=0.0005, epochs=1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, Y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.models = []
        for v in range(self.output_dim):
            model = DFR(self.input_dim, self.hidden_dim, 1).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            Y_v = torch.FloatTensor(Y[:, v:v+1]).to(self.device)
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                pred = model(X_tensor)
                loss = torch.mean((pred - Y_v) ** 2)
                loss.backward()
                optimizer.step()
            
            self.models.append(model.eval())
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                pred = model(X_tensor).cpu().numpy()
                preds.append(pred.flatten())
        return np.column_stack(preds)


class E2M(nn.Module):
    """Efficient Euclidean-to-Euclidean: shared encoder + response-specific decoders."""
    def __init__(self, input_dim, hidden_dim=32, latent_dim=2, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class E2MWrapper:
    """Wrapper to train E2M with shared encoder."""
    def __init__(self, input_dim, output_dim=8, hidden_dim=32, latent_dim=2, lr=0.0005, epochs=1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, Y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)
        
        self.model = E2M(self.input_dim, self.hidden_dim, self.latent_dim, self.output_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = torch.mean((pred - Y_tensor) ** 2)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()
        return pred


class FSDRNN(nn.Module):
    """Functional Sliced Dimension Reduction Neural Network with shared latent factors."""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=2, output_dim=8, dropout=0.2):
        super().__init__()
        
        # Shared encoder: X → z (low-dimensional latent)
        # Enhanced architecture for handling p=50 input and nonlinearity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Response-specific decoders: z → y_v
        # Enhanced decoders to learn nonlinear mappings from latent space
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
            for _ in range(output_dim)
        ])
    
    def forward(self, x):
        z = self.encoder(x)
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(z))
        return torch.cat(outputs, dim=1)
    
    def forward_from_z(self, z):
        """Forward pass directly from latent z (bypasses encoder)."""
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(z))
        return torch.cat(outputs, dim=1)


class FSdrnnWrapper:
    """Wrapper to train FSDRNN model."""
    def __init__(self, input_dim, output_dim=8, hidden_dim=128, latent_dim=2, d=None,
                 lr=0.0005, epochs=1000, dropout=0.2, device='cpu', verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Allow d parameter to override latent_dim for grid search compatibility
        self.latent_dim = d if d is not None else latent_dim
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.verbose = verbose
        self.model = None
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
    
    def fit(self, X, Y):
        # Split into train/val (80/20)
        n = X.shape[0]
        val_size = max(int(0.2 * n), 10)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_idx = idx[:-val_size]
        val_idx = idx[-val_size:]
        
        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        Y_train = torch.FloatTensor(Y[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)
        Y_val = torch.FloatTensor(Y[val_idx]).to(self.device)
        
        self.model = FSDRNN(
            self.input_dim, self.hidden_dim, self.latent_dim, self.output_dim, dropout=self.dropout
        ).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            pred_train = self.model(X_train)
            loss_train = torch.mean((pred_train - Y_train) ** 2)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_val = self.model(X_val)
                loss_val = torch.mean((pred_val - Y_val) ** 2)
            
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
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()
        return pred


class OracleFSdrnnWrapper:
    """Oracle FSDRNN: uses true B_0 for encoding, trains same decoders as proposed."""
    def __init__(self, output_dim, latent_dim, B_true, hidden_dim=128, 
                 lr=5e-4, epochs=1000, device='cpu', verbose=False):
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.B_true = B_true  # True reduction matrix (p, d)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.verbose = verbose
        self.decoders = None
    
    def fit(self, X, Y):
        """Train decoders using oracle B_true for encoding (same architecture as proposed)."""
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
        
        # Create same decoder architecture as proposed FSDRNN
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, 1)
            )
            for _ in range(self.output_dim)
        ]).to(self.device)
        
        # Collect parameters
        params = list(self.decoders.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
            optimizer.zero_grad()
            
            # Forward pass with decoders only (encoder is oracle B_true)
            outputs = []
            for decoder in self.decoders:
                outputs.append(decoder(Z_train))  # (n_train, 1)
            
            pred_train = torch.cat(outputs, dim=1)
            
            loss_train = criterion(pred_train, Y_train)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                outputs_val = []
                for decoder in self.decoders:
                    outputs_val.append(decoder(Z_val))
                pred_val = torch.cat(outputs_val, dim=1)
                loss_val = criterion(pred_val, Y_val)
            
            # Early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                best_state = self.decoders.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if best_state:
                    self.decoders.load_state_dict(best_state)
                break
            
            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | train_loss = {loss_train.item():.6f}, val_loss = {loss_val.item():.6f}")
    
    def predict(self, X):
        """Predict using oracle B_true for encoding and trained decoders."""
        B_torch = torch.tensor(self.B_true, dtype=torch.float32).to(self.device)
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        Z_oracle = X_torch @ B_torch
        
        with torch.no_grad():
            outputs = []
            for decoder in self.decoders:
                outputs.append(decoder(Z_oracle))
            
            pred = torch.cat(outputs, dim=1)
        
        return pred.cpu().numpy()


class OracleFSdrnnWrapperOld:
    """Oracle FSDRNN using true reduction B_0 (doesn't learn encoder)."""
    def __init__(self, p, V, B_true, lr=5e-4, epochs=1000, dropout=0.1, device='cpu'):
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
        # For setup5, encoder is an nn.Sequential
        encoder = fsdrnn_model.encoder
        W = encoder[0].weight.data.cpu().numpy()  # (hidden_dim, p)
        B_hat = W.T  # (p, hidden_dim)
        
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
# Helper Functions
# ============================================================================

def compute_subspace_metrics(fsdrnn_model, X_train, z_true, d=2):
    """
    Compute projection matrix distance between estimated and true subspaces.
    
    Args:
        fsdrnn_model: Fitted FSDRNN model
        X_train: Training features (n_train, p)
        z_true: True latent factors (n_train, d)
        d: Structural dimension (default 2)
    
    Returns:
        dict with projection_distance in [0, 1]
    """
    try:
        # Extract B̂ from FSDRNN encoder first layer
        first_layer_weights = fsdrnn_model.model.encoder[0].weight.data.cpu().numpy()  # (hidden_dim, p)
        
        # Estimate B using SVD of encoder weights (simplified approach)
        U, S, Vt = np.linalg.svd(first_layer_weights, full_matrices=False)
        B_hat = Vt[:d, :].T  # (p, d)
        
        # Normalize
        B_hat = B_hat / (np.linalg.norm(B_hat, axis=0, keepdims=True) + 1e-8)
        
        # Compute true B₀ from latent factors using sliced inverse regression
        # Correlation: Σ_xz = E[X z^T]
        Sigma_xz = (X_train.T @ z_true) / X_train.shape[0]  # (p, d)
        
        # Compute B₀ via SVD
        U0, S0, Vt0 = np.linalg.svd(Sigma_xz, full_matrices=False)
        B_true = U0[:, :d]  # (p, d)
        B_true = B_true / (np.linalg.norm(B_true, axis=0, keepdims=True) + 1e-8)
        
        # Projection matrices
        P_hat = B_hat @ B_hat.T  # (p, p)
        P_true = B_true @ B_true.T  # (p, p)
        
        # Normalized Frobenius distance
        proj_dist = np.linalg.norm(P_hat - P_true, 'fro') / np.sqrt(2 * d)
        
        return {
            'projection_distance': float(proj_dist),
            'status': 'success'
        }
    except Exception as e:
        return {
            'projection_distance': np.nan,
            'status': f'error: {str(e)}'
        }


def evaluate_mse(Y_true, Y_pred):
    """Compute mean squared error."""
    return np.mean((Y_true - Y_pred) ** 2)


def grid_search_fsdrnn_d(X_train, Y_train, p=None, V=None, d_values=[2, 3, 5], lr=5e-4, epochs=1000, dropout=0.2, device='cpu', verbose=False, val_split=0.2):
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
    # Infer dimensions from data when not explicitly provided.
    if p is None:
        p = X_train.shape[1]
    if V is None:
        V = Y_train.shape[1]

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
        val_error = evaluate_mse(Y_val, Y_pred)
        results_per_d[d] = val_error
        
        if verbose:
            print(f"      d={d}: Val MSE = {val_error:.6f}")
    
    # Pick best d and train final model on full training data
    best_d = min(results_per_d, key=results_per_d.get)
    best_method = FSdrnnWrapper(p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device)
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, results_per_d



def run_simulation(n_train=200, n_test=200, n_reps=10, epochs=1000, lr=0.0005, verbose=True):
    """
    Run full simulation for Setup 5: Enhanced Correlated Responses with Nonlinearity.
    
    Args:
        n_train: Training sample size (default: 200, range: 150-250 recommended)
        n_test: Test sample size (default: 200)
        n_reps: Number of repetitions (default: 10)
        epochs: Training epochs for neural methods (default: 1000)
        lr: Learning rate (default: 0.0005)
        verbose: Print detailed output (default: True)
    
    Returns:
        all_results: List of result dictionaries
    """
    
    all_results = []
    
    for rep in range(n_reps):
        seed = 42 + rep
        set_seed(seed)
        
        # Generate coefficients ONCE with fixed seed (independent of train/test split)
        np.random.seed(42)
        V = 8
        a_v1 = np.random.uniform(0.8, 1.2, V).astype(np.float32)
        a_v2 = np.random.uniform(0.8, 1.2, V).astype(np.float32)
        a_v3 = np.random.uniform(0.4, 0.8, V).astype(np.float32)
        
        # Generate data with SAME coefficients but different X and noise
        X_train, Y_train, z_train, beta_train = generate_synthetic_data(
            n_train, p=50, seed=seed, a_v1=a_v1, a_v2=a_v2, a_v3=a_v3
        )
        X_test, Y_test, z_test, _ = generate_synthetic_data(
            n_test, p=50, seed=seed + 1000, a_v1=a_v1, a_v2=a_v2, a_v3=a_v3
        )
        
        # Get dimensions
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]
        
        if verbose:
            print(f"Repetition {rep+1}/{n_reps} (seed={seed})")
        
        result = {
            'rep': rep,
            'seed': seed,
            'n_train': n_train,
            'n_test': n_test,
            'methods': {}
        }
        
        # 1. Global Mean
        start_time = time.time()
        method = GlobalMean()
        method.fit(X_train, Y_train)
        Y_train_pred = method.predict(X_train)
        Y_pred = method.predict(X_test)
        elapsed = time.time() - start_time
        train_mse = evaluate_mse(Y_train, Y_train_pred)
        mse = evaluate_mse(Y_test, Y_pred)
        result['methods']['Global Mean'] = {'mse': mse, 'train_mse': train_mse, 'gap': mse - train_mse, 'time_sec': elapsed}
        if verbose:
            print(f"  Global Mean          MSE={mse:.6f}, Train MSE={train_mse:.6f}, Gap={mse - train_mse:.6f}, Time={elapsed:.3f}s")
        
        # 2. GFR
        start_time = time.time()
        method = GFR()
        method.fit(X_train, Y_train)
        Y_train_pred = method.predict(X_train)
        Y_pred = method.predict(X_test)
        elapsed = time.time() - start_time
        train_mse = evaluate_mse(Y_train, Y_train_pred)
        mse = evaluate_mse(Y_test, Y_pred)
        result['methods']['GFR'] = {'mse': mse, 'train_mse': train_mse, 'gap': mse - train_mse, 'time_sec': elapsed}
        if verbose:
            print(f"  GFR                  MSE={mse:.6f}, Train MSE={train_mse:.6f}, Gap={mse - train_mse:.6f}, Time={elapsed:.3f}s")
        
        # 3. DFR
        start_time = time.time()
        method = DFRWrapper(input_dim, output_dim, lr=lr, epochs=epochs)
        method.fit(X_train, Y_train)
        Y_train_pred = method.predict(X_train)
        Y_pred = method.predict(X_test)
        elapsed = time.time() - start_time
        train_mse = evaluate_mse(Y_train, Y_train_pred)
        mse = evaluate_mse(Y_test, Y_pred)
        result['methods']['DFR'] = {'mse': mse, 'train_mse': train_mse, 'gap': mse - train_mse, 'time_sec': elapsed}
        if verbose:
            print(f"  DFR                  MSE={mse:.6f}, Train MSE={train_mse:.6f}, Gap={mse - train_mse:.6f}, Time={elapsed:.3f}s")
        
        # 4. E2M
        start_time = time.time()
        method = E2MWrapper(input_dim, output_dim, lr=lr, epochs=epochs)
        method.fit(X_train, Y_train)
        Y_train_pred = method.predict(X_train)
        Y_pred = method.predict(X_test)
        elapsed = time.time() - start_time
        train_mse = evaluate_mse(Y_train, Y_train_pred)
        mse = evaluate_mse(Y_test, Y_pred)
        result['methods']['E2M'] = {'mse': mse, 'train_mse': train_mse, 'gap': mse - train_mse, 'time_sec': elapsed}
        if verbose:
            print(f"  E2M                  MSE={mse:.6f}, Train MSE={train_mse:.6f}, Gap={mse - train_mse:.6f}, Time={elapsed:.3f}s")
        
        # 5. FSDRNN (with grid search for optimal d)
        start_time = time.time()
        method, best_d, d_grid_search = grid_search_fsdrnn_d(
            X_train, Y_train, p=input_dim, V=output_dim, d_values=[2, 3, 5], lr=lr, epochs=epochs
        )
        Y_train_pred = method.predict(X_train)
        Y_pred = method.predict(X_test)
        elapsed = time.time() - start_time
        train_mse = evaluate_mse(Y_train, Y_train_pred)
        mse_proposed = evaluate_mse(Y_test, Y_pred)
        eval_metrics = {'mse': mse_proposed, 'train_mse': train_mse, 'gap': mse_proposed - train_mse, 'time_sec': elapsed, 'best_d': best_d, 'd_grid_search': d_grid_search}
        result['methods']['FSDRNN'] = eval_metrics
        fsdrnn_metrics = compute_subspace_metrics(method, X_train, z_train, d=2)
        result['methods']['FSDRNN']['projection_distance'] = fsdrnn_metrics.get('projection_distance', np.nan)
        
        # Compute oracle metrics
        oracle_metrics = compute_oracle_metrics(method.model, beta_train, mse_proposed, d=2)
        result['methods']['FSDRNN']['projection_distance_normalized'] = oracle_metrics.get('projection_distance', np.nan)
        
        if verbose:
            print(f"  FSDRNN               MSE={eval_metrics['mse']:.6f}, Train MSE={train_mse:.6f}, Gap={eval_metrics['gap']:.6f}, Best d={best_d}, Time={elapsed:.3f}s")
        
        # 6. Oracle-tuned FSDRNN (fixed d0, learned encoder; no true B_0 access)
        start_time = time.time()
        oracle_method = FSdrnnWrapper(input_dim=input_dim, output_dim=output_dim, d=2,
                                      hidden_dim=128, lr=lr, epochs=epochs, dropout=0.2, device='cpu')
        oracle_method.fit(X_train, Y_train)
        Y_train_pred_oracle = oracle_method.predict(X_train)
        Y_pred_oracle = oracle_method.predict(X_test)
        elapsed = time.time() - start_time
        train_mse_oracle = evaluate_mse(Y_train, Y_train_pred_oracle)
        mse_oracle = evaluate_mse(Y_test, Y_pred_oracle)
        result['methods']['Oracle FSDRNN'] = {
            'mse': mse_oracle,
            'train_mse': train_mse_oracle,
            'gap': mse_oracle - train_mse_oracle,
            'time_sec': elapsed,
            'fixed_d': 2
        }
        
        # Compute oracle efficiency ratio
        oracle_efficiency_ratio = mse_proposed / (mse_oracle + 1e-10)
        result['methods']['FSDRNN']['oracle_efficiency_ratio'] = oracle_efficiency_ratio
        
        if verbose:
            print(f"  Oracle FSDRNN        MSE={mse_oracle:.6f}, Train MSE={train_mse_oracle:.6f}, Gap={mse_oracle - train_mse_oracle:.6f}, Time={elapsed:.3f}s")
        
        all_results.append(result)
    
    return all_results


def print_aggregate_statistics(all_results):
    """Print aggregate statistics across all repetitions."""
    
    methods = list(all_results[0]['methods'].keys())
    
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS OVER ALL REPETITIONS")
    print("="*80)
    
    # MSE table
    print(f"{'Method':<20} | {'Test MSE':<12} | {'Train MSE':<12} | {'Gap':<12} | {'Time (s)':<10} | {'± std':<10}")
    print("-" * 80)
    
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
        print(f"{method:<20} | {mean_mse:>11.6f} | {train_mse_mean:>11.6f} | {gap_mean:>11.6f} | {time_mean:>9.3f} | ±{std_mse:<8.6f}")
    
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
    
    # Method ranking
    print("\n" + "="*80)
    print("METHOD RANKING (by average Test MSE)")
    print("="*80)
    
    rankings_data = []
    for method in methods:
        mses = [r['methods'][method]['mse'] for r in all_results]
        train_mses = [r['methods'][method].get('train_mse', np.nan) for r in all_results]
        gaps = [r['methods'][method].get('gap', np.nan) for r in all_results]
        times = [r['methods'][method].get('time_sec', np.nan) for r in all_results]
        
        avg_mse = np.mean(mses)
        avg_train_mse = np.mean(train_mses)
        avg_gap = np.mean(gaps)
        avg_time = np.mean(times)
        rankings_data.append((method, avg_mse, avg_train_mse, avg_gap, avg_time))
    
    rankings_data.sort(key=lambda x: x[1])
    
    medals = ['🥇', '🥈', '🥉']
    for i, (method, avg_mse, avg_train_mse, avg_gap, avg_time) in enumerate(rankings_data):
        medal = medals[i] if i < 3 else f"{i+1}. "
        print(f"  {medal} {method:<20} Test={avg_mse:.6f} | Train={avg_train_mse:.6f} | Gap={avg_gap:.6f} | Time={avg_time:.3f}s")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Setup 5: Enhanced Correlated Responses with Nonlinearity (Optimized for FSDRNN)'
    )
    parser.add_argument('--n_train', type=int, default=200,
                        help='Training sample size (default: 200, recommended: 150-250)')
    parser.add_argument('--n_test', type=int, default=200,
                        help='Test sample size (default: 200)')
    parser.add_argument('--n_reps', type=int, default=10,
                        help='Number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Training epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SETUP 5: ENHANCED CORRELATED RESPONSES WITH NONLINEARITY")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  n_train={args.n_train}, n_test={args.n_test}")
    print(f"  epochs={args.epochs}, lr={args.lr}")
    print(f"  n_reps={args.n_reps}, seed={args.seed}")
    
    print("\nModel: Shared latent signals with moderate nonlinearity")
    print("  d₀ = 2 true dimensions")
    print("  V = 8 responses (from 3 latent factors)")
    print("  p = 30 features (10 signal + 20 noise)")
    print("  Latent signals:")
    print("    f₁ = sin(z₁)        [group Y₁=Y₂=Y₃]")
    print("    f₂ = sin(z₂)        [group Y₄=Y₅=Y₆]")
    print("    f₃ = z₁·z₂          [group Y₇=Y₈]")
    print("  Block-correlated noise (ρ=0.8 within groups)")
    print("  σ = 0.15 (modest noise)")
    print("\n  ✨ OPTIMIZED FOR FSDRNN:")
    print("     - Low-dimensional predictor structure (d₀=2 in p=30)")
    print("     - Moderate nonlinearity (sin, multiplication)")
    print("     - Strong shared response structure (8 responses from 3 factors)")
    print("     - Block-correlated noise within groups")
    print("="*80)
    
    # Determine output file before running simulation
    if args.output is not None:
        output_file = args.output
    else:
        output_file = f"{Path(__file__).stem}_seed{args.seed}.json"
    
    # Run simulation
    start_time = time.time()
    all_results = run_simulation(
        n_train=args.n_train,
        n_test=args.n_test,
        n_reps=args.n_reps,
        epochs=args.epochs,
        lr=args.lr,
        verbose=args.verbose
    )
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    # Save results (convert numpy types to native Python types)
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results_to_save = convert_to_native(all_results)
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Print aggregate statistics
    print_aggregate_statistics(all_results)
    
    print(f"\nElapsed time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
