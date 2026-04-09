"""
Shared utilities for all SDR simulation setups.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class GlobalMean:
    """Global mean predictor."""
    def fit(self, X, Y):
        self.mean_Y = np.mean(Y, axis=0)
    def predict(self, X):
        return np.tile(self.mean_Y, (X.shape[0], 1))


class GFR:
    """Global Fréchet Regression."""
    def fit(self, X, Y):
        self.models = []
        for v in range(Y.shape[1]):
            XtX = X.T @ X
            XtY = X.T @ Y[:, v:v+1]
            try:
                beta = np.linalg.solve(XtX + 1e-6 * np.eye(X.shape[1]), XtY)
            except:
                beta = np.linalg.pinv(XtX) @ XtY
            self.models.append(beta.flatten())
    
    def predict(self, X):
        return np.column_stack([X @ beta for beta in self.models])


class DFR(nn.Module):
    """Deep Fréchet Regression."""
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
    """Wrapper for DFR."""
    def __init__(self, p, V, hidden_dim=32, lr=1e-3, epochs=100, device='cpu'):
        self.p, self.V, self.hidden_dim = p, V, hidden_dim
        self.lr, self.epochs, self.device = lr, epochs, device
        self.models = []
    
    def fit(self, X, Y):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        for v in range(self.V):
            model = DFR(self.p, self.hidden_dim, 1).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            Y_v_torch = torch.tensor(Y[:, v:v+1], dtype=torch.float32).to(self.device)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = criterion(model(X_torch), Y_v_torch)
                loss.backward()
                optimizer.step()
            self.models.append(model.eval())
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return np.column_stack([m(X_torch).cpu().numpy().flatten() for m in self.models])


class E2M(nn.Module):
    """Embedding to Manifold."""
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(output_dim)])
    
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        return torch.cat([head(z) for head in self.heads], dim=1)


class E2MWrapper:
    """Wrapper for E2M."""
    def __init__(self, p, V, latent_dim=3, lr=1e-3, epochs=100, device='cpu'):
        self.p, self.V, self.latent_dim = p, V, latent_dim
        self.lr, self.epochs, self.device = lr, epochs, device
        self.model = None
    
    def fit(self, X, Y):
        self.model = E2M(self.p, self.latent_dim, self.V).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y, dtype=torch.float32).to(self.device)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X_torch), Y_torch)
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_torch).cpu().numpy()


class FSDRNN(nn.Module):
    """FSDRNN with Adaptive LoRA."""
    def __init__(self, input_dim, d, output_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d)
        )
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(output_dim)])
        self.r_max = min(output_dim, 6)
        self.lora_A = nn.Parameter(torch.randn(d, self.r_max) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.r_max, output_dim) * 0.01)
    
    def forward(self, x):
        z = self.encoder(x)
        outputs = torch.cat([head(z) for head in self.heads], dim=1)
        lora = z @ self.lora_A @ self.lora_B
        return outputs + 0.1 * lora


class FSdrnnWrapper:
    """Wrapper for FSDRNN."""
    def __init__(self, p, V, d=2, lr=1e-3, epochs=100, dropout=0.1, device='cpu', verbose=False):
        self.p, self.V, self.d = p, V, d
        self.lr, self.epochs, self.dropout = lr, epochs, dropout
        self.device, self.verbose = device, verbose
        self.model = None
    
    def fit(self, X, Y):
        self.model = FSDRNN(self.p, self.d, self.V, dropout=self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y, dtype=torch.float32).to(self.device)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X_torch), Y_torch)
            loss.backward()
            optimizer.step()
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs} | loss = {loss.item():.6f}")
    
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_torch).cpu().numpy()


def evaluate_prediction(Y_test, Y_pred):
    """Compute MSE and RMSE."""
    mse = np.mean((Y_test - Y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}


def run_experiment(setup_name, data_generator, n_train=300, n_test=150, seed=42, n_reps=3, verbose=True):
    """Generic experiment runner."""
    all_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for rep in range(n_reps):
        if verbose:
            print(f"\n{'#'*70}\nREPETITION {rep+1}/{n_reps}\n{'#'*70}\n")
        
        set_seed(seed + rep)
        
        X_train, Y_train, info_train = data_generator(n_train, seed=seed + rep)
        X_test, Y_test, _ = data_generator(n_test, seed=seed + rep + 1000)
        
        results = {
            'setup': setup_name,
            'rep': rep,
            'seed': seed + rep,
            'n_train': n_train,
            'n_test': n_test,
            **info_train,
            'methods': {}
        }
        
        methods = {
            'Global Mean': GlobalMean(),
            'GFR': GFR(),
            'DFR': DFRWrapper(X_train.shape[1], Y_train.shape[1], lr=1e-3, epochs=200, device=device),
            'E2M': E2MWrapper(X_train.shape[1], Y_train.shape[1], latent_dim=3, lr=1e-3, epochs=200, device=device),
            'FSDRNN': FSdrnnWrapper(X_train.shape[1], Y_train.shape[1], d=2, lr=1e-3, epochs=200, dropout=0.1, device=device)
        }
        
        for method_name, method in methods.items():
            if verbose:
                print(f"Fitting {method_name}...")
            start_time = time.time()
            method.fit(X_train, Y_train)
            Y_pred = method.predict(X_test)
            elapsed = time.time() - start_time
            metrics = evaluate_prediction(Y_test, Y_pred)
            metrics['time_sec'] = elapsed
            results['methods'][method_name] = metrics
            if verbose:
                print(f"  MSE: {metrics['mse']:.6f}, Time: {elapsed:.3f}s")
        
        all_results.append(results)
    
    return all_results
