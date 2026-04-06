# spd_frechet.py
"""
Fréchet mean regression for SPD (Symmetric Positive Definite) matrix-valued responses.

Implements:
  - Five distance metrics on SPD manifold:
      1. Frobenius
      2. Affine-Invariant (Pennec et al., 2006)
      3. Power metric (Dryden et al., 2009)
      4. Log-Cholesky (Lin, 2019)
      5. Bures-Wasserstein (Bhatia et al., 2019)
  - Weighted Fréchet mean computation via iterative gradient descent on the manifold
  - Neural network model that outputs mixture weights for Fréchet mean estimation
  - Data generation: Wishart-distributed SPD matrices with input-dependent scale
"""

from typing import Any, Dict, List, Optional, Literal, Tuple
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

# ======================================================================
# SPD matrix utilities
# ======================================================================

def _sym(A: torch.Tensor) -> torch.Tensor:
    """Symmetrise a matrix (or batch)."""
    return 0.5 * (A + A.transpose(-2, -1))


def _spd_sqrt(A: torch.Tensor) -> torch.Tensor:
    """
    Matrix square root for SPD matrix (or batch) via eigen-decomposition.
    A = Q diag(λ) Q^T  =>  A^{1/2} = Q diag(√λ) Q^T
    """
    eps = 1e-6
    A_reg = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    eigvals, eigvecs = torch.linalg.eigh(A_reg)
    eigvals = eigvals.clamp(min=1e-12)
    return eigvecs @ torch.diag_embed(eigvals.sqrt()) @ eigvecs.transpose(-2, -1)


def _spd_invsqrt(A: torch.Tensor) -> torch.Tensor:
    """Matrix inverse square root for SPD matrix."""
    eps = 1e-6
    A_reg = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    eigvals, eigvecs = torch.linalg.eigh(A_reg)
    eigvals = eigvals.clamp(min=1e-12)
    return eigvecs @ torch.diag_embed(1.0 / eigvals.sqrt()) @ eigvecs.transpose(-2, -1)


def _spd_log(A: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm for SPD matrix."""
    eps = 1e-4
    A_reg = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    eigvals, eigvecs = torch.linalg.eigh(A_reg)
    eigvals = eigvals.clamp(min=1e-12)
    return eigvecs @ torch.diag_embed(eigvals.log()) @ eigvecs.transpose(-2, -1)


def _spd_exp(A: torch.Tensor) -> torch.Tensor:
    """Matrix exponential for symmetric matrix (result is SPD)."""
    eps = 1e-4
    A_reg = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    eigvals, eigvecs = torch.linalg.eigh(A_reg)
    return eigvecs @ torch.diag_embed(eigvals.exp()) @ eigvecs.transpose(-2, -1)


def _spd_pow(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """Matrix power A^alpha for SPD matrix."""
    A = 0.5 * (A + A.transpose(-2, -1))  # symmetrize
    # Add small regularization to ensure numerical stability
    eps = 1e-1
    A_reg = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    eigvals, eigvecs = torch.linalg.eigh(A_reg)
    eigvals = eigvals.clamp(min=1e-12)
    return eigvecs @ torch.diag_embed(eigvals.pow(alpha)) @ eigvecs.transpose(-2, -1)


def _cholesky_log(A: torch.Tensor) -> torch.Tensor:
    """
    Log-Cholesky representation: L = chol(A), then take log of diagonal.
    Returns the lower-triangular matrix with log-transformed diagonal.
    """
    L = torch.linalg.cholesky(A)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    log_diag = diag_L.log()
    # Build result: strictly lower triangle of L + diag(log(diag(L)))
    result = L - torch.diag_embed(diag_L) + torch.diag_embed(log_diag)
    return result


# ======================================================================
# Distance functions on SPD manifold
# ======================================================================

def dist_frobenius(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Frobenius distance: d_F(A, B) = ||A - B||_F
    A, B: [..., p, p]  =>  returns [...]
    """
    diff = A - B
    return torch.sqrt((diff * diff).sum(dim=(-2, -1)).clamp(min=1e-30))


def dist_affine_invariant(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Affine-invariant Riemannian metric (Pennec et al., 2006):
      d_AI(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F
    A, B: [..., p, p]  =>  returns [...]
    """
    A_invsqrt = _spd_invsqrt(A)
    M = A_invsqrt @ B @ A_invsqrt
    M = _sym(M)  # numerical symmetrisation
    log_M = _spd_log(M)
    return torch.sqrt((log_M * log_M).sum(dim=(-2, -1)).clamp(min=1e-30))


def dist_power(A: torch.Tensor, B: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Power metric (Dryden et al., 2009):
      d_alpha(A, B) = (1/alpha) ||A^alpha - B^alpha||_F
    Default alpha=0.5 (square-root Euclidean metric).
    A, B: [..., p, p]  =>  returns [...]
    """
    A_alpha = _spd_pow(A, alpha)
    B_alpha = _spd_pow(B, alpha)
    diff = A_alpha - B_alpha
    return (1.0 / alpha) * torch.sqrt((diff * diff).sum(dim=(-2, -1)).clamp(min=1e-30))


def dist_log_cholesky(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Log-Cholesky metric (Lin, 2019):
      d_LC(A, B) = || logchol(A) - logchol(B) ||_F
    where logchol maps L (Cholesky factor) to (strictly-lower(L), log(diag(L))).
    A, B: [..., p, p]  =>  returns [...]
    """
    lc_A = _cholesky_log(A)
    lc_B = _cholesky_log(B)
    diff = lc_A - lc_B
    return torch.sqrt((diff * diff).sum(dim=(-2, -1)).clamp(min=1e-30))


def dist_bures_wasserstein(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Bures-Wasserstein metric (Bhatia et al., 2019):
      d_BW(A, B)^2 = tr(A) + tr(B) - 2 tr( (A^{1/2} B A^{1/2})^{1/2} )
    A, B: [..., p, p]  =>  returns [...]
    """
    A_sqrt = _spd_sqrt(A)
    inner = A_sqrt @ B @ A_sqrt
    inner = _sym(inner)
    inner_sqrt = _spd_sqrt(inner)
    tr_A = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
    tr_B = torch.diagonal(B, dim1=-2, dim2=-1).sum(-1)
    tr_inner_sqrt = torch.diagonal(inner_sqrt, dim1=-2, dim2=-1).sum(-1)
    d_sq = tr_A + tr_B - 2 * tr_inner_sqrt
    return torch.sqrt(d_sq.clamp(min=1e-30))


def dist_euclidean(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance for scalar or vector responses.

    Supports any shape as long as A and B match:
      - Scalar:  [B]          =>  returns [B]
      - Vector:  [B, q]       =>  returns [B]
      - Matrix:  [B, p, p]    =>  returns [B]  (reduces to Frobenius)

    The Fréchet mean under Euclidean distance is the standard weighted
    arithmetic mean,  so this metric is appropriate for  Y ∈ ℝ  or
    Y ∈ ℝ^q  responses.
    """
    diff = A - B
    if diff.dim() == 1:
        # Scalar responses: diff is [B]
        return diff.abs()
    # Vector / matrix responses: flatten trailing dims
    flat = diff.reshape(diff.shape[0], -1)  # [B, *]
    return torch.sqrt((flat * flat).sum(dim=-1).clamp(min=1e-30))


# Registry
DISTANCE_FUNCTIONS = {
    "frobenius": dist_frobenius,
    "affine_invariant": dist_affine_invariant,
    "power": dist_power,
    "log_cholesky": dist_log_cholesky,
    "bures_wasserstein": dist_bures_wasserstein,
    "euclidean": dist_euclidean,
}


def get_distance_fn(name: str):
    """Return the distance function by name."""
    if name not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown distance: '{name}'. Choose from {list(DISTANCE_FUNCTIONS.keys())}"
        )
    return DISTANCE_FUNCTIONS[name]



# ======================================================================
# Differentiable weighted Fréchet mean from weights
# ======================================================================

def differentiable_frechet_mean(
    weights: torch.Tensor,
    Y_ref: torch.Tensor,
    dist_name: str = "frobenius",
) -> torch.Tensor:
    """
    Compute the weighted Fréchet mean of reference responses Y_ref
    using weights w, in a differentiable manner.

        m_w = argmin_{y} sum_i w_i d^2(y, y_i)

    For ``"euclidean"`` and ``"frobenius"`` the mean is the simple
    weighted arithmetic mean.  For SPD metrics with closed-form
    solutions (Power, Log-Cholesky), the mean is computed in the
    appropriate space.  For Affine-Invariant and Bures-Wasserstein,
    unrolled fixed-point iterations are used so gradients flow.

    Args:
        weights: [B, n] simplex weights (each row sums to 1)
        Y_ref:   responses — one of
                   [n]          scalar
                   [n, q]       vector
                   [n, p, p]    SPD matrix
        dist_name: distance metric name

    Returns:
        mean with the same trailing shape as Y_ref, prepended by B:
          [B],  [B, q],  or  [B, p, p]
    """
    B = weights.shape[0]

    # --- Euclidean / generic weighted arithmetic mean ---
    if dist_name == "euclidean":
        # Works for any Y_ref shape: [n], [n, q], [n, p, p]
        if Y_ref.dim() == 1:
            # Scalar: weights [B, n] * Y_ref [n] -> [B]
            return weights @ Y_ref
        else:
            # Vector or matrix: einsum over the first (sample) axis
            # Y_ref: [n, *rest],  weights: [B, n]
            rest = Y_ref.shape[1:]
            Y_flat = Y_ref.reshape(Y_ref.shape[0], -1)      # [n, D]
            m_flat = weights @ Y_flat                         # [B, D]
            return m_flat.reshape(B, *rest)

    # --- For all SPD metrics below, Y_ref must be [n, p, p] ---
    if Y_ref.dim() == 4:
        # If called with multi-response Y_ref [n, V, p, p], take first response
        Y_ref = Y_ref[:, 0, :, :]
    n, p, _ = Y_ref.shape

    if dist_name == "frobenius":
        # Closed-form: weighted arithmetic mean  m = sum_i w_i Y_i
        # weights[:, :, None, None] * Y_ref[None, :, :, :] -> [B, n, p, p]
        return torch.einsum('bn,nij->bij', weights, Y_ref)  # [B, p, p]

    elif dist_name == "power":
        # Power metric (alpha=0.5): mean in the power space
        #   m^alpha = sum_i w_i Y_i^alpha, then m = (m^alpha)^{1/alpha}
        alpha = 0.5
        # Ensure Y_ref is positive definite
        eps_ref = 1e-6
        Y_ref_reg = Y_ref + eps_ref * torch.eye(Y_ref.shape[-1], device=Y_ref.device, dtype=Y_ref.dtype).unsqueeze(0).expand(Y_ref.shape[0], -1, -1)
        # Use precomputed Y_ref^alpha if available (stored as attribute)
        if hasattr(Y_ref, '_powered') and Y_ref._powered is not None:
            Y_powered = Y_ref._powered
        else:
            Y_powered = _spd_pow(Y_ref_reg, alpha)  # [n, p, p]  (batched)
        mean_powered = torch.einsum('bn,nij->bij', weights, Y_powered)  # [B, p, p]
        mean_powered = _sym(mean_powered)
        # Ensure positive definiteness by adding regularization
        eps_pd = 1e-2
        mean_powered = mean_powered + eps_pd * torch.eye(mean_powered.shape[-1], device=mean_powered.device, dtype=mean_powered.dtype).unsqueeze(0).expand(B, -1, -1)
        # Add small random perturbation to avoid repeated eigenvalues
        noise = torch.randn_like(mean_powered) * 1e-4
        noise = (noise + noise.transpose(-2, -1)) / 2  # symmetrize noise
        mean_powered = mean_powered + noise
        # m = mean_powered^{1/alpha}
        result = []
        for b in range(B):
            result.append(_spd_pow(mean_powered[b], 1.0 / alpha))
        return torch.stack(result)  # [B, p, p]

    elif dist_name == "log_cholesky":
        # Average in Log-Cholesky space, then map back
        lc_ref = torch.stack([_cholesky_log(Y_ref[i]) for i in range(n)])  # [n, p, p]
        lc_mean = torch.einsum('bn,nij->bij', weights, lc_ref)  # [B, p, p]
        result = []
        for b in range(B):
            diag_vals = torch.diagonal(lc_mean[b], dim1=-2, dim2=-1)
            exp_diag = diag_vals.exp()
            L_b = lc_mean[b] - torch.diag_embed(diag_vals) + torch.diag_embed(exp_diag)
            result.append(L_b @ L_b.t())
        return torch.stack(result)  # [B, p, p]

    elif dist_name == "affine_invariant":
        # Unrolled tangent-space iterations (differentiable)
        # Init with arithmetic mean
        mu = torch.einsum('bn,nij->bij', weights, Y_ref)  # [B, p, p]
        n_iter = 5  # few unrolled steps
        for _ in range(n_iter):
            result = []
            for b in range(B):
                mu_b = mu[b]  # [p, p]
                mu_invsqrt = _spd_invsqrt(mu_b)
                mu_sqrt = _spd_sqrt(mu_b)
                tangent_avg = torch.zeros(p, p, dtype=mu.dtype, device=mu.device)
                for i in range(n):
                    transported = mu_invsqrt @ Y_ref[i] @ mu_invsqrt
                    transported = _sym(transported)
                    tangent_avg = tangent_avg + weights[b, i] * _spd_log(transported)
                mu_new = mu_sqrt @ _spd_exp(tangent_avg) @ mu_sqrt
                mu_new = _sym(mu_new)
                result.append(mu_new)
            mu = torch.stack(result)  # [B, p, p]
        return mu

    elif dist_name == "bures_wasserstein":
        # Unrolled fixed-point iteration (differentiable)
        mu = torch.einsum('bn,nij->bij', weights, Y_ref)  # [B, p, p] init
        n_iter = 5
        for _ in range(n_iter):
            result = []
            for b in range(B):
                mu_b = mu[b]
                mu_sqrt = _spd_sqrt(mu_b)
                mu_invsqrt = _spd_invsqrt(mu_b)
                S = torch.zeros(p, p, dtype=mu.dtype, device=mu.device)
                for i in range(n):
                    inner = mu_sqrt @ Y_ref[i] @ mu_sqrt
                    inner = _sym(inner)
                    S = S + weights[b, i] * _spd_sqrt(inner)
                mu_new = mu_invsqrt @ S @ S @ mu_invsqrt
                mu_new = _sym(mu_new)
                result.append(mu_new)
            mu = torch.stack(result)  # [B, p, p]
        return mu

    else:
        raise ValueError(f"Unknown distance: {dist_name}")


def entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Shannon entropy of weight vectors:  H(w) = -sum_i w_i log(w_i)
    Args:
        weights: [B, n] simplex weights
    Returns:
        [B] entropy per sample
    """
    return -(weights * (weights + eps).log()).sum(dim=-1)  # [B]


# ======================================================================
# Data generation: Wishart SPD responses with input-dependent scale
# ======================================================================

def generate_scale_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Paper SPD toy (pages 19-20): build diagonal Σ(X) for X ∈ R^12.

    X1~U(0,1), X2~U(-1/2,1/2), X3~U(1,2),
    X4~Gamma(3,2), X5~Gamma(4,2), X6~Gamma(5,2),
    X7,X8,X9~N(0,1), X10~Ber(0.4), X11~Ber(0.5), X12~Ber(0.6).

    Σ is diagonal with entries:
      Σ11 = { sin(πX1) X10 + cos(πX2)(1−X10) }^2
      Σ22 = sin^2(πX1) cos^2(πX2)
      Σ33 = { (X4/X5)(1/10) X11 + sqrt(X5/X4)(1/10)(1−X11) }^2
      Σ44 = |X7 X8| / 25
      Σ55 = |X9 / X6| / 9

    Args:
        X: [12] or [B, 12] input vector(s)

    Returns:
        Sigma: [5, 5] or [B, 5, 5] diagonal SPD scale matrix
    """
    squeeze = False
    if X.dim() == 1:
        X = X.unsqueeze(0)
        squeeze = True

    eps = 1e-10
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12 = [
        X[:, i] for i in range(12)
    ]

    s11 = (torch.sin(np.pi * X1) * X10 + torch.cos(np.pi * X2) * (1.0 - X10)) ** 2
    s22 = (torch.sin(np.pi * X1) ** 2) * (torch.cos(np.pi * X2) ** 2)

    s33 = (
        (X4 / (X5 + eps)) * 0.1 * X11
        + torch.sqrt((X5 + eps) / (X4 + eps)) * 0.1 * (1.0 - X11)
    ) ** 2

    s44 = (X7 * X8).abs() / 25.0
    s55 = (X9 / (X6 + eps)).abs() / 9.0

    diag = torch.stack([s11, s22, s33, s44, s55], dim=-1).clamp(min=1e-8)
    Sigma = torch.diag_embed(diag)  # [B, 5, 5]

    if squeeze:
        Sigma = Sigma.squeeze(0)
    return Sigma


def sample_wishart(Sigma: torch.Tensor, df: int, num_samples: int = 1) -> torch.Tensor:
    """
    Sample from Wishart distribution W(Σ, df) using the Bartlett decomposition.

    For Wishart(Σ, df), we can write Y = L A A^T L^T where L = chol(Σ)
    and A is a lower triangular matrix from the Bartlett decomposition.

    Args:
        Sigma: [p, p] scale matrix (SPD)
        df: degrees of freedom (must be >= p)
        num_samples: number of independent samples

    Returns:
        Y: [num_samples, p, p] SPD matrices sampled from W(Σ, df)
           or [p, p] if num_samples == 1
    """
    p = Sigma.shape[-1]
    assert df >= p, f"df={df} must be >= p={p} for Wishart"
    L = torch.linalg.cholesky(Sigma)  # [p, p]

    samples = []
    for _ in range(num_samples):
        # Bartlett decomposition
        A = torch.zeros(p, p, dtype=Sigma.dtype, device=Sigma.device)
        for i in range(p):
            # Diagonal: sqrt of chi-squared with (df - i) degrees of freedom
            chi2 = torch.distributions.Chi2(df=df - i)
            A[i, i] = chi2.sample().sqrt()
            # Below diagonal: standard normal
            for j in range(i):
                A[i, j] = torch.randn(1, dtype=Sigma.dtype, device=Sigma.device).squeeze()
        # Y = L A A^T L^T
        LA = L @ A
        Y = LA @ LA.t()
        samples.append(Y)

    result = torch.stack(samples)
    if num_samples == 1:
        return result.squeeze(0)
    return result


class WishartSPDDataset(Dataset):
    """
    Synthetic dataset for Fréchet mean regression with SPD matrix responses.

    Data generation (paper pages 19-20):
      - X ∈ R^12 with:
          X1 ~ U(0,1),  X2 ~ U(-1/2, 1/2),  X3 ~ U(1,2),
          X4 ~ Gamma(3,2),  X5 ~ Gamma(4,2),  X6 ~ Gamma(5,2),
          X7 ~ N(0,1),  X8 ~ N(0,1),  X9 ~ N(0,1),
          X10 ~ Ber(0.4),  X11 ~ Ber(0.5),  X12 ~ Ber(0.6)
      - Σ(X) = diag(Σ11, ..., Σ55) with input-dependent entries
      - Y ~ Wishart(Σ(X), df)  — V × 5×5 SPD matrix responses per sample

    Args:
        n: number of data points
        n_responses: number of responses per sample (V)
        df: degrees of freedom for Wishart distribution (default: 6)
        seed: random seed for reproducibility
    """

    def __init__(self, n: int, n_responses: int = 1, df: int = 6, seed: int = 42):
        super().__init__()
        self.n = n
        self.n_responses = n_responses
        self.df = df
        self.p = 5  # SPD matrix dimension

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate X: [n, 12] as in the paper
        X1  = torch.rand(n)                        # U(0,1)
        X2  = torch.rand(n) - 0.5                  # U(-1/2, 1/2)
        X3  = torch.rand(n) + 1.0                  # U(1,2)

        # Gamma(k, rate=2) => scale=1/2
        X4  = torch.from_numpy(np.random.gamma(shape=3.0, scale=1.0/2.0, size=n)).float()  # Gamma(3,2)
        X5  = torch.from_numpy(np.random.gamma(shape=4.0, scale=1.0/2.0, size=n)).float()  # Gamma(4,2)
        X6  = torch.from_numpy(np.random.gamma(shape=5.0, scale=1.0/2.0, size=n)).float()  # Gamma(5,2)

        X7  = torch.randn(n)                       # N(0,1)
        X8  = torch.randn(n)                       # N(0,1)
        X9  = torch.randn(n)                       # N(0,1)

        X10 = torch.bernoulli(0.4 * torch.ones(n)) # Ber(0.4)
        X11 = torch.bernoulli(0.5 * torch.ones(n)) # Ber(0.5)
        X12 = torch.bernoulli(0.6 * torch.ones(n)) # Ber(0.6)

        self.X = torch.stack([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12], dim=1)  # [n,12]

        # Generate Σ(X) for each sample
        self.Sigma = generate_scale_matrix(self.X)  # [n, 5, 5]

        # Sample Y ~ Wishart(Σ_i, df) for each sample and each response
        Y_list = []
        for i in range(n):
            Yi_list = []
            for v in range(n_responses):
                Yiv = sample_wishart(self.Sigma[i], df=self.df, num_samples=1)
                Yi_list.append(Yiv)
            Y_list.append(torch.stack(Yi_list))  # [V, 5, 5]
        self.Y = torch.stack(Y_list)  # [n, V, 5, 5]

        # Compute the true conditional Fréchet mean for each x and response
        # For Wishart(Σ, df), E[Y] = df * Σ (the true Frobenius Fréchet mean)
        self.true_mean = self.df * self.Sigma.unsqueeze(1).expand(-1, n_responses, -1, -1)  # [n, V, 5, 5]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_true_mean(self, idx=None):
        """Return E[Y|X] = df * Σ(X), which is the population Fréchet mean under Frobenius."""
        if idx is not None:
            return self.true_mean[idx]
        return self.true_mean


# ======================================================================
# Section 6.3 — SDR dataset: symmetric matrix-variate Normal
# (Dimension Reduction for Fréchet Regression, Scenario II)
# ======================================================================

def _sym_mat_normal(M: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """
    Sample from symmetric matrix-variate Normal N_{rr}(M, σ²).

    Z has independent N(0,1) diagonal and N(0,1/2) off-diagonal entries,
    so Y = σZ + M is symmetric with log(Y) ~ N_{rr}(M, σ²).

    Args:
        M: [r, r] symmetric mean matrix
        sigma: scalar standard deviation (paper uses σ = 0.5, i.e. σ² = 0.25)

    Returns:
        Y: [r, r] symmetric matrix sample
    """
    r = M.shape[0]
    # Diagonal ~ N(0, 1), off-diagonal ~ N(0, 1/2)
    Z = torch.zeros(r, r, dtype=M.dtype)
    for i in range(r):
        Z[i, i] = torch.randn(1).item()
        for j in range(i + 1, r):
            val = torch.randn(1).item() / np.sqrt(2)
            Z[i, j] = val
            Z[j, i] = val
    return sigma * Z + M


class SDRCorrSPDDataset(Dataset):
    """
    Synthetic dataset for the SDR simulation (Section 6.3, Scenario II).

    Generates correlation/SPD matrix responses that depend on X ∈ ℝᵖ only
    through a low-dimensional linear projection B₀ᵀX.

    **Model II-1** (d₀ = 1, r = 2):

        D(X) = [[1, ρ(X)], [ρ(X), 1]],   ρ(X) = tanh(β₁ᵀX / 2)

    **Model II-2** (d₀ = 2, r = 3):

        D(X) = [[1, ρ₁, ρ₂], [ρ₁, 1, ρ₁], [ρ₂, ρ₁, 1]]
        ρ₁(X) = 0.4·tanh(β₁ᵀX / 2),  ρ₂(X) = 0.4·sin(β₃ᵀX)

    Response:  log(Y) ~ N_{rr}(log D(X), 0.25)   (symmetric matrix-variate
    Normal with σ = 0.5),  so  Y = exp(log D(X) + 0.5·Z) ∈ SPD.

    Directions (p = 10):
        β₁ = (1,1,0,...,0),  β₂ = (0,...,0,1,1),
        β₃ = (1,2,0,...,0,2),  β₄ = (0,0,1,2,2,...,0)

    Args:
        n:       number of observations
        p:       predictor dimension (default 10)
        model:   ``'II-1'`` or ``'II-2'``
        setting: ``'a'`` (X ~ N(0,I)) or ``'b'`` (correlated non-elliptical X)
        sigma:   noise standard deviation for the matrix-variate Normal (0.5)
        n_responses: number of responses per sample
        seed:    random seed
    """

    def __init__(
        self,
        n: int,
        p: int = 10,
        model: str = "II-2",
        setting: str = "a",
        sigma: float = 0.5,
        n_responses: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.n = n
        self.p = p
        self.model_name = model
        self.setting = setting
        self.sigma = sigma
        self.n_responses = n_responses

        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- Directions (length p) ---
        beta1 = np.zeros(p); beta1[0] = 1; beta1[1] = 1
        beta2 = np.zeros(p); beta2[-2] = 1; beta2[-1] = 1
        beta3 = np.zeros(p); beta3[0] = 1; beta3[1] = 2; beta3[-1] = 2
        beta4 = np.zeros(p); beta4[1] = 0; beta4[2] = 1; beta4[3] = 2
        if p > 4:
            beta4[4] = 2

        self.beta1 = torch.from_numpy(beta1).float()
        self.beta2 = torch.from_numpy(beta2).float()
        self.beta3 = torch.from_numpy(beta3).float()
        self.beta4 = torch.from_numpy(beta4).float()

        # --- Generate X ---
        if setting == "a":
            self.X = torch.randn(n, p)
        else:
            # Setting (b): AR(1) with Σ_{ij} = 0.5^|i-j|, then transform
            cov = np.array([[0.5 ** abs(i - j) for j in range(p)] for i in range(p)])
            L = np.linalg.cholesky(cov)
            U = (L @ np.random.randn(p, n)).T  # [n, p]
            X_np = U.copy()
            X_np[:, 0] = np.sin(U[:, 0])
            X_np[:, 1] = np.abs(U[:, 1])
            # X_np[:, 2:] unchanged
            self.X = torch.from_numpy(X_np).float()

        # --- Matrix dimension ---
        if model == "II-1":
            self.mat_dim = 2
            self.structural_dim = 1  # d₀
        elif model == "II-2":
            self.mat_dim = 3
            self.structural_dim = 2  # d₀
        else:
            raise ValueError(f"Unknown model: {model}. Use 'II-1' or 'II-2'.")

        # Store the true central subspace basis for evaluation
        if model == "II-1":
            self.B0 = self.beta1.unsqueeze(1)  # [p, 1]
        else:
            self.B0 = torch.stack([self.beta1, self.beta3], dim=1)  # [p, 2]

        # --- Generate D(X) and Y ---
        Y_list = []
        D_list = []
        for i in range(n):
            xi = self.X[i]
            D = self._build_D(xi)
            D_list.append(D)

            # Generate V responses
            Y_v_list = []
            for v in range(n_responses):
                # log(Y) ~ N_{rr}(log D, sigma^2)
                log_D = _spd_log(D)
                log_Y = _sym_mat_normal(log_D, sigma=self.sigma)
                Y = _spd_exp(log_Y)
                Y_v_list.append(Y)
            Y_list.append(torch.stack(Y_v_list))  # [V, r, r]

        self.Y = torch.stack(Y_list).float()      # [n, V, r, r]
        self.D_true = torch.stack(D_list).float().unsqueeze(1).expand(-1, n_responses, -1, -1)  # [n, V, r, r]

    def _build_D(self, x: torch.Tensor) -> torch.Tensor:
        """Build the conditional centre matrix D(x)."""
        if self.model_name == "II-1":
            z1 = (self.beta1 @ x).item()
            rho = (np.exp(z1) - 1) / (np.exp(z1) + 1)      # tanh(z1/2)-ish
            D = torch.tensor([[1.0, rho], [rho, 1.0]])
        else:  # II-2
            z1 = (self.beta1 @ x).item()
            z3 = (self.beta3 @ x).item()
            rho1 = 0.4 * (np.exp(z1) - 1) / (np.exp(z1) + 1)
            rho2 = 0.4 * np.sin(z3)
            D = torch.tensor([
                [1.0,  rho1, rho2],
                [rho1, 1.0,  rho1],
                [rho2, rho1, 1.0 ],
            ])
        # Ensure SPD (eigenvalue floor)
        eps = 1e-6
        D_reg = D + eps * torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
        eigvals, eigvecs = torch.linalg.eigh(D_reg)
        eigvals = eigvals.clamp(min=1e-6)
        D = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        return D

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_D_true(self, idx=None):
        """Return the conditional centre matrix D(X)."""
        if idx is not None:
            return self.D_true[idx]
        return self.D_true


# ======================================================================
# Pairwise distance matrix computation
# ======================================================================

def compute_pairwise_distances(
    Y: torch.Tensor,
    dist_name: str = "frobenius",
) -> torch.Tensor:
    """
    Compute the n×n pairwise distance matrix for SPD matrices.

    For Frobenius and Power metrics the computation is vectorised via
    ``torch.cdist``.  Other metrics fall back to an O(n²) loop.

    Args:
        Y: [n, p, p] SPD matrices
        dist_name: distance metric name

    Returns:
        D: [n, n] symmetric distance matrix with zeros on the diagonal
    """
    n = Y.shape[0]

    if dist_name == "frobenius":
        Y_flat = Y.reshape(n, -1)
        return torch.cdist(Y_flat, Y_flat)

    if dist_name == "power":
        alpha = 0.5
        Y_alpha = _spd_pow(Y, alpha)
        Y_flat = Y_alpha.reshape(n, -1)
        return (1.0 / alpha) * torch.cdist(Y_flat, Y_flat)

    if dist_name == "log_cholesky":
        lc = torch.stack([_cholesky_log(Y[i]) for i in range(n)])
        lc_flat = lc.reshape(n, -1)
        return torch.cdist(lc_flat, lc_flat)

    # General fallback — loop
    dist_fn = get_distance_fn(dist_name)
    D = torch.zeros(n, n, dtype=Y.dtype)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(Y[i].unsqueeze(0), Y[j].unsqueeze(0)).squeeze()
            D[i, j] = d
            D[j, i] = d
    return D


# ======================================================================
# Point-wise weighted Fréchet mean (supports negative weights)
# ======================================================================

def weighted_frechet_mean_pointwise(
    weights: torch.Tensor,
    Y: torch.Tensor,
    dist_name: str = "frobenius",
) -> torch.Tensor:
    """
    Compute the weighted Fréchet mean for a *single* set of weights.

    Supports the ``"euclidean"`` metric (scalar/vector responses) as
    well as SPD metrics with closed-form solutions (Frobenius, Power,
    Log-Cholesky).  For Affine-Invariant / Bures-Wasserstein the weights
    are clamped to the simplex and the iterative solver from
    ``differentiable_frechet_mean`` is used.

    Args:
        weights: [n] weight vector (may contain negative values)
        Y:       [n, p, p] SPD matrices  **or**  [n] / [n, q] for euclidean
        dist_name: distance metric

    Returns:
        mean: same trailing shape as Y
    """
    if dist_name == "euclidean":
        if Y.dim() == 1:
            return (weights * Y).sum()
        return torch.einsum("n,n...->...", weights, Y)

    if dist_name == "frobenius":
        return torch.einsum("n,nij->ij", weights, Y)

    if dist_name == "power":
        alpha = 0.5
        Y_alpha = _spd_pow(Y, alpha)
        m_alpha = torch.einsum("n,nij->ij", weights, Y_alpha)
        m_alpha = _sym(m_alpha)
        return _spd_pow(m_alpha, 1.0 / alpha)

    if dist_name == "log_cholesky":
        n = Y.shape[0]
        lc = torch.stack([_cholesky_log(Y[i]) for i in range(n)])
        lc_m = torch.einsum("n,nij->ij", weights, lc)
        diag_vals = torch.diagonal(lc_m)
        L = lc_m - torch.diag(diag_vals) + torch.diag(diag_vals.exp())
        return L @ L.t()

    # Fallback: clamp weights to simplex and use iterative solver
    w = weights.clamp(min=0.0)
    w_sum = w.sum()
    if w_sum > 1e-12:
        w = w / w_sum
    else:
        w = torch.ones_like(weights) / len(weights)
    return differentiable_frechet_mean(
        w.unsqueeze(0), Y, dist_name
    ).squeeze(0)


# ======================================================================
# Global Fréchet Regression  (Petersen & Müller, 2019)
# ======================================================================

class GlobalFrechetRegression:
    """
    Global Fréchet Regression for SPD matrix responses.

    Model (Petersen & Müller, 2019):

        m̂(x) = argmin_{M ∈ SPD} Σ_i  ŝ_i(x) d²(M, Y_i)

    where the weights are

        ŝ_i(x) = (1/n) [ 1 + (x − X̄)ᵀ Σ̂⁻¹ (X_i − X̄) ]

    These are standard OLS hat-matrix entries and can be negative.

    Args:
        dist_name: SPD distance metric name
    """

    def __init__(self, dist_name: str = "frobenius"):
        self.dist_name = dist_name
        self.X_train: Optional[torch.Tensor] = None
        self.Y_train: Optional[torch.Tensor] = None
        self.X_mean: Optional[torch.Tensor] = None
        self.Sigma_inv: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """Store training data and compute centering/covariance.

        Args:
            X: [n, p] predictor matrix
            Y: [n, d, d] or [n, V, d, d] SPD matrix responses
        """
        self.X_train = X.float()
        self.Y_train = Y.float()
        n = X.shape[0]

        self.X_mean = self.X_train.mean(dim=0)
        X_c = self.X_train - self.X_mean
        Sigma = (X_c.T @ X_c) / n
        Sigma += 1e-6 * torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
        self.Sigma_inv = torch.linalg.inv(Sigma)

    @torch.no_grad()
    def predict(self, X_out: torch.Tensor) -> torch.Tensor:
        """Predict SPD matrices at new predictor values.

        Args:
            X_out: [m, p]

        Returns:
            [m, d, d] or [m, V, d, d] predicted SPD matrices
        """
        device = X_out.device
        self.X_train = self.X_train.to(device)
        self.X_mean = self.X_mean.to(device)
        self.Sigma_inv = self.Sigma_inv.to(device)
        self.Y_train = self.Y_train.to(device)

        n = self.X_train.shape[0]
        X_out = X_out.float()
        X_c = self.X_train - self.X_mean  # [n, p]

        if self.Y_train.dim() == 3:
            # Single response
            preds = []
            for j in range(X_out.shape[0]):
                x_c = X_out[j] - self.X_mean  # [p]
                inner = X_c @ (self.Sigma_inv @ x_c)  # [n]
                s = (1.0 / n) * (1.0 + inner)  # [n]
                pred = weighted_frechet_mean_pointwise(s, self.Y_train, self.dist_name)
                preds.append(pred)
            return torch.stack(preds)  # [m, d, d]
        else:
            # Multi-response
            V = self.Y_train.shape[1]
            preds = []
            for j in range(X_out.shape[0]):
                x_c = X_out[j] - self.X_mean  # [p]
                inner = X_c @ (self.Sigma_inv @ x_c)  # [n]
                s = (1.0 / n) * (1.0 + inner)  # [n]
                pred_v = []
                for v in range(V):
                    Y_v = self.Y_train[:, v, :, :]  # [n, d, d]
                    pred_v.append(weighted_frechet_mean_pointwise(s, Y_v, self.dist_name))
                preds.append(torch.stack(pred_v))  # [V, d, d]
            return torch.stack(preds)  # [m, V, d, d]


# ======================================================================
# DNN regressor helper for Deep Fréchet Regression
# ======================================================================

class _DNNRegressor(nn.Module):
    """
    Simple feed-forward DNN for scalar regression  X → z_j.

    Used internally by :class:`DeepFrechetRegression` to learn each
    coordinate of the manifold embedding.
    """

    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden: int = 32, layer: int = 4, dropout: float = 0.0):
        super().__init__()
        modules: list = []
        prev = input_dim
        for _ in range(layer):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            prev = hidden
        modules.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# Deep Fréchet Regression  (based on DFR, R package)
# ======================================================================

class DeepFrechetRegression:
    """
    Deep Fréchet Regression (DFR) for SPD matrix responses.

    Algorithm:
      1. Compute the n×n pairwise distance matrix of the training
         responses Y using the chosen SPD metric.
      2. Apply manifold learning (ISOMAP by default) to embed the
         responses into ℝʳ  (r ≤ 2).
      3. Standardise the embedding coordinates Z.
      4. Train r independent DNNs  f_j : X → z_j   (j = 1, …, r).
      5. At prediction time, estimate ẑ = f(x) for each test point,
         then compute Nadaraya–Watson kernel weights in the embedding
         space and return the kernel-weighted Fréchet mean of the
         training responses.

    Args:
        dist_name:       SPD distance metric
        manifold_method: ``'isomap'`` (default), ``'tsne'``, or ``'umap'``
        manifold_dim:    dimension r of the embedding (default 2)
        manifold_k:      number of neighbours for ISOMAP (default 10)
        hidden:          neurons per hidden layer in each DNN
        layer:           number of hidden layers
        num_epochs:      training epochs for the DNN
        lr:              learning rate
        dropout:         dropout probability
        bw:              bandwidth for kernel regression (auto if ``None``)
        seed:            random seed
        device:          device for computations (default: infer from input)
    """

    def __init__(
        self,
        dist_name: str = "frobenius",
        manifold_method: str = "isomap",
        manifold_dim: int = 2,
        manifold_k: int = 10,
        hidden: int = 32,
        layer: int = 4,
        num_epochs: int = 2000,
        lr: float = 5e-4,
        dropout: float = 0.0,
        bw: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.dist_name = dist_name
        self.manifold_method = manifold_method
        self.manifold_dim = manifold_dim
        self.manifold_k = manifold_k
        self.hidden = hidden
        self.layer = layer
        self.num_epochs = num_epochs
        self.lr = lr
        self.dropout = dropout
        self.bw = bw
        self.seed = seed if seed is not None else np.random.randint(1000)
        self.device = device or "cpu"

        # Populated by fit()
        self.X_train: Optional[torch.Tensor] = None
        self.Y_train: Optional[torch.Tensor] = None
        self.Z_pred_train: Optional[torch.Tensor] = None
        self._dnn_models: List[_DNNRegressor] = []
        self._bw: Optional[np.ndarray] = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor, verbose: bool = False):
        """
        Fit the DFR pipeline (manifold learning → DNN → bandwidth).

        Args:
            X: [n, p] predictors
            Y: [n, d, d] SPD matrix responses
            verbose: print training progress
        """
        from sklearn.manifold import Isomap

        self.X_train = X.float()
        self.Y_train = Y.float()
        if self.device is None or self.device == "cpu":
            self.device = X.device
        self.device = torch.device(self.device)
        n = X.shape[0]
        r = self.manifold_dim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- Step 1: pairwise distance matrix ---
        if verbose:
            print("  [DFR] computing pairwise distance matrix …")
        if Y.dim() == 3:
            D = compute_pairwise_distances(Y, self.dist_name).cpu().numpy()
        else:
            # Multi-response: average distance matrices over responses
            V = Y.shape[1]
            D_list = []
            for v in range(V):
                Y_v = Y[:, v, :, :]
                D_v = compute_pairwise_distances(Y_v, self.dist_name)
                D_list.append(D_v)
            D = torch.stack(D_list).mean(dim=0).cpu().numpy()

        # --- Step 2: manifold learning ---
        if verbose:
            print(f"  [DFR] manifold learning ({self.manifold_method}, r={r}) …")

        if self.manifold_method == "isomap":
            iso = Isomap(
                n_neighbors=min(self.manifold_k, n - 1),
                n_components=r,
                metric="precomputed",
            )
            Z = iso.fit_transform(D)
        elif self.manifold_method == "tsne":
            from sklearn.manifold import TSNE
            Z = TSNE(
                n_components=r, metric="precomputed", init="random"
            ).fit_transform(D)
        else:
            raise ValueError(f"Unsupported manifold method: {self.manifold_method}")

        # --- Step 3: standardise Z ---
        Z_mean = Z.mean(axis=0)
        Z_std = Z.std(axis=0)
        Z_std[Z_std == 0] = 1.0
        Z_scaled = (Z - Z_mean) / Z_std
        Z_tensor = torch.from_numpy(Z_scaled).float()

        # --- Step 4: train r DNNs ---
        X_tensor = self.X_train.float()
        self._dnn_models = []
        for dim_idx in range(r):
            if verbose:
                print(f"  [DFR] training DNN for Z dimension {dim_idx + 1}/{r} …")
            z_target = Z_tensor[:, dim_idx : dim_idx + 1]  # [n, 1]
            model = _DNNRegressor(
                input_dim=X.shape[1],
                output_dim=1,
                hidden=self.hidden,
                layer=self.layer,
                dropout=self.dropout,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            model.train()
            for ep in range(1, self.num_epochs + 1):
                pred = model(X_tensor)
                loss = F.mse_loss(pred, z_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose and ep % max(1, self.num_epochs // 5) == 0:
                    print(f"       epoch {ep}/{self.num_epochs}  MSE={loss.item():.6f}")
            model.eval()
            self._dnn_models.append(model)

        # --- Step 5: training-set predictions of Z ---
        with torch.no_grad():
            self.Z_pred_train = torch.cat(
                [m(X_tensor) for m in self._dnn_models], dim=1
            )  # [n, r]

        # Move to device
        self._dnn_models = [m.to(self.device) for m in self._dnn_models]
        self.Z_pred_train = self.Z_pred_train.to(self.device)
        self.Y_train = self.Y_train.to(self.device)

        # --- Step 6: bandwidth ---
        if self.bw is not None:
            self._bw = np.asarray(self.bw, dtype=np.float64)
        else:
            rg = (
                self.Z_pred_train.max(dim=0).values
                - self.Z_pred_train.min(dim=0).values
            ).cpu().numpy()
            self._bw = rg * 0.1
            self._bw[self._bw == 0] = 0.1

        if verbose:
            print(f"  [DFR] bandwidth = {self._bw}")

    @torch.no_grad()
    def predict(self, X_out: torch.Tensor) -> torch.Tensor:
        """Predict SPD matrices for new predictor values.

        Args:
            X_out: [m, p]

        Returns:
            [m, d, d] predicted SPD matrices
        """
        device = X_out.device
        self._dnn_models = [m.to(device) for m in self._dnn_models]
        self.Z_pred_train = self.Z_pred_train.to(device)
        self.Y_train = self.Y_train.to(device)
        X_out = X_out.float()
        Z_out = torch.cat(
            [m(X_out) for m in self._dnn_models], dim=1
        ).cpu().numpy()  # [m, r]
        Z_train = self.Z_pred_train.cpu().numpy()  # [n, r]

        preds = []
        for j in range(X_out.shape[0]):
            diff = (Z_train - Z_out[j]) / self._bw  # [n, r]
            kvals = np.exp(-0.5 * np.sum(diff ** 2, axis=1))  # [n]
            w_sum = kvals.sum()
            if w_sum > 1e-12:
                w = kvals / w_sum
            else:
                w = np.ones(len(kvals)) / len(kvals)
            w_tensor = torch.from_numpy(w).float().to(device)
            if self.Y_train.dim() == 3:
                pred = weighted_frechet_mean_pointwise(
                    w_tensor, self.Y_train, self.dist_name
                )
            else:
                V = self.Y_train.shape[1]
                pred_v = []
                for v in range(V):
                    Y_v = self.Y_train[:, v, :, :]
                    pred_v.append(weighted_frechet_mean_pointwise(
                        w_tensor, Y_v, self.dist_name
                    ))
                pred = torch.stack(pred_v)
            preds.append(pred)
        return torch.stack(preds)


# ======================================================================
# Neural network for Fréchet weight prediction  (w_θ)
# ======================================================================

class FrechetWeightNet(nn.Module):
    """
    Neural network w_θ(X) that outputs simplex weights over a fixed
    reference set of n training SPD matrices.

    Architecture:
      - MLP backbone:  X  →  hidden features
      - Linear head:   features  →  logits  ∈ R^n_ref
      - Softmax:       logits  →  w  ∈ Δ^{n_ref-1}

    Given the weights, the weighted Fréchet mean is computed externally:
        m_θ(X) = argmin_y  Σ_i w_i(X) d²(y, Y_i)

    Args:
        input_dim:    dimensionality of X
        n_ref:        number of reference SPD matrices (= training set size)
        hidden_sizes: list of hidden layer sizes for the backbone
        activation:   'relu', 'gelu', or 'tanh'
        dropout:      dropout probability
    """

    def __init__(
        self,
        input_dim: int = 12,
        n_ref: int = 200,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_ref = n_ref

        # Backbone MLP
        hs = hidden_sizes or [64, 64]
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers = []
        prev = input_dim
        for h in hs:
            layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Output head: logits over the n_ref reference points
        self.head = nn.Linear(prev, n_ref)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute simplex weights w_θ(X) over the reference set.
        x: [B, input_dim]
        Returns: [B, n_ref]  weights on the probability simplex
        """
        h = self.backbone(x)
        logits = self.head(h)          # [B, n_ref]
        w = F.softmax(logits, dim=-1)  # [B, n_ref]
        return w


class FrechetBottleneckNet(nn.Module):
    """
    Fréchet weight network with an information bottleneck.

    Architecture:  X → encoder → z ∈ ℝ^{bottleneck_dim} → decoder → w ∈ Δ^{n_ref-1}

    By forcing all information through a low-dimensional bottleneck, the
    network is biased toward learning the sufficient dimension reduction
    f : ℝ^p → ℝ^{d₀} when the response truly depends on X only through
    B₀ᵀX ∈ ℝ^{d₀}.

    This connects to the reduction discussed in the Fréchet SDR paper:
    the bottleneck layer learns f = B₀ᵀX, and the decoder learns the
    map from the reduced space to the Fréchet mean.

    Args:
        input_dim:      predictor dimension p
        n_ref:          number of training reference SPD matrices
        bottleneck_dim: dimension d₀ of the bottleneck (e.g. 2)
        encoder_sizes:  hidden layer sizes *before* the bottleneck
        decoder_sizes:  hidden layer sizes *after* the bottleneck
        activation:     'relu', 'gelu', or 'tanh'
        dropout:        dropout probability
    """

    def __init__(
        self,
        input_dim: int = 10,
        n_ref: int = 200,
        bottleneck_dim: int = 2,
        encoder_sizes: Optional[List[int]] = None,
        decoder_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_ref = n_ref
        self.bottleneck_dim = bottleneck_dim

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        enc_hs = encoder_sizes or [128, 64]
        dec_hs = decoder_sizes or [64, 128]

        # --- Encoder: X → z ---
        enc_layers: list = []
        prev = input_dim
        for h in enc_hs:
            enc_layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        # no activation after bottleneck (linear projection)
        self.encoder = nn.Sequential(*enc_layers)

        # --- Decoder: z → w ---
        dec_layers: list = []
        prev = bottleneck_dim
        for h in dec_hs:
            dec_layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h
        self.decoder = nn.Sequential(*dec_layers)

        # Output head
        self.head = nn.Linear(prev, n_ref)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck representation z = encoder(x).

        Useful for inspecting or visualising the learned reduction.
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        Returns: [B, n_ref]  simplex weights
        """
        z = self.encoder(x)            # [B, bottleneck_dim]
        h = self.decoder(z)            # [B, hidden]
        logits = self.head(h)          # [B, n_ref]
        w = F.softmax(logits, dim=-1)  # [B, n_ref]
        return w


class ResponseLoRARefinement(nn.Module):
    """LoRA-style response refinement for response logits.

    Given initial logits ``H`` of shape ``[B, n_ref, V]``, applies

        M = I_V + (alpha / r) * (B @ A^T)
        W_logits = H @ M

    where ``A, B ∈ R^{V×r}`` are trainable response factors.
    """

    def __init__(
        self,
        n_responses: int,
        rank: int,
        alpha: float = 1.0,
        use_mask: bool = False,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be a positive integer")

        self.n_responses = n_responses
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        # LoRA-style init: start from identity map (Delta = 0).
        self.A = nn.Parameter(torch.randn(n_responses, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(n_responses, rank))

        self.use_mask = use_mask
        if use_mask:
            if mask is None:
                raise ValueError("mask must be provided when use_mask=True")
            if mask.shape != (n_responses, n_responses):
                raise ValueError(
                    f"mask must have shape {(n_responses, n_responses)}, got {tuple(mask.shape)}"
                )
            self.register_buffer("mask", mask.float())
        else:
            self.mask = None

    def mixing_matrix(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        I = torch.eye(self.n_responses, device=device, dtype=dtype)
        delta = self.B @ self.A.T
        M = I + self.scaling * delta
        if self.use_mask:
            M = M * self.mask.to(device=device, dtype=dtype)
        return M

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        M = self.mixing_matrix(device=H.device, dtype=H.dtype)
        return torch.einsum("bnv,vw->bnw", H, M)


class FrechetDRNN(nn.Module):
    """
    Fréchet Dimension Reduction Neural Network (FDRNN).

    Three-stage architecture for multi-task Fréchet SDR:

    Stage (i) — **Shared SDR reduction**  f : ℝᵖ → ℝᵈ
        Either *linear* (a single weight matrix, suited for estimating
        B₀ᵀX) or *nonlinear* (deep encoder for general f).
        Nuclear-norm regularisation on the first layer's weight matrix
        promotes low effective rank and guarantees sufficient reduction
        (Tang & Li, 2025).

    Stage (ii) — **V independent weight heads**  h_v : ℝᵈ → Δⁿ⁻¹
        Each head feeds f(x) through an MLP followed by Softmax to
        produce n-dimensional weights for the weighted Fréchet mean
        of response v.

    Stage (iii) — **Response LoRA refining** (active only when V > 1)
        The initial n × V logits matrix is right-multiplied by
        a LoRA-style response interaction matrix
            M = I + (alpha / r) B Aᵀ ∈ ℝ^{V×V},
        optionally masked by a fixed neighbourhood matrix.
        This captures inter-response borrowing via a low-rank update.

    When ``n_responses=1`` the ``forward()`` method returns a single
    ``[B, n_ref]`` weight tensor, matching the API of
    :class:`FrechetWeightNet` / :class:`FrechetBottleneckNet` so that the
    existing training and evaluation functions work unchanged.

    Args:
        input_dim:      predictor dimension p
        n_ref:          training-set size n (= number of reference SPD matrices)
        reduction_dim:  SDR dimension d (should match structural dim d₀)
        n_responses:    number of object-valued responses V
        reduction_type: ``'linear'`` or ``'nonlinear'``
        encoder_sizes:  hidden-layer sizes *before* the belt layer (nonlinear)
        head_sizes:     hidden-layer sizes in each weight head
        response_rank:  rank r of response LoRA correction (V > 1);
                ``None`` disables response refining
        response_alpha: LoRA scaling numerator alpha (effective scale alpha/r)
        response_mask:  optional fixed neighbourhood mask C ∈ {0,1}^{V×V}
        refine_rank:    deprecated alias of ``response_rank``
        activation:     ``'relu'``, ``'gelu'``, or ``'tanh'``
        dropout:        dropout probability
    """

    def __init__(
        self,
        input_dim: int = 10,
        n_ref: int = 200,
        reduction_dim: int = 2,
        n_responses: int = 1,
        reduction_type: str = "nonlinear",
        encoder_sizes: Optional[List[int]] = None,
        head_sizes: Optional[List[int]] = None,
        response_rank: Optional[int] = None,
        response_alpha: float = 1.0,
        response_mask: Optional[torch.Tensor] = None,
        refine_rank: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_ref = n_ref
        self.reduction_dim = reduction_dim
        self.n_responses = n_responses
        self.reduction_type = reduction_type
        self.response_rank = response_rank if response_rank is not None else refine_rank
        self.response_alpha = response_alpha
        self.response_mask = response_mask

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        # === Stage (i): Shared reduction network f ===
        if reduction_type == "linear":
            # f(x) = W x,  W ∈ R^{d×p}  (no bias, no activation)
            self.reduction = nn.Sequential(
                nn.Linear(input_dim, reduction_dim, bias=False)
            )
        else:
            enc_hs = encoder_sizes or [128, 64]
            layers: list = []
            prev = input_dim
            for h in enc_hs:
                layers += [nn.Linear(prev, h), act_cls()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            # Belt (bottleneck) — linear projection, no activation
            layers.append(nn.Linear(prev, reduction_dim))
            self.reduction = nn.Sequential(*layers)

        # === Stage (ii): V weight heads ===
        h_sizes = head_sizes or [64, 128]
        self.heads = nn.ModuleList()
        for _ in range(n_responses):
            head_layers: list = []
            prev = reduction_dim
            for h in h_sizes:
                head_layers += [nn.Linear(prev, h), act_cls()]
                if dropout > 0:
                    head_layers.append(nn.Dropout(dropout))
                prev = h
            head_layers.append(nn.Linear(prev, n_ref))
            self.heads.append(nn.Sequential(*head_layers))

        # === Stage (iii): Response LoRA refining (V > 1 only) ===
        self._use_refining = (self.response_rank is not None and n_responses > 1)
        if self._use_refining:
            self.response_refine = ResponseLoRARefinement(
                n_responses=n_responses,
                rank=self.response_rank,
                alpha=self.response_alpha,
                use_mask=(response_mask is not None),
                mask=response_mask,
            )
        else:
            self.response_refine = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --- public helpers ---------------------------------------------------

    def get_reduction(self, x: torch.Tensor) -> torch.Tensor:
        """Return the d-dimensional SDR representation f(x)."""
        return self.reduction(x)

    def nuclear_norm(self) -> torch.Tensor:
        """Nuclear norm ‖W₁‖_* of the first linear layer in the reduction.

        Used as a regulariser to promote low effective rank, ensuring
        the learned reduction is (approximately) sufficient.
        """
        for module in self.reduction.modules():
            if isinstance(module, nn.Linear):
                eps = 1e-6
                weight_reg = module.weight + eps * torch.eye(module.weight.shape[0], module.weight.shape[1], device=module.weight.device, dtype=module.weight.dtype)
                return torch.linalg.svdvals(weight_reg).sum()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    # --- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor  [B, p]

        Returns
        -------
        When ``n_responses == 1``:  Tensor [B, n_ref]  (logits)
        When ``n_responses >  1``:  Tensor [B, n_ref, V]  (logits)
        """
        z = self.reduction(x)  # [B, d]

        if self.n_responses == 1:
            logits = self.heads[0](z)              # [B, n_ref]
            return logits                          # logits, not softmaxed

        # --- V > 1: multi-response path ---
        all_logits = torch.stack(
            [self.heads[v](z) for v in range(self.n_responses)], dim=-1
        )  # [B, n_ref, V]

        if self._use_refining:
            all_logits = self.response_refine(all_logits)

        return all_logits  # logits, not softmaxed

    def get_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get normalized weights from input x.

        Parameters
        ----------
        x : Tensor  [B, p]

        Returns
        -------
        Tensor [B, n_ref, V]  (softmaxed weights)
        """
        logits = self.forward(x)  # [B, n_ref] or [B, n_ref, V]
        if self.n_responses == 1:
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, n_ref, 1]
        else:
            weights = torch.softmax(logits, dim=1)  # [B, n_ref, V]
        return weights


# ======================================================================
# Training  (Algorithm 2 — Adam with entropy-regularised Fréchet loss)
# ======================================================================

def train_frechet_model(
    model: nn.Module,
    Y_ref: torch.Tensor,
    train_loader: DataLoader,
    dist_name: str = "frobenius",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    entropy_reg: float = 0.0,
    nuclear_reg: float = 0.0,
    device: str = "cpu",
    verbose: bool = True,
    val_loader: Optional[DataLoader] = None,
    patience: int = 15,
) -> List[float]:
    """
    Train w_θ using the entropy-regularised Fréchet loss (Algorithm 2).

    Supports both single-response and multi-response settings.
    For multi-response, Y_ref should be [n_ref, V, p, p], and the loss sums over responses.

    Per-sample loss:
        l(θ; (X, Y)) = sum_v d²( m_{θ,v}(X), Y_v ) + λ_H sum_v H( w_{θ,v}(X) ) + λ_N ‖W₁‖_*

    where for each response v, m_{θ,v}(X) is the weighted Fréchet mean of Y_ref[:, v, :, :]
    using weights w_{θ,v}(X), and H is Shannon entropy.

    Args:
        model:       FrechetWeightNet  (outputs simplex weights)
        Y_ref:       reference responses — [n_ref, p, p] or [n_ref, V, p, p]
        train_loader: DataLoader yielding (X, Y) mini-batches
        dist_name:   distance metric name
        epochs:      number of training epochs
        lr:          learning rate  η
        weight_decay: L2 regularisation in Adam
        entropy_reg: λ_H  (entropy regularisation coefficient)
        nuclear_reg: λ_N  (nuclear-norm regularisation coefficient;
                     only applies when model has ``nuclear_norm()``)
        device:      'cpu' or 'cuda'
        verbose:     print progress
        val_loader:  optional validation DataLoader for early stopping
        patience:    epochs to wait for improvement before stopping

    Returns:
        history: list of per-epoch average losses
    """
    import copy as _copy

    model.to(device)
    Y_ref = Y_ref.to(device)

    # Precompute Y_ref^alpha once for power metric (avoids repeated eigh)
    if dist_name == "power":
        alpha = 0.5
        Y_ref._powered = _spd_pow(Y_ref, alpha)   # [n_ref, p, p]
    else:
        Y_ref._powered = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dist_fn = get_distance_fn(dist_name)
    _has_nuclear = nuclear_reg > 0 and hasattr(model, 'nuclear_norm')
    history = []

    # Early stopping state
    _use_es = val_loader is not None
    best_val = float("inf")
    wait = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)   # [B, input_dim]
            Y_batch = Y_batch.to(device)   # [B, p, p] or [B, V, p, p]
            B = X_batch.size(0)

            # --- Step 3-4 of Algorithm 2 ---
            logits = model(X_batch)  # [B, n_ref] or [B, n_ref, V]
            W = torch.softmax(logits, dim=1)  # normalize to weights

            if model.n_responses == 1:
                # Single-response
                if Y_ref.dim() == 4:
                    Y_ref_use = Y_ref[:, 0, :, :]  # [n_ref, p, p]
                else:
                    Y_ref_use = Y_ref
                m_w = differentiable_frechet_mean(W, Y_ref_use, dist_name)  # [B, p, p]
                d = dist_fn(m_w, Y_batch)  # [B]
                loss_dist = (d ** 2).mean()  # scalar
                entropy_term = 0.0
                if entropy_reg > 0:
                    H_w = entropy(W)  # [B]
                    entropy_term = entropy_reg * H_w.mean()
            else:
                # Multi-response: W [B, n_ref, V], Y_batch [B, V, p, p], Y_ref [n_ref, V, p, p]
                V = W.shape[2]
                loss_dist = 0.0
                entropy_term = 0.0
                for v in range(V):
                    w_v = W[:, :, v]  # [B, n_ref]
                    Y_ref_v = Y_ref[:, v, :, :]  # [n_ref, p, p]
                    Y_batch_v = Y_batch[:, v, :, :]  # [B, p, p]
                    m_w_v = differentiable_frechet_mean(w_v, Y_ref_v, dist_name)  # [B, p, p]
                    d_v = dist_fn(m_w_v, Y_batch_v)  # [B]
                    loss_dist += (d_v ** 2).mean()  # sum over responses
                    if entropy_reg > 0:
                        H_w_v = entropy(w_v)  # [B]
                        entropy_term += entropy_reg * H_w_v.mean()

            loss = loss_dist + entropy_term

            # Nuclear-norm regularisation  λ_N ‖W₁‖_*
            if _has_nuclear:
                loss = loss + nuclear_reg * model.nuclear_norm()

            # --- Steps 5-9 of Algorithm 2 (handled by Adam) ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss_dist.item() * B   # track distance loss only
            n_samples += B

        avg_loss = total_loss / max(1, n_samples)
        history.append(avg_loss)

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
            print(f"Epoch {ep:4d}/{epochs} | {dist_name} loss = {avg_loss:.6f}")

        # --- Early stopping check ---
        if _use_es:
            val_loss = _evaluate_val_loss(model, Y_ref, val_loader, dist_name, device)
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                wait = 0
                best_state = _copy.deepcopy(model.state_dict())
            else:
                wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stop at epoch {ep} "
                          f"(best val d²={best_val:.6f} at epoch {ep - patience})")
                break

    # Restore best model if early stopping was used
    if _use_es and best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def evaluate_frechet_model(
    model: nn.Module,
    Y_ref: torch.Tensor,
    test_loader: DataLoader,
    dist_name: str = "frobenius",
    device: str = "cpu",
    true_means: Optional[torch.Tensor] = None,
) -> dict:
    """
    Evaluate the trained Fréchet weight network.

    Supports single and multi-response settings.
    For multi-response, computes separate predictions for each response and averages distances.

    For each test sample (X, Y):
      - Compute weights W = w_θ(X)
      - For each response v, compute m_v = weighted Fréchet mean of Y_ref[:, v, :, :]
      - Measure d(m_v, Y_v) and average over v and samples

    Returns dict with avg_dist, avg_dist_sq, avg_dist_to_mean, etc.
    """
    model.eval()
    model.to(device)
    Y_ref = Y_ref.to(device)
    dist_fn = get_distance_fn(dist_name)

    all_dists = []
    all_dists_to_mean = []
    idx_offset = 0

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)  # [B, p, p] or [B, V, p, p]
        B = X_batch.size(0)

        W = model(X_batch)  # [B, n_ref] or [B, n_ref, V]
        W = torch.softmax(W, dim=1)  # normalize logits to weights

        if W.dim() == 2:
            # Single-response
            m_w = differentiable_frechet_mean(W, Y_ref, dist_name)  # [B, p, p]
            d = dist_fn(m_w, Y_batch)  # [B]
            all_dists.extend(d.cpu().tolist())
            if true_means is not None:
                tm = true_means[idx_offset: idx_offset + B].to(device)  # [B, p, p]
                d_mean = dist_fn(m_w, tm)
                all_dists_to_mean.extend(d_mean.cpu().tolist())
        else:
            # Multi-response
            V = W.shape[2]
            d_batch = []
            d_mean_batch = []
            for v in range(V):
                w_v = W[:, :, v]  # [B, n_ref]
                Y_ref_v = Y_ref[:, v, :, :]  # [n_ref, p, p]
                Y_batch_v = Y_batch[:, v, :, :]  # [B, p, p]
                m_w_v = differentiable_frechet_mean(w_v, Y_ref_v, dist_name)  # [B, p, p]
                d_v = dist_fn(m_w_v, Y_batch_v)  # [B]
                d_batch.append(d_v)
                if true_means is not None:
                    tm_v = true_means[idx_offset: idx_offset + B, v, :, :].to(device)  # [B, p, p]
                    d_mean_v = dist_fn(m_w_v, tm_v)
                    d_mean_batch.append(d_mean_v)
            # Average over responses for each sample
            d_avg = torch.stack(d_batch).mean(dim=0)  # [B]
            all_dists.extend(d_avg.cpu().tolist())
            if true_means is not None:
                d_mean_avg = torch.stack(d_mean_batch).mean(dim=0)  # [B]
                all_dists_to_mean.extend(d_mean_avg.cpu().tolist())

        idx_offset += B

    all_dists = torch.tensor(all_dists)
    results = {
        "avg_dist": all_dists.mean().item(),
        "avg_dist_sq": (all_dists ** 2).mean().item(),
    }
    if true_means is not None:
        all_dists_to_mean = torch.tensor(all_dists_to_mean)
        results["avg_dist_to_mean"] = all_dists_to_mean.mean().item()
        results["avg_dist_sq_to_mean"] = (all_dists_to_mean ** 2).mean().item()

    return results


# ======================================================================
# Hyperparameter tuning utilities
# ======================================================================

def train_val_split(
    dataset: Dataset,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into training and validation subsets.

    Args:
        dataset:  a ``torch.utils.data.Dataset``
        val_frac: fraction in (0, 1) held out for validation
        seed:     random seed for the split

    Returns:
        (train_subset, val_subset)
    """
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=gen).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _collect_Y_from_subset(subset: Subset, parent_Y: torch.Tensor) -> torch.Tensor:
    """Gather Y matrices for a Subset, preserving order."""
    return parent_Y[subset.indices].float()


def _evaluate_val_loss(
    model: nn.Module,
    Y_ref: torch.Tensor,
    val_loader: DataLoader,
    dist_name: str,
    device: str,
) -> float:
    """Average d²(pred, Y) on a validation set (no grad)."""
    model.eval()
    dist_fn = get_distance_fn(dist_name)
    total, count = 0.0, 0
    with torch.no_grad():
        for X_b, Y_b in val_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            W = model(X_b)
            W = torch.softmax(W, dim=1)  # normalize logits to weights
            if W.dim() == 2:
                # Single-response
                m = differentiable_frechet_mean(W, Y_ref.to(device), dist_name)
                d = dist_fn(m, Y_b)
                total += (d ** 2).sum().item()
            else:
                # Multi-response
                V = W.shape[2]
                for v in range(V):
                    w_v = W[:, :, v]
                    Y_ref_v = Y_ref[:, v, :, :]
                    Y_b_v = Y_b[:, v, :, :]
                    m_v = differentiable_frechet_mean(w_v, Y_ref_v.to(device), dist_name)
                    d_v = dist_fn(m_v, Y_b_v)
                    total += (d_v ** 2).sum().item()
            count += X_b.size(0)
    return total / max(1, count)


def grid_search_frechet(
    dataset: Dataset,
    parent_Y: torch.Tensor,
    model_class,
    dist_name: str = "frobenius",
    param_grid: Optional[Dict[str, list]] = None,
    fixed_model_kwargs: Optional[Dict[str, Any]] = None,
    fixed_train_kwargs: Optional[Dict[str, Any]] = None,
    val_frac: float = 0.2,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search for FSDRNN / FSDRNN-BN hyperparameters using a validation set.

    Splits ``dataset`` into train/val, iterates over all combinations in
    ``param_grid``, trains each configuration, and returns the one with
    lowest validation loss (avg d²).

    Recognised grid keys (with sensible defaults):

    +-------------------+-----------------------------------------------+
    | Key               | Description                                   |
    +===================+===============================================+
    | ``lr``            | learning rate                                 |
    | ``entropy_reg``   | entropy regularisation λ                      |
    | ``weight_decay``  | L2 weight decay in Adam                       |
    | ``epochs``        | number of training epochs                     |
    | ``dropout``       | dropout probability                           |
    | ``hidden_sizes``  | MLP backbone sizes (FSDRNN)                   |
    | ``bottleneck_dim``| bottleneck dimension (FSDRNN-BN)              |
    | ``encoder_sizes`` | encoder layer sizes (FSDRNN-BN)               |
    | ``decoder_sizes`` | decoder layer sizes (FSDRNN-BN)               |
    +-------------------+-----------------------------------------------+

    Args:
        dataset:            full training Dataset (will be split internally)
        parent_Y:           [n, r, r] the full ``dataset.Y`` tensor
        model_class:        ``FrechetWeightNet`` or ``FrechetBottleneckNet``
        dist_name:          SPD distance metric name
        param_grid:         ``{param_name: [val1, val2, ...], ...}``
                            If ``None``, a compact default grid is used.
        fixed_model_kwargs: model constructor kwargs that are NOT searched
        fixed_train_kwargs: ``train_frechet_model`` kwargs that are NOT searched
        val_frac:           validation fraction
        batch_size:         mini-batch size
        device:             ``'cpu'`` or ``'cuda'``
        seed:               seed for the train/val split
        verbose:            print progress

    Returns:
        dict with ``best_params``, ``best_val_loss``, and ``all_results``
        (list of ``(params, val_loss)`` sorted by loss)
    """
    if fixed_model_kwargs is None:
        fixed_model_kwargs = {}
    if fixed_train_kwargs is None:
        fixed_train_kwargs = {}

    # ---- Default grid ----
    if param_grid is None:
        if model_class is FrechetBottleneckNet:
            param_grid = {
                "lr": [5e-4, 1e-3, 5e-3],
                "entropy_reg": [0.0, 0.01, 0.05],
                "bottleneck_dim": [1, 2, 3],
            }
        elif model_class is FrechetDRNN:
            param_grid = {
                "lr": [5e-4, 1e-3, 5e-3],
                "entropy_reg": [0.0, 0.01, 0.05],
                "nuclear_reg": [0.0, 0.005, 0.01],
                "reduction_dim": [1, 2, 3],
                "response_rank": [None, 1, 2],  # for multi-response
                "response_alpha": [1.0],  # can add more if needed
            }
        else:
            param_grid = {
                "lr": [5e-4, 1e-3, 5e-3],
                "entropy_reg": [0.0, 0.01, 0.05],
            }

    # ---- Split data ----
    train_sub, val_sub = train_val_split(dataset, val_frac=val_frac, seed=seed)
    Y_train = _collect_Y_from_subset(train_sub, parent_Y)
    n_train = len(train_sub)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    # ---- Enumerate grid ----
    grid_keys = sorted(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    configs = list(itertools.product(*grid_values))

    if verbose:
        print(f"\n  Grid search: {len(configs)} configurations")
        for k in grid_keys:
            print(f"    {k}: {param_grid[k]}")
        print(f"    train/val sizes: {n_train} / {len(val_sub)}")

    # ---- Model-constructor vs training params ----
    MODEL_KEYS = {
        "hidden_sizes", "bottleneck_dim", "encoder_sizes",
        "decoder_sizes", "dropout", "activation",
        # FrechetDRNN keys
        "reduction_dim", "reduction_type", "head_sizes",
        "n_responses", "response_rank", "response_alpha", "refine_rank",
    }

    all_results = []

    for ci, combo in enumerate(configs):
        params = dict(zip(grid_keys, combo))

        # Separate into model kwargs and training kwargs
        m_kw = dict(fixed_model_kwargs)
        t_kw = dict(fixed_train_kwargs)

        for k, v in params.items():
            if k in MODEL_KEYS:
                m_kw[k] = v
            else:
                t_kw[k] = v

        # Ensure input_dim and n_ref are set
        m_kw.setdefault("input_dim", train_sub[0][0].shape[0])
        m_kw.setdefault("n_ref", n_train)

        # Build model
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_class(**m_kw)

        # Train
        epochs = t_kw.pop("epochs", fixed_train_kwargs.get("epochs", 100))
        lr = t_kw.pop("lr", 1e-3)
        ent_reg = t_kw.pop("entropy_reg", 0.0)
        wd = t_kw.pop("weight_decay", 1e-5)
        nuc_reg = t_kw.pop("nuclear_reg", 0.0)

        train_frechet_model(
            model=model,
            Y_ref=Y_train,
            train_loader=train_loader,
            dist_name=dist_name,
            epochs=epochs,
            lr=lr,
            weight_decay=wd,
            entropy_reg=ent_reg,
            nuclear_reg=nuc_reg,
            device=device,
            verbose=False,
            val_loader=val_loader,
            patience=15,
        )

        val_loss = _evaluate_val_loss(model, Y_train, val_loader, dist_name, device)
        all_results.append((params, val_loss))

        if verbose:
            tag = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    [{ci+1}/{len(configs)}] {tag}  →  val d²={val_loss:.6f}")

    # ---- Sort & return ----
    all_results.sort(key=lambda x: x[1])
    best_params, best_val_loss = all_results[0]

    if verbose:
        print(f"\n  Best params:    {best_params}")
        print(f"  Best val d²:    {best_val_loss:.6f}")

    return {
        "best_params": best_params,
        "best_val_loss": best_val_loss,
        "all_results": all_results,
    }


def grid_search_dfr(
    dataset: Dataset,
    parent_X: torch.Tensor,
    parent_Y: torch.Tensor,
    dist_name: str = "frobenius",
    param_grid: Optional[Dict[str, list]] = None,
    val_frac: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search for Deep Fréchet Regression hyperparameters.

    Recognised grid keys:

    +--------------------+---------------------------------------------+
    | Key                | Description                                 |
    +====================+=============================================+
    | ``manifold_dim``   | ISOMAP embedding dimension                  |
    | ``manifold_k``     | number of ISOMAP neighbours                 |
    | ``hidden``         | DNN hidden-layer width                      |
    | ``layer``          | DNN depth (number of hidden layers)         |
    | ``num_epochs``     | DNN training epochs                         |
    | ``lr``             | learning rate for the DNN                   |
    | ``bw_scale``       | bandwidth multiplier (applied to auto bw)   |
    +--------------------+---------------------------------------------+

    Args:
        dataset:   full training Dataset (will be split)
        parent_X:  [n, p] predictor matrix of the full dataset
        parent_Y:  [n, r, r] SPD matrix responses of the full dataset
        dist_name: SPD distance metric name
        param_grid: ``{key: [v1, v2, ...], ...}``; defaults provided
        val_frac:  fraction held-out for validation
        batch_size: mini-batch size for evaluation
        seed:      random seed for the split
        verbose:   print progress

    Returns:
        dict with ``best_params``, ``best_val_loss``, ``all_results``
    """
    if param_grid is None:
        param_grid = {
            "manifold_dim": [1, 2],
            "hidden": [16, 32],
            "layer": [3, 4],
            "lr": [5e-4, 1e-3],
        }

    train_sub, val_sub = train_val_split(dataset, val_frac=val_frac, seed=seed)
    X_train = parent_X[train_sub.indices].float()
    Y_train = parent_Y[train_sub.indices].float()
    X_val = parent_X[val_sub.indices].float()
    Y_val = parent_Y[val_sub.indices].float()

    grid_keys = sorted(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    configs = list(itertools.product(*grid_values))

    if verbose:
        print(f"\n  DFR Grid search: {len(configs)} configurations")
        for k in grid_keys:
            print(f"    {k}: {param_grid[k]}")
        print(f"    train/val sizes: {len(train_sub)} / {len(val_sub)}")

    dist_fn = get_distance_fn(dist_name)
    all_results = []

    for ci, combo in enumerate(configs):
        params = dict(zip(grid_keys, combo))
        bw_scale = params.pop("bw_scale", None)

        dfr = DeepFrechetRegression(
            dist_name=dist_name,
            manifold_method="isomap",
            manifold_dim=params.get("manifold_dim", 2),
            manifold_k=params.get("manifold_k", 10),
            hidden=params.get("hidden", 32),
            layer=params.get("layer", 4),
            num_epochs=params.get("num_epochs", 500),
            lr=params.get("lr", 5e-4),
            dropout=params.get("dropout", 0.0),
            seed=seed,
        )
        dfr.fit(X_train, Y_train, verbose=False)

        # Optionally rescale bandwidth
        if bw_scale is not None:
            dfr._bw = dfr._bw * bw_scale

        # Validate
        Y_pred = dfr.predict(X_val)
        d = dist_fn(Y_pred, Y_val)
        val_loss = (d ** 2).mean().item()

        # Restore bw_scale in the params for reporting
        if bw_scale is not None:
            params["bw_scale"] = bw_scale

        all_results.append((params, val_loss))

        if verbose:
            tag = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    [{ci+1}/{len(configs)}] {tag}  →  val d²={val_loss:.6f}")

    all_results.sort(key=lambda x: x[1])
    best_params, best_val_loss = all_results[0]

    if verbose:
        print(f"\n  Best DFR params: {best_params}")
        print(f"  Best val d²:     {best_val_loss:.6f}")

    return {
        "best_params": best_params,
        "best_val_loss": best_val_loss,
        "all_results": all_results,
    }
