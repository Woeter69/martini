"""
FastICA Engine Implementation.

This module provides functions for independent component analysis using the 
fixed-point algorithm (FastICA). It supports both time-domain and 
frequency-domain (bin-wise) source separation.
"""

import numpy as np
import logging
from src.exceptions import ConvergenceError, UnderdeterminedError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contrast functions for FastICA
# ---------------------------------------------------------------------------

def g_tanh(x):
    """
    Contrast function: tanh(u). Good general-purpose choice.
    This corresponds to an independence measure based on log-cosh(u).
    """
    return np.tanh(x)

def g_prime_tanh(x):
    """
    Second derivative of log-cosh(u): 1 - tanh^2(u).
    Used in the Newton-Raphson update step.
    """
    return 1 - np.square(np.tanh(x))

def g_kurtosis(x):
    """
    Contrast function: u^3 (kurtosis-based).
    Good for separating sub-Gaussian sources (signals with flatter distributions).
    """
    return x ** 3

def g_prime_kurtosis(x):
    """
    Derivative of u^3: 3u^2.
    """
    return 3 * np.square(x)

# Registry mapping names to (g, g') pairs for CLI selection
CONTRAST_FUNCTIONS = {
    'tanh': (g_tanh, g_prime_tanh),
    'kurtosis': (g_kurtosis, g_prime_kurtosis),
}

# Backward-compatible aliases
g = g_tanh
g_prime = g_prime_tanh

def center(X):
    """
    Center the data by subtracting the mean of each source.
    
    Parameters:
        X (np.ndarray): (N_SOURCES, N_SAMPLES) signal matrix.
        
    Returns:
        np.ndarray: Zero-mean signal matrix.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean

def whiten(X):
    """
    Whiten the data using Principal Component Analysis (PCA).
    
    Whitening ensures the data is decorrelated (covariance = Identity) and 
    normalized to unit variance. This reduces the search space for ICA 
    to orthogonal matrices.
    
    Returns: 
        tuple: (X_whitened, whitening_matrix, dewhitening_matrix)
    """
    # Covariance matrix: E[XX^T]
    cov = np.cov(X)
    
    # Eigendecomposition: cov = E * D * E.T
    d, E = np.linalg.eigh(cov)
    
    # Whitening matrix: W_white = D^(-1/2) * E.T
    # This transforms X such that E[(W_white X)(W_white X)^T] = I
    # Add small epsilon to avoid division by zero
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    W_white = D_inv_sqrt @ E.T
    
    # Dewhitening matrix: W_dewhite = E * D^(1/2)
    # Inverse of whitening transformation
    W_dewhite = E @ np.diag(np.sqrt(d + 1e-10))
    
    return W_white @ X, W_white, W_dewhite

def fast_ica(X, n_sources=None, n_iter=200, tol=1e-5, contrast='tanh'):
    """
    FastICA algorithm using a deflationary approach.
    
    Extracts independent components one by one by maximizing non-gaussianity
    (as a proxy for statistical independence).
    
    Parameters:
        X (np.ndarray): (N_MIXTURES, N_SAMPLES) — input mixed signals.
        n_sources (int): number of independent components to extract.
        n_iter (int): maximum iterations per component.
        tol (float): convergence tolerance.
        contrast (str): 'tanh' or 'kurtosis' non-linearity.

    Returns:
        tuple: (S, W, convergence_info)
            - S: Estimated source signals (N_SOURCES, N_SAMPLES).
            - W: The learned unmixing matrix.
            - convergence_info: Metadata for each component's convergence.
    """
    if np.any(np.isnan(X)):
        raise ValueError("Input data contains NaNs.")

    if contrast not in CONTRAST_FUNCTIONS:
        raise ValueError(
            f"Unknown contrast '{contrast}'. "
            f"Available: {list(CONTRAST_FUNCTIONS.keys())}"
        )
    g_func, g_prime_func = CONTRAST_FUNCTIONS[contrast]

    n_mixtures = X.shape[0]
    if n_sources is None:
        n_sources = n_mixtures
    
    if n_mixtures < n_sources:
        raise UnderdeterminedError(f"Underdetermined problem: {n_mixtures} mixtures < {n_sources} sources.")

    logger.debug(f"Starting FastICA: {n_mixtures} mixtures, {n_sources} sources, {n_iter} max iterations, contrast={contrast}.")
        
    # 1. Center the data
    X_centered = center(X)
    
    # 2. Whiten the data to simplify unmixing to a rotation
    X_whitened, W_white, W_dewhite = whiten(X_centered)
    
    # 3. Iterate to find independent components (Deflation)
    # We find one vector w_i at a time such that w_i^T X_whitened is maximally non-gaussian.
    W_ica = np.zeros((n_sources, n_mixtures))
    convergence_info = []
    
    for i in range(n_sources):
        # Initialize random weight vector for the current component
        w = np.random.randn(n_mixtures)
        w /= np.linalg.norm(w)
        converged = False
        iterations = 0
        
        for j in range(n_iter):
            iterations = j + 1
            wtx = w @ X_whitened
            
            # FastICA Fixed-point Update Rule:
            # w_new = E[x * g(w^T x)] - E[g'(w^T x)] * w
            # This is derived from maximizing the negentropy J(w^T x).
            term1 = (X_whitened * g_func(wtx)).mean(axis=1)
            term2 = g_prime_func(wtx).mean() * w
            w_new = term1 - term2
            
            # Orthogonalize against previously found components (Gram-Schmidt)
            # This ensures we don't find the same component multiple times.
            if i > 0:
                w_new -= (w_new @ W_ica[:i].T) @ W_ica[:i]
                
            w_new /= np.linalg.norm(w_new)
            
            # Check convergence by comparing alignment of new and old vectors
            if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol:
                logger.debug(f"Component {i} converged in {j+1} iterations.")
                converged = True
                w = w_new
                break
            w = w_new
            
        if not converged:
            logger.warning(f"Component {i} did not converge within {n_iter} iterations.")
            # We raise ConvergenceError to satisfy project requirements.
            raise ConvergenceError(f"Component {i} failed to converge after {n_iter} iterations.")
            
        W_ica[i, :] = w
        convergence_info.append({
            'component': i,
            'converged': converged,
            'iterations': iterations,
        })
        
    # The total unmixing matrix W = W_ica (learned rotation) @ W_white (PCA whitening)
    W = W_ica @ W_white
    # Reconstruct signals: S = W X
    S = W @ X
    
    return S, W, convergence_info

def solve_permutation(Y_stft):
    """
    Solve the frequency-domain permutation ambiguity across bins.

    ICA in the frequency domain is performed on each bin independently. 
    However, the order of sources is random for each bin. This results in 
    scrambled frequencies across separated tracks.
    
    This function aligns source indices by maximizing the correlation 
    of signal envelopes between adjacent frequency bins.

    Parameters:
        Y_stft (np.ndarray): (N_BINS, N_SOURCES, N_FRAMES) — separated complex STFT.
        
    Returns: 
        np.ndarray: Permutation-aligned STFT matrix.
    """
    from scipy.optimize import linear_sum_assignment

    n_bins, n_sources, n_frames = Y_stft.shape
    Y_aligned = Y_stft.copy()

    for b in range(1, n_bins):
        # We assume frequency bins next to each other have correlated envelopes.
        # We correlate current bin against the previous (already aligned) bin.
        ref_env = np.abs(Y_aligned[b - 1])  # (n_sources, n_frames) reference
        cur_env = np.abs(Y_aligned[b])       # (n_sources, n_frames) candidate

        # Build cost matrix for the Hungarian Algorithm
        # cost[i, j] is the negative correlation between reference i and candidate j.
        cost = np.zeros((n_sources, n_sources))
        for i in range(n_sources):
            for j in range(n_sources):
                r_norm = np.linalg.norm(ref_env[i])
                c_norm = np.linalg.norm(cur_env[j])
                if r_norm > 1e-12 and c_norm > 1e-12:
                    cost[i, j] = -np.dot(ref_env[i], cur_env[j]) / (r_norm * c_norm)
                else:
                    cost[i, j] = 0.0

        # Hungarian algorithm finds the global minimum cost assignment.
        _, col_ind = linear_sum_assignment(cost)

        # Reorder the current bin's sources to match the reference bin.
        Y_aligned[b] = Y_aligned[b, col_ind, :]

    return Y_aligned


if __name__ == "__main__":
    from config import setup_logging
    setup_logging(level=logging.DEBUG)
    
    # Test ICA with simple signals
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 1, 1000)
    s1 = np.sin(2 * np.pi * 5 * t)
    s2 = np.sign(np.sin(2 * np.pi * 3 * t))
    S_true = np.stack([s1, s2])
    
    # Mix signals: X = A S
    A = np.array([[0.5, 0.5], [0.3, 0.7]])
    X = A @ S_true
    
    # Recover: S_rec = W X
    try:
        S_rec, W_rec, conv_info = fast_ica(X)
        logger.info(f"Original mixing matrix A:\n{A}")
        logger.info(f"Recovered W inverse (estimated A):\n{np.linalg.inv(W_rec)}")
        for ci in conv_info:
            status = "converged" if ci['converged'] else "NOT converged"
            logger.info(f"  Component {ci['component']}: {status} in {ci['iterations']} iterations")
    except Exception as e:
        logger.error(f"ICA failed: {e}")
