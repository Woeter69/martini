"""
Audio Mixing Engine.

This module provides functions for mixing isolated sources using 
a well-conditioned mixing matrix to simulate the cocktail party effect.
"""

import numpy as np
import os
import soundfile as sf
import logging
from config import SAMPLE_RATE, DATA_MIXED_DIR

logger = logging.getLogger(__name__)

def generate_mixing_matrix(n_sources=5, seed=None):
    """
    Generate a random N_SOURCES x N_SOURCES mixing matrix.
    
    The matrix is guaranteed to be well-conditioned by ensuring 
    the singular values are within a reasonable range.
    
    Parameters:
        n_sources (int): Number of independent components.
        seed (int): Optional seed for reproducible results.
        
    Returns:
        np.ndarray: Well-conditioned mixing matrix (A).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random normal matrix
    A = np.random.randn(n_sources, n_sources)
    
    # Ensure it's not singular by forcing singular values (SVD-based)
    u, s, vh = np.linalg.svd(A)
    # Singular values: 0.5 to 2.0
    s = np.linspace(0.5, 2.0, n_sources)
    A = u @ np.diag(s) @ vh
    
    return A

def mix_stems(S, A=None, seed=None):
    """
    Mix stems S using matrix A (X = AS).
    
    Parameters:
        S (np.ndarray): (N_SOURCES, N_SAMPLES) signal matrix.
        A (np.ndarray): Optional fixed mixing matrix.
        seed (int): Seed for random matrix generation.
        
    Returns:
        tuple: (X, A) where X is the mixture matrix and A is the mixing matrix used.
    """
    if S.ndim != 2:
        raise ValueError(f"S must be 2D (N_SOURCES, N_SAMPLES), got shape {S.shape}")
        
    n_sources, _ = S.shape
    if A is not None:
        if A.shape != (n_sources, n_sources):
             raise ValueError(f"A must be square ({n_sources}, {n_sources}), got shape {A.shape}")
    else:
        A = generate_mixing_matrix(n_sources, seed=seed)
    
    X = A @ S
    return X, A

def save_mixes(X, data_dir=DATA_MIXED_DIR, sr=SAMPLE_RATE):
    """
    Save each mixed channel as a separate WAV file in the data directory.
    
    Parameters:
        X (np.ndarray): Mixture matrix (N_MIXTURES, N_SAMPLES).
        data_dir (str): Destination directory.
        sr (int): Target sample rate.
    """
    n_channels, _ = X.shape
    for i in range(n_channels):
        file_path = os.path.join(data_dir, f"mix_{i}.wav")
        # Normalize to avoid clipping
        max_val = np.max(np.abs(X[i, :]))
        if max_val > 0:
            audio = X[i, :] / max_val
        else:
            audio = X[i, :]
        sf.write(file_path, audio, sr)
    logger.info(f"Saved {n_channels} mixes in {data_dir}.")

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    # Test mixing with dummy noise
    n_sources = 5
    n_samples = 44100 * 5
    S = np.random.randn(n_sources, n_samples)
    
    X, A = mix_stems(S, seed=42)
    logger.info(f"Mixing Matrix A:\n{A}")
    logger.info(f"Mix shape: {X.shape}")
    
    save_mixes(X)
