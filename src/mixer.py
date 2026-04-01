import numpy as np
import os
import soundfile as sf
from config import SAMPLE_RATE, DATA_MIXED_DIR

def generate_mixing_matrix(n_sources=5, seed=None):
    """
    Generate a random N_SOURCES x N_SOURCES mixing matrix.
    Ensure it's well-conditioned (not near-singular).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random normal matrix
    A = np.random.randn(n_sources, n_sources)
    
    # Optional: ensure it's not singular by checking condition number
    # Or just use SVD-based generation
    u, s, vh = np.linalg.svd(A)
    # Force singular values to be in a reasonable range (e.g., 0.5 to 2.0)
    s = np.linspace(0.5, 2.0, n_sources)
    A = u @ np.diag(s) @ vh
    
    return A

def mix_stems(S, A=None, seed=None):
    """
    Mix stems S (N_SOURCES, N_SAMPLES) using matrix A (N_SOURCES, N_SOURCES).
    If A is None, generate a random matrix.
    Returns X (N_SOURCES, N_SAMPLES) and the mixing matrix A.
    """
    n_sources, _ = S.shape
    if A is None:
        A = generate_mixing_matrix(n_sources, seed=seed)
    
    X = A @ S
    return X, A

def save_mixes(X, data_dir=DATA_MIXED_DIR, sr=SAMPLE_RATE):
    """
    Save each mixed channel as a .wav file.
    """
    n_channels, _ = X.shape
    for i in range(n_channels):
        file_path = os.path.join(data_dir, f"mix_{i}.wav")
        # Normalize to avoid clipping
        audio = X[i, :] / np.max(np.abs(X[i, :]))
        sf.write(file_path, audio, sr)
    print(f"Saved {n_channels} mixes in {data_dir}.")

if __name__ == "__main__":
    # Example usage (test with dummy data)
    n_sources = 5
    n_samples = 44100 * 5 # 5 seconds
    S = np.random.randn(n_sources, n_samples) # Mock stems
    
    X, A = mix_stems(S, seed=42)
    print(f"Mixing Matrix A:\n{A}")
    print(f"Mix shape: {X.shape}")
    
    save_mixes(X)
