import numpy as np
import librosa
import logging
from config import N_FFT, HOP_LENGTH

logger = logging.getLogger(__name__)

def compute_stft(X, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Compute STFT of each channel in X.
    X: (N_SOURCES, N_SAMPLES)
    Returns: (N_SOURCES, N_BINS, N_FRAMES)
    """
    n_sources, _ = X.shape
    stft_list = []
    for i in range(n_sources):
        # Compute STFT (returns complex matrix)
        stft = librosa.stft(X[i, :], n_fft=n_fft, hop_length=hop_length)
        stft_list.append(stft)
    
    X_stft = np.stack(stft_list)
    logger.debug(f"Computed STFT. Shape: {X_stft.shape}")
    return X_stft

def reorder_stft_for_ica(X_stft):
    """
    Reorder STFT from (N_SOURCES, N_BINS, N_FRAMES) 
    to (N_BINS, N_SOURCES, N_FRAMES) for bin-wise ICA.
    """
    X_reordered = X_stft.transpose(1, 0, 2)
    logger.debug(f"Reordered STFT for ICA. Shape: {X_reordered.shape}")
    return X_reordered

def reconstruct_stft_from_ica(Y_stft_reordered):
    """
    Reorder STFT from (N_BINS, N_SOURCES, N_FRAMES)
    back to (N_SOURCES, N_BINS, N_FRAMES).
    """
    return Y_stft_reordered.transpose(1, 0, 2)

if __name__ == "__main__":
    from config import setup_logging
    setup_logging(level=logging.DEBUG)
    # Test STFT
    n_sources = 5
    n_samples = 44100 # 1 second
    X = np.random.randn(n_sources, n_samples)
    
    X_stft = compute_stft(X)
    X_reordered = reorder_stft_for_ica(X_stft)
