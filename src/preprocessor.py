"""
Audio Preprocessing and Spectral Transformation.

This module provides functions for short-time Fourier transform (STFT), 
reordering spectral bins for ICA, and spectral reconstruction.
"""

import numpy as np
import librosa
import logging
from config import N_FFT, HOP_LENGTH

logger = logging.getLogger(__name__)

def compute_stft(X, n_fft=N_FFT, hop_length=None, overlap_ratio=0.75, window='hann'):
    """
    Compute STFT of each channel in a signal matrix.
    
    The output is complex-valued, containing both magnitude and phase.
    
    Parameters:
        X (np.ndarray): Mixture matrix of shape (N_SOURCES, N_SAMPLES).
        n_fft (int): STFT window size.
        hop_length (int): Number of samples between successive frames.
        overlap_ratio (float): If hop_length is None, it is calculated from this.
        window (str): The window function to apply.
        
    Returns:
        tuple: (X_stft, hop_length) 
            - X_stft: Complex-valued STFT of shape (N_SOURCES, N_BINS, N_FRAMES).
            - hop_length: The calculated hop size used.
    """
    if hop_length is None:
        # Calculate hop_length: n_fft * (1 - overlap_ratio)
        hop_length = int(n_fft * (1.0 - overlap_ratio))
    
    n_sources, _ = X.shape
    stft_list = []
    logger.debug(f"Computing STFT with n_fft={n_fft}, hop_length={hop_length}, window={window}.")
    for i in range(n_sources):
        # STFT returns a complex spectral matrix
        stft = librosa.stft(X[i, :], n_fft=n_fft, hop_length=hop_length, window=window)
        stft_list.append(stft)
    
    X_stft = np.stack(stft_list)
    logger.debug(f"Computed STFT. Shape: {X_stft.shape}")
    return X_stft, hop_length

def reorder_stft_for_ica(X_stft):
    """
    Reorder STFT from (SOURCES, BINS, FRAMES) to (BINS, SOURCES, FRAMES).
    
    This enables bin-wise (independent) frequency domain ICA processing.
    
    Parameters:
        X_stft (np.ndarray): (N_SOURCES, N_BINS, N_FRAMES) matrix.
        
    Returns:
        np.ndarray: Reordered matrix for bin-wise processing.
    """
    X_reordered = X_stft.transpose(1, 0, 2)
    logger.debug(f"Reordered STFT for ICA. Shape: {X_reordered.shape}")
    return X_reordered

def reconstruct_stft_from_ica(Y_stft_reordered):
    """
    Reorder STFT from (BINS, SOURCES, FRAMES) back to (SOURCES, BINS, FRAMES).
    
    Inverse of `reorder_stft_for_ica`.
    
    Parameters:
        Y_stft_reordered (np.ndarray): Reordered spectral matrix.
        
    Returns:
        np.ndarray: Matrix of shape (N_SOURCES, N_BINS, N_FRAMES).
    """
    return Y_stft_reordered.transpose(1, 0, 2)

if __name__ == "__main__":
    from config import setup_logging
    setup_logging(level=logging.DEBUG)
    # Test STFT transformation
    n_sources = 2
    n_samples = 44100
    X = np.random.randn(n_sources, n_samples)
    
    X_stft, _ = compute_stft(X)
    X_reordered = reorder_stft_for_ica(X_stft)
