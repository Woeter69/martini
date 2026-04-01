import numpy as np
import librosa
from config import N_FFT, HOP_LENGTH

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
    
    return np.stack(stft_list)

def reorder_stft_for_ica(X_stft):
    """
    Reorder STFT from (N_SOURCES, N_BINS, N_FRAMES) 
    to (N_BINS, N_SOURCES, N_FRAMES) for bin-wise ICA.
    """
    return X_stft.transpose(1, 0, 2)

def reconstruct_stft_from_ica(Y_stft_reordered):
    """
    Reorder STFT from (N_BINS, N_SOURCES, N_FRAMES)
    back to (N_SOURCES, N_BINS, N_FRAMES).
    """
    return Y_stft_reordered.transpose(1, 0, 2)

if __name__ == "__main__":
    # Test STFT
    n_sources = 5
    n_samples = 44100 # 1 second
    X = np.random.randn(n_sources, n_samples)
    
    X_stft = compute_stft(X)
    print(f"STFT shape: {X_stft.shape}") # (5, 1025, 87) approximately
    
    X_reordered = reorder_stft_for_ica(X_stft)
    print(f"Reordered STFT shape: {X_reordered.shape}") # (1025, 5, 87)
