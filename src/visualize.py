import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import OUTPUTS_PLOTS_DIR, STEM_NAMES

def plot_waveforms(signals, title, names=None, save_path=None):
    """
    Plot multiple waveforms in subplots.
    signals: (N_SOURCES, N_SAMPLES)
    """
    n_sources, _ = signals.shape
    fig, axes = plt.subplots(n_sources, 1, figsize=(10, 2 * n_sources), sharex=True)
    if n_sources == 1:
        axes = [axes]
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        axes[i].plot(signals[i, :])
        axes[i].set_ylabel(name)
        axes[i].set_title(name if i == 0 else "")
    
    plt.suptitle(title)
    plt.xlabel("Samples")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_spectrograms(stfts, title, names=None, save_path=None):
    """
    Plot multiple spectrograms (magnitudes) in subplots.
    stfts: (N_SOURCES, N_BINS, N_FRAMES)
    """
    import librosa.display
    n_sources, _, _ = stfts.shape
    fig, axes = plt.subplots(n_sources, 1, figsize=(10, 2 * n_sources), sharex=True)
    if n_sources == 1:
        axes = [axes]
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        S_db = librosa.amplitude_to_db(np.abs(stfts[i, :, :]), ref=np.max)
        librosa.display.specshow(S_db, ax=axes[i], y_axis='log', x_axis='time')
        axes[i].set_ylabel(name)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_matrix(matrix, title, save_path=None):
    """
    Plot a heatmap of a matrix (mixing or unmixing).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # Test visualizations
    n_sources = 5
    n_samples = 1000
    signals = np.random.randn(n_sources, n_samples)
    plot_waveforms(signals, "Test Waveforms", names=STEM_NAMES)
    
    matrix = np.random.randn(n_sources, n_sources)
    plot_matrix(matrix, "Test Mixing Matrix")
