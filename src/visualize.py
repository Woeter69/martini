import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import librosa.display
from config import OUTPUTS_PLOTS_DIR, STEM_NAMES, SAMPLE_RATE

def plot_waveforms(signals, title, names=None, save_path=None, sr=SAMPLE_RATE):
    """
    Plot multiple waveforms in subplots with time axis.
    signals: (N_SOURCES, N_SAMPLES)
    """
    n_sources, n_samples = signals.shape
    time = np.linspace(0, n_samples / sr, n_samples)
    
    fig, axes = plt.subplots(n_sources, 1, figsize=(12, 2 * n_sources), sharex=True)
    if n_sources == 1:
        axes = [axes]
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        axes[i].plot(time, signals[i, :], linewidth=0.5)
        axes[i].set_ylabel("Amp")
        axes[i].set_title(name, fontsize=10, loc='left')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_spectrograms(stfts, title, names=None, save_path=None, sr=SAMPLE_RATE, hop_length=512):
    """
    Plot multiple spectrograms (magnitudes) in subplots with labeled axes.
    stfts: (N_SOURCES, N_BINS, N_FRAMES)
    """
    n_sources, _, _ = stfts.shape
    fig, axes = plt.subplots(n_sources, 1, figsize=(12, 3 * n_sources))
    if n_sources == 1:
        axes = [axes]
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        S_db = librosa.amplitude_to_db(np.abs(stfts[i, :, :]), ref=np.max)
        img = librosa.display.specshow(S_db, ax=axes[i], y_axis='log', x_axis='time', 
                                      sr=sr, hop_length=hop_length, cmap='magma')
        axes[i].set_title(f"Spectrogram: {name}")
        fig.colorbar(img, ax=axes[i], format="%+2.0f dB")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_comparison(original, estimated, title, names=None, save_path=None, sr=SAMPLE_RATE):
    """
    Side-by-side comparison of original (true) and estimated signals.
    original, estimated: (N_SOURCES, N_SAMPLES)
    """
    n_sources, n_samples = original.shape
    time = np.linspace(0, n_samples / sr, n_samples)
    
    fig, axes = plt.subplots(n_sources, 2, figsize=(15, 2 * n_sources), sharex=True)
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        
        # Original
        axes[i, 0].plot(time, original[i, :], color='blue', linewidth=0.5)
        axes[i, 0].set_title(f"True {name}")
        axes[i, 0].grid(True, alpha=0.3)
        
        # Estimated
        axes[i, 1].plot(time, estimated[i, :], color='green', linewidth=0.5)
        axes[i, 1].set_title(f"Estimated {name}")
        axes[i, 1].grid(True, alpha=0.3)
        
    plt.suptitle(title, fontsize=14)
    plt.xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_matrix(matrix, title, save_path=None):
    """
    Plot a heatmap of a matrix (mixing or unmixing).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                square=True, cbar_kws={"shrink": .8})
    plt.title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Test visualizations
    n_sources = 2
    n_samples = 44100
    signals = np.random.randn(n_sources, n_samples)
    plot_waveforms(signals, "Test Waveforms", names=['Vocals', 'Drums'])
    
    matrix = np.random.randn(n_sources, n_sources)
    plot_matrix(matrix, "Test Mixing Matrix")
