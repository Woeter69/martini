"""
Audio Postprocessing and Reconstruction Module.

This module provides functions for inverse STFT transformation, 
DC-offset removal, and output normalization (Peak/LUFS).
"""

import numpy as np
import librosa
import soundfile as sf
import os
import logging
from config import SAMPLE_RATE, HOP_LENGTH, OUTPUTS_STEMS_DIR, STEM_NAMES

logger = logging.getLogger(__name__)

def reconstruct_audio_from_stft(S_stft, hop_length=HOP_LENGTH):
    """
    Apply inverse STFT (iSTFT) to reconstruct time-domain waveforms.
    
    Parameters:
        S_stft (np.ndarray): (N_SOURCES, N_BINS, N_FRAMES) spectral signals.
        hop_length (int): The hop size used in the original STFT.
        
    Returns:
        np.ndarray: Reconstructed signals of shape (N_SOURCES, N_SAMPLES).
    """
    n_sources, _, _ = S_stft.shape
    audio_list = []
    for i in range(n_sources):
        # iSTFT converts complex spectral frames back to audio samples
        audio = librosa.istft(S_stft[i, :, :], hop_length=hop_length)
        audio_list.append(audio)
    
    S_audio = np.stack(audio_list)
    logger.debug(f"Reconstructed audio from STFT. Shape: {S_audio.shape}")
    return S_audio

def postprocess_audio(audio, sr=SAMPLE_RATE, remove_dc=True, normalize='peak'):
    """
    Apply DC-offset removal and amplitude normalization.
    
    Parameters:
        audio (np.ndarray): Input samples.
        sr (int): Sample rate.
        remove_dc (bool): Whether to remove DC bias (mean subtraction).
        normalize (str): 'peak', 'lufs', or 'none'.
        
    Returns:
        np.ndarray: Postprocessed audio.
    """
    # 1. Remove DC offset
    if remove_dc:
        logger.debug("Removing DC offset.")
        audio = audio - np.mean(audio)
    
    # 2. Normalization
    if normalize == 'peak':
        logger.debug("Applying peak normalization.")
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    elif normalize == 'lufs':
        # Naive implementation of LUFS-like (RMS) normalization.
        # This keeps consistent loudness across output stems.
        logger.debug("Applying LUFS-like (RMS) normalization.")
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 0.1 # Approximately -20dBFS target
        if rms > 0:
            audio = audio * (target_rms / rms)
        # Final clip prevention
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            
    return audio

def save_separated_stems(S, names=STEM_NAMES, output_dir=OUTPUTS_STEMS_DIR, sr=SAMPLE_RATE, normalize='peak'):
    """
    Apply postprocessing and save separated sources to WAV files.
    
    Parameters:
        S (np.ndarray): Separated signal matrix (N_SOURCES, N_SAMPLES).
        names (list): The names for each source channel.
        output_dir (str): Destination directory.
        sr (int): Target sample rate.
        normalize (str): Normalization strategy.
    """
    n_sources, _ = S.shape
    for i in range(n_sources):
        # Handle cases where we might have more/fewer sources than STEM_NAMES
        name = names[i] if i < len(names) else f"source_{i}"
        file_path = os.path.join(output_dir, f"separated_{name}.wav")
        
        # Postprocess and save
        audio = postprocess_audio(S[i, :], sr=sr, normalize=normalize)
        sf.write(file_path, audio, sr)
        
    logger.info(f"Saved {n_sources} separated stems in {output_dir}.")

if __name__ == "__main__":
    from config import setup_logging
    setup_logging(level=logging.DEBUG)
    # Test reconstruction with random spectral frames
    n_sources = 2
    n_bins = 1025
    n_frames = 100
    S_stft = np.random.randn(n_sources, n_bins, n_frames) + 1j * np.random.randn(n_sources, n_bins, n_frames)
    
    S_audio = reconstruct_audio_from_stft(S_stft)
