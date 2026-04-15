"""
Audio Data Ingestion Module.

This module provides functions for loading audio files (WAV, FLAC, MP3), 
handling multi-channel input, and managing stem matrices.
"""

import os
import librosa
import numpy as np
import logging
from config import SAMPLE_RATE, STEM_NAMES, DATA_RAW_DIR
from src.exceptions import StemLoadError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ['.wav', '.flac', '.mp3']

def load_audio(file_path, sr=SAMPLE_RATE, mono=True, duration=None):
    """
    Load an audio file and resample it to a target sample rate.
    
    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Target sample rate.
        mono (bool): Whether to mix channels to mono.
        duration (float): Number of seconds to load (None for all).
        
    Returns:
        np.ndarray: Audio samples as a 1D or 2D array.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=mono, duration=duration)
        return audio
    except Exception as e:
        raise StemLoadError(f"Failed to load audio from {file_path}: {e}")

def find_stem_file(data_dir, stem_name):
    """
    Search for a stem file in the directory using supported extensions.
    
    Parameters:
        data_dir (str): Directory to search in.
        stem_name (str): The name of the stem (e.g., 'vocals').
        
    Returns:
        str or None: Full path to the file if found, else None.
    """
    for ext in SUPPORTED_EXTENSIONS:
        file_path = os.path.join(data_dir, f"{stem_name}{ext}")
        if os.path.exists(file_path):
            return file_path
    return None

def process_channels(audio, strategy='mono'):
    """
    Process audio channels according to the chosen strategy.
    
    Parameters:
        audio (np.ndarray): Input audio (1D or 2D).
        strategy (str): 'mono' (downmix) or 'stereo' (upmix/trim).
        
    Returns:
        np.ndarray: Processed audio samples.
    """
    if audio.ndim == 1: # Mono
        if strategy == 'stereo':
            logger.debug("Upmixing mono to stereo.")
            return np.stack([audio, audio])
        return audio # Already mono
    
    elif audio.ndim == 2: # Multi-channel
        n_channels = audio.shape[0]
        if strategy == 'mono':
            logger.debug(f"Mixing {n_channels} channels down to mono.")
            return np.mean(audio, axis=0)
        elif strategy == 'stereo':
            if n_channels == 2:
                return audio
            logger.debug(f"Taking first 2 channels from {n_channels} channels.")
            return audio[:2, :]
        return audio

def load_stems(data_dir=DATA_RAW_DIR, sr=SAMPLE_RATE, channel_strategy='mono', duration=None):
    """
    Load all stem files defined in STEM_NAMES from the data directory.
    
    Parameters:
        data_dir (str): Root path for the stem files.
        sr (int): Sample rate to use.
        channel_strategy (str): 'mono' or 'stereo' handling.
        duration (float): Duration in seconds to load.
        
    Returns:
        dict: Dictionary mapping stem names to audio arrays.
    """
    stems = {}
    logger.info(f"Loading stems from {data_dir} with strategy '{channel_strategy}'...")
    if not os.path.isdir(data_dir):
        raise StemLoadError(f"Raw data directory does not exist: {data_dir}")
    
    # We load with mono=False to get raw channels if needed
    mono_load = (channel_strategy == 'mono')
    
    for stem in STEM_NAMES:
        file_path = find_stem_file(data_dir, stem)
        if not file_path:
            raise StemLoadError(f"Missing stem: {stem} in {data_dir}. "
                              f"Supported formats: {SUPPORTED_EXTENSIONS}")
        
        audio = load_audio(file_path, sr=sr, mono=mono_load, duration=duration)
        stems[stem] = process_channels(audio, strategy=channel_strategy)
    
    if not stems:
        raise StemLoadError("No stems were loaded.")

    # Ensure all stems have the same length (clip to the shortest)
    lengths = [len(s.T) if s.ndim > 1 else len(s) for s in stems.values()]
    if not lengths:
        raise StemLoadError("Loaded stems are empty.")
        
    min_len = min(lengths)
    for stem in stems:
        if stems[stem].ndim > 1:
            stems[stem] = stems[stem][:, :min_len]
        else:
            stems[stem] = stems[stem][:min_len]
        
    logger.info(f"Successfully loaded {len(stems)} stems.")
    return stems

def get_stem_matrix(stems):
    """
    Convert dictionary of stems into a stacked signal matrix.
    
    Parameters:
        stems (dict): Dictionary mapping name to array.
        
    Returns:
        np.ndarray: Signal matrix of shape (N_SOURCES, N_SAMPLES).
    """
    # Order matters; let's use the order in STEM_NAMES
    return np.stack([stems[name] for name in STEM_NAMES])

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    # Test loading
    try:
        stems = load_stems(duration=5.0) # Load 5 seconds for testing
        X = get_stem_matrix(stems)
        logger.info(f"Loaded {X.shape[0]} stems with {X.shape[1]} samples each.")
    except Exception as e:
        logger.error(f"Error: {e}")
