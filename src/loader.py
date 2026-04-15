import os
import librosa
import numpy as np
import logging
from config import SAMPLE_RATE, STEM_NAMES, DATA_RAW_DIR
from src.exceptions import StemLoadError

logger = logging.getLogger(__name__)

def load_audio(file_path, sr=SAMPLE_RATE, duration=None):
    """
    Load an audio file and resample to SR.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
        return audio
    except Exception as e:
        raise StemLoadError(f"Failed to load audio from {file_path}: {e}")

def load_stems(data_dir=DATA_RAW_DIR, sr=SAMPLE_RATE, duration=None):
    """
    Load all 5 stems (vocals, drums, bass, guitar, piano) from the raw data directory.
    Returns a dictionary of {stem_name: audio_data}.
    """
    stems = {}
    logger.info(f"Loading stems from {data_dir}...")
    if not os.path.isdir(data_dir):
        raise StemLoadError(f"Raw data directory does not exist: {data_dir}")
        
    for stem in STEM_NAMES:
        # Assuming stems are named like vocals.wav, drums.wav, etc.
        file_path = os.path.join(data_dir, f"{stem}.wav")
        if not os.path.exists(file_path):
            raise StemLoadError(f"Missing stem: {file_path}. Please download/place your stems in {data_dir}.")
        
        stems[stem] = load_audio(file_path, sr=sr, duration=duration)
    
    if not stems:
        raise StemLoadError("No stems were loaded.")

    # Ensure all stems have the same length (clip to the shortest)
    lengths = [len(s) for s in stems.values()]
    if not lengths:
        raise StemLoadError("Loaded stems are empty.")
        
    min_len = min(lengths)
    for stem in stems:
        stems[stem] = stems[stem][:min_len]
        
    logger.info(f"Successfully loaded {len(stems)} stems.")
    return stems

def get_stem_matrix(stems):
    """
    Convert dictionary of stems into a 2D numpy array (N_SOURCES, N_SAMPLES).
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
