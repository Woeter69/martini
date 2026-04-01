import os
import librosa
import numpy as np
from config import SAMPLE_RATE, STEM_NAMES, DATA_RAW_DIR

def load_audio(file_path, sr=SAMPLE_RATE, duration=None):
    """
    Load an audio file and resample to SR.
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    return audio

def load_stems(data_dir=DATA_RAW_DIR, sr=SAMPLE_RATE, duration=None):
    """
    Load all 5 stems (vocals, drums, bass, guitar, piano) from the raw data directory.
    Returns a dictionary of {stem_name: audio_data}.
    """
    stems = {}
    for stem in STEM_NAMES:
        # Assuming stems are named like vocals.wav, drums.wav, etc.
        file_path = os.path.join(data_dir, f"{stem}.wav")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing stem: {file_path}. Please download/place your stems in {data_dir}.")
        
        stems[stem] = load_audio(file_path, sr=sr, duration=duration)
    
    # Ensure all stems have the same length (clip to the shortest)
    min_len = min(len(s) for s in stems.values())
    for stem in stems:
        stems[stem] = stems[stem][:min_len]
        
    return stems

def get_stem_matrix(stems):
    """
    Convert dictionary of stems into a 2D numpy array (N_SOURCES, N_SAMPLES).
    """
    # Order matters; let's use the order in STEM_NAMES
    return np.stack([stems[name] for name in STEM_NAMES])

if __name__ == "__main__":
    # Test loading
    try:
        stems = load_stems(duration=5.0) # Load 5 seconds for testing
        X = get_stem_matrix(stems)
        print(f"Loaded {X.shape[0]} stems with {X.shape[1]} samples each.")
    except Exception as e:
        print(f"Error: {e}")
