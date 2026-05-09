"""
Audio Data Ingestion Module.
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
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=mono, duration=duration)
        return audio
    except Exception as e:
        raise StemLoadError(f"Failed to load audio from {file_path}: {e}")

def find_stem_file(data_dir, stem_name):
    for ext in SUPPORTED_EXTENSIONS:
        file_path = os.path.join(data_dir, f"{stem_name}{ext}")
        if os.path.exists(file_path): return file_path
    return None

def process_channels(audio, strategy='mono'):
    if audio.ndim == 1:
        if strategy == 'stereo': return np.stack([audio, audio])
        return audio
    elif audio.ndim == 2:
        if strategy == 'mono': return np.mean(audio, axis=0)
        return audio[:2, :]
    return audio

def load_stems(data_dir=DATA_RAW_DIR, sr=SAMPLE_RATE, channel_strategy='mono', duration=None):
    stems = {}
    if not os.path.isdir(data_dir): raise StemLoadError(f"Raw data directory does not exist: {data_dir}")
    for stem in STEM_NAMES:
        file_path = find_stem_file(data_dir, stem)
        if not file_path: raise StemLoadError(f"Missing stem: {stem}")
        audio = load_audio(file_path, sr=sr, mono=(channel_strategy == 'mono'), duration=duration)
        stems[stem] = process_channels(audio, strategy=channel_strategy)
    min_len = min(len(s.T) if s.ndim > 1 else len(s) for s in stems.values())
    for stem in stems:
        if stems[stem].ndim > 1: stems[stem] = stems[stem][:, :min_len]
        else: stems[stem] = stems[stem][:min_len]
    return stems

def get_stem_matrix(stems):
    return np.stack([stems[name] for name in STEM_NAMES])

def load_stereo_as_mixtures(file_path, sr=SAMPLE_RATE, duration=None):
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=False, duration=duration)
        if audio.ndim == 1:
            logger.warning("File is mono. Creating pseudo-stereo.")
            return np.stack([audio, audio + 0.001 * np.random.randn(len(audio))])
        return audio[:2, :]
    except Exception as e:
        raise StemLoadError(f"Failed to load stereo mixtures: {e}")
