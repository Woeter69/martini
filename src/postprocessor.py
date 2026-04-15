import numpy as np
import librosa
import soundfile as sf
import os
import logging
from config import SAMPLE_RATE, HOP_LENGTH, OUTPUTS_STEMS_DIR, STEM_NAMES

logger = logging.getLogger(__name__)

def reconstruct_audio_from_stft(S_stft, hop_length=HOP_LENGTH):
    """
    Inverse STFT for each source.
    S_stft: (N_SOURCES, N_BINS, N_FRAMES)
    Returns: (N_SOURCES, N_SAMPLES)
    """
    n_sources, _, _ = S_stft.shape
    audio_list = []
    for i in range(n_sources):
        audio = librosa.istft(S_stft[i, :, :], hop_length=hop_length)
        audio_list.append(audio)
    
    S_audio = np.stack(audio_list)
    logger.debug(f"Reconstructed audio from STFT. Shape: {S_audio.shape}")
    return S_audio

def save_separated_stems(S, names=STEM_NAMES, output_dir=OUTPUTS_STEMS_DIR, sr=SAMPLE_RATE):
    """
    Save separated stems as .wav files.
    S: (N_SOURCES, N_SAMPLES)
    """
    n_sources, _ = S.shape
    for i in range(n_sources):
        # Handle cases where we might have more/fewer sources than STEM_NAMES
        name = names[i] if i < len(names) else f"source_{i}"
        file_path = os.path.join(output_dir, f"separated_{name}.wav")
        
        # Normalize
        audio = S[i, :]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        sf.write(file_path, audio, sr)
    logger.info(f"Saved {n_sources} separated stems in {output_dir}.")

if __name__ == "__main__":
    from config import setup_logging
    setup_logging(level=logging.DEBUG)
    # Test reconstruction
    n_sources = 5
    n_bins = 1025
    n_frames = 100
    S_stft = np.random.randn(n_sources, n_bins, n_frames) + 1j * np.random.randn(n_sources, n_bins, n_frames)
    
    S_audio = reconstruct_audio_from_stft(S_stft)
