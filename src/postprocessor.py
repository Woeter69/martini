"""
Audio Postprocessing and Reconstruction Module.
"""
import numpy as np
import librosa
import soundfile as sf
import os
import logging
from config import SAMPLE_RATE, HOP_LENGTH, OUTPUTS_STEMS_DIR, STEM_NAMES

logger = logging.getLogger(__name__)

def reconstruct_audio_from_stft(S_stft, hop_length=HOP_LENGTH):
    return np.stack([librosa.istft(S_stft[i], hop_length=hop_length) for i in range(S_stft.shape[0])])

def postprocess_audio(audio, sr=SAMPLE_RATE, remove_dc=True, normalize='peak', fade_ms=10):
    if remove_dc: audio = audio - np.mean(audio)
    fade_len = int(fade_ms * sr / 1000)
    if len(audio) > 2 * fade_len:
        fade_in = np.linspace(0, 1, fade_len); fade_out = np.linspace(1, 0, fade_len)
        audio[:fade_len] *= fade_in; audio[-fade_len:] *= fade_out
    if normalize == 'peak':
        mv = np.max(np.abs(audio))
        if mv > 1e-6: audio /= mv
    elif normalize == 'lufs':
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms > 1e-6: audio *= (0.2 / rms)
        mv = np.max(np.abs(audio))
        if mv > 0.99: audio /= (mv + 1e-6)
    return audio

def save_separated_stems(S, names=STEM_NAMES, output_dir=OUTPUTS_STEMS_DIR, sr=SAMPLE_RATE, normalize='peak'):
    for i in range(S.shape[0]):
        name = names[i] if i < len(names) else f"source_{i}"
        file_path = os.path.join(output_dir, f"separated_{name}.wav")
        audio = postprocess_audio(S[i], sr=sr, normalize=normalize)
        sf.write(file_path, audio, sr)
    logger.info(f"Saved {S.shape[0]} separated stems in {output_dir}.")
