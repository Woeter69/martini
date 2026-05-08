"""
Data Preparation Script for Martini.
"""
import os
import numpy as np
import soundfile as sf
import logging
import sys
import librosa

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW_DIR, STEM_NAMES, SAMPLE_RATE, setup_logging

logger = setup_logging()

def generate_synthetic_stems(duration=10.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration))
    stems = {}
    stems['vocals'] = np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
    stems['drums'] = np.random.randn(len(t)) * (np.cos(2 * np.pi * 2 * t) > 0.9)
    stems['bass'] = np.sin(2 * np.pi * 60 * t)
    stems['guitar'] = (t * 220 % 1.0) - 0.5
    stems['piano'] = np.sin(2 * np.pi * 261 * t) + 0.5 * np.sin(2 * np.pi * 523 * t)
    for name, audio in stems.items():
        sf.write(os.path.join(DATA_RAW_DIR, f"{name}.wav"), audio / (np.max(np.abs(audio)) + 1e-9), sr)

def generate_real_stems(duration=10.0, sr=SAMPLE_RATE):
    stems_info = {'vocals': ('nutcracker', 'harmonic'), 'drums': ('pistachio', 'percussive'),
                  'bass': ('brahms', 'lowpass'), 'guitar': ('choice', 'highpass'), 'piano': ('sweetwaltz', 'harmonic')}
    for name, (example, proc) in stems_info.items():
        try:
            path = librosa.util.example(example); y, _ = librosa.load(path, sr=sr, duration=duration)
            if proc == 'harmonic': y_proc, _ = librosa.effects.hpss(y)
            elif proc == 'percussive': _, y_proc = librosa.effects.hpss(y)
            elif proc == 'lowpass':
                Y = np.fft.rfft(y); freqs = np.fft.rfftfreq(len(y), 1/sr); Y[freqs > 300] = 0; y_proc = np.fft.irfft(Y, n=len(y))
            elif proc == 'highpass':
                Y = np.fft.rfft(y); freqs = np.fft.rfftfreq(len(y), 1/sr); Y[freqs < 2000] = 0; y_proc = np.fft.irfft(Y, n=len(y))
            else: y_proc = y
            sf.write(os.path.join(DATA_RAW_DIR, f"{name}.wav"), y_proc / (np.max(np.abs(y_proc)) + 1e-9), sr)
            logger.info(f"Successfully generated {name}")
        except Exception as e:
            logger.error(f"Failed {name}: {e}")
            sf.write(os.path.join(DATA_RAW_DIR, f"{name}.wav"), np.random.randn(int(sr*duration)) * 0.1, sr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    if args.real: generate_real_stems()
    else: generate_synthetic_stems()
