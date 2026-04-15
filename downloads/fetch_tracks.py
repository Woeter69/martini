"""
Data Preparation Script for Martini.

This script provides instructions for obtaining real stem data and 
can generate synthetic audio signals for testing the ICA pipeline.
"""

import os
import numpy as np
import soundfile as sf
import logging
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW_DIR, STEM_NAMES, SAMPLE_RATE, setup_logging

logger = setup_logging()

def generate_synthetic_stems(duration=5.0, sr=SAMPLE_RATE):
    """
    Generate 5 synthetic sources that are statistically independent.
    
    1. Vocals: AM-modulated sine wave
    2. Drums: Periodic filtered noise bursts
    3. Bass: Low-frequency sine wave
    4. Guitar: Mid-frequency sawtooth wave
    5. Piano: Superposition of a few harmonics
    """
    t = np.linspace(0, duration, int(sr * duration))
    stems = {}
    
    # 1. Vocals (AM Sine)
    stems['vocals'] = np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
    
    # 2. Drums (Noise Bursts)
    noise = np.random.randn(len(t))
    envelope = np.zeros_like(t)
    for i in range(0, len(t), int(sr * 0.5)): # 2Hz beat
        burst_len = int(sr * 0.1)
        if i + burst_len < len(t):
            envelope[i:i+burst_len] = np.exp(-np.linspace(0, 5, burst_len))
    stems['drums'] = noise * envelope
    
    # 3. Bass (Low Sine)
    stems['bass'] = np.sin(2 * np.pi * 60 * t)
    
    # 4. Guitar (Sawtooth)
    stems['guitar'] = (t * 220 % 1.0) - 0.5
    
    # 5. Piano (Harmonics)
    stems['piano'] = np.sin(2 * np.pi * 261.63 * t) + 0.5 * np.sin(2 * np.pi * 523.25 * t)
    
    logger.info(f"Generated {len(stems)} synthetic stems.")
    
    for name, audio in stems.items():
        # Normalize and save
        audio = audio / np.max(np.abs(audio))
        path = os.path.join(DATA_RAW_DIR, f"{name}.wav")
        sf.write(path, audio, sr)
        logger.info(f"Saved synthetic {name} to {path}")

def print_data_instructions():
    """Print instructions for downloading real stem datasets."""
    print("\n" + "="*60)
    print("MARTINI DATA PREPARATION")
    print("="*60)
    print("\nTo use real-world data, please download the MUSDB18-HQ dataset.")
    print("Link: https://zenodo.org/record/3338373")
    print("\nSteps:")
    print("1. Download the dataset and extract the tracks.")
    print("2. Choose a track and extract its isolated stems.")
    print(f"3. Place the stem files (vocals, drums, etc.) in: {DATA_RAW_DIR}")
    print("4. Ensure they are named: vocals.wav, drums.wav, bass.wav, guitar.wav, piano.wav")
    print("\nAlternatively, run this script to generate synthetic test data:")
    print("python downloads/fetch_tracks.py --synthetic")
    print("="*60 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data for Martini.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test data.")
    args = parser.parse_args()
    
    if args.synthetic:
        generate_synthetic_stems()
    else:
        print_data_instructions()
