import click
import numpy as np
import os
import logging
import librosa
from src.loader import load_stems, get_stem_matrix, find_stem_file
from src.mixer import mix_stems, save_mixes
from src.preprocessor import compute_stft, reorder_stft_for_ica, reconstruct_stft_from_ica
from src.ica import fast_ica, solve_permutation
from src.postprocessor import reconstruct_audio_from_stft, save_separated_stems
from src.evaluate import evaluate_separation, log_evaluation
from src.visualize import plot_waveforms, plot_matrix, plot_spectrograms
from src.exceptions import MartiniError, UnderdeterminedError, StemLoadError, ConvergenceError
from config import DATA_RAW_DIR, OUTPUTS_PLOTS_DIR, STEM_NAMES, setup_logging, SAMPLE_RATE

@click.group()
@click.option('--verbose', is_flag=True, help='Show debug messages.')
@click.option('--quiet', is_flag=True, help='Show only warnings and errors.')
@click.pass_context
def cli(ctx, verbose, quiet):
    """Martini — Blind Source Separation for Music."""
    # Determine log level
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.WARNING
    
    ctx.ensure_object(dict)
    ctx.obj['logger'] = setup_logging(level=log_level)

@cli.command()
@click.pass_context
def info(ctx):
    """Print information about input stems in data/raw/."""
    logger = ctx.obj['logger']
    logger.info("Scanning data/raw/ for stems...")
    
    found_stems = []
    for name in STEM_NAMES:
        path = find_stem_file(DATA_RAW_DIR, name)
        if path:
            try:
                duration = librosa.get_duration(path=path)
                sr = librosa.get_samplerate(path)
                found_stems.append((name, path, duration, sr))
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
    
    if not found_stems:
        logger.error(f"No stems found in {DATA_RAW_DIR}.")
        logger.info("Hint: Place .wav, .flac, or .mp3 files named after STEM_NAMES in data/raw/.")
        return

    logger.info(f"Found {len(found_stems)}/{len(STEM_NAMES)} stems:")
    logger.info("-" * 60)
    logger.info(f"{'Stem':<10} | {'Duration (s)':<12} | {'Sample Rate':<12} | {'Format'}")
    logger.info("-" * 60)
    for name, path, dur, sr in found_stems:
        ext = os.path.splitext(path)[1]
        logger.info(f"{name:<10} | {dur:<12.2f} | {sr:<12} | {ext}")
    logger.info("-" * 60)

@cli.command()
@click.option('--duration', default=10.0, help='Duration of audio to mix (seconds).')
@click.option('--seed', default=42, help='Random seed for mixing matrix.')
@click.option('--channels', type=click.Choice(['mono', 'stereo']), default='mono', help='Channel strategy.')
@click.pass_context
def mix(ctx, duration, seed, channels):
    """Generate mixes from raw stems and save to data/mixed/."""
    logger = ctx.obj['logger']
    logger.info("Starting mixing process...")
    
    try:
        stems_dict = load_stems(duration=duration, channel_strategy=channels)
        S_true = get_stem_matrix(stems_dict)
        logger.info(f"Loaded {S_true.shape[0]} stems ({duration}s).")
        
        X, A_true = mix_stems(S_true, seed=seed)
        save_mixes(X)
        logger.info(f"Generated mixes using random matrix (seed={seed}).")
        
        # Visualize mixing matrix
        plot_matrix(A_true, "True Mixing Matrix", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "true_mixing_matrix.png"))
        logger.info(f"Mixing matrix plot saved to {OUTPUTS_PLOTS_DIR}")
        
    except MartiniError as e:
        logger.error(f"Mixing failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

from src.pipeline import run_separation_pipeline

@cli.command()
@click.option('--duration', default=10.0, help='Duration of audio to process (seconds).')
@click.option('--mode', type=click.Choice(['time', 'frequency']), default='time', help='ICA domain (time or frequency).')
@click.option('--seed', default=42, help='Random seed for mixing matrix.')
@click.option('--contrast', type=click.Choice(['tanh', 'kurtosis']), default='tanh', help='Contrast function for ICA non-linearity.')
@click.option('--channels', type=click.Choice(['mono', 'stereo']), default='mono', help='Channel strategy.')
@click.option('--window', type=click.Choice(['hann', 'hamming', 'blackman']), default='hann', help='STFT window function.')
@click.option('--overlap', default=0.75, help='STFT overlap ratio.')
@click.option('--normalize', type=click.Choice(['peak', 'lufs', 'none']), default='peak', help='Output normalization strategy.')
@click.pass_context
def separate(ctx, duration, mode, seed, contrast, channels, window, overlap, normalize):
    """Run the full source separation pipeline (Mix -> Separate -> Evaluate)."""
    logger = ctx.obj['logger']
    logger.info(f"Starting Music Source Separation ({mode} domain, {contrast} contrast)...")
    
    try:
        results = run_separation_pipeline(
            duration=duration, 
            mode=mode, 
            seed=seed, 
            contrast=contrast, 
            channels=channels, 
            window=window, 
            overlap=overlap, 
            normalize=normalize
        )
        
        log_evaluation(results['sdr'], results['sir'], results['sar'], names=STEM_NAMES)
        logger.info("Done! Check the 'outputs' directory for results.")
        
    except MartiniError as e:
        logger.error(f"Separation failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    cli()
