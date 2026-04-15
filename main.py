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
        # 1. Load Stems
        try:
            stems_dict = load_stems(duration=duration, channel_strategy=channels)
            S_true = get_stem_matrix(stems_dict)
            logger.info(f"Loaded {S_true.shape[0]} stems ({duration}s).")
        except StemLoadError as e:
            logger.error(f"Error loading stems: {e}")
            logger.info("Hint: Use a script to download stems to data/raw/ or place .wav files there.")
            return

        # 2. Mix Stems
        X, A_true = mix_stems(S_true, seed=seed)
        save_mixes(X)
        logger.info(f"Generated mixes.")

        # 3. ICA Separation
        if mode == 'time':
            logger.info("Running FastICA on time-domain signals...")
            S_est, W_est, conv_info = fast_ica(X, contrast=contrast)
            
            # Log convergence summary
            logger.info("Convergence summary:")
            for ci in conv_info:
                status = "✓" if ci['converged'] else "✗"
                logger.info(f"  Component {ci['component']}: {status} ({ci['iterations']} iterations)")
            n_conv = sum(1 for ci in conv_info if ci['converged'])
            logger.info(f"  {n_conv}/{len(conv_info)} components converged.")
            
        else:
            logger.info("Running FastICA on STFT frequency bins (this may take a while)...")
            X_stft, hop_length = compute_stft(X, window=window, overlap_ratio=overlap)
            X_reordered = reorder_stft_for_ica(X_stft) # (BINS, SOURCES, FRAMES)
            
            n_bins, n_sources_stft, n_frames = X_reordered.shape
            Y_stft_reordered = np.zeros_like(X_reordered, dtype=complex)
            bin_convergence = []
            
            # Frequency-Domain ICA: apply FastICA independently to each frequency bin
            for b in range(n_bins):
                if b % 100 == 0:
                    logger.info(f"Processing bin {b}/{n_bins}...")
                
                X_bin = X_reordered[b, :, :] # (n_sources, n_frames) complex
                X_bin_real = np.hstack([np.real(X_bin), np.imag(X_bin)])
                
                try:
                    _, W_bin, conv_info = fast_ica(X_bin_real, contrast=contrast)
                    bin_convergence.append(conv_info)
                    Y_stft_reordered[b, :, :] = W_bin @ X_bin
                except ConvergenceError as e:
                    logger.warning(f"Bin {b} failed to converge: {e}")
                    Y_stft_reordered[b, :, :] = X_bin 
                    bin_convergence.append([])
                
            # --- Bin-wise convergence summary ---
            total_bins = len(bin_convergence)
            fully_converged = sum(1 for bc in bin_convergence if bc and all(c['converged'] for c in bc))
            all_iters = [c['iterations'] for bc in bin_convergence for c in bc]
            
            if all_iters:
                avg_iters = np.mean(all_iters)
                max_iters = int(np.max(all_iters))
                logger.info(f"Bin-wise convergence summary:")
                logger.info(f"  Fully converged: {fully_converged}/{total_bins} bins ({100 * fully_converged / total_bins:.1f}%)")
                logger.info(f"  Avg iterations:  {avg_iters:.1f}  (max {max_iters})")

            # Solve the permutation ambiguity across frequency bins
            logger.info("Solving permutation alignment across bins...")
            Y_stft_reordered = solve_permutation(Y_stft_reordered)

            # Transpose from (BINS, SOURCES, FRAMES) -> (SOURCES, BINS, FRAMES)
            S_stft = reconstruct_stft_from_ica(Y_stft_reordered)
            S_est = reconstruct_audio_from_stft(S_stft, hop_length=hop_length)

        # 4. Save Separated Stems
        save_separated_stems(S_est, normalize=normalize)

        # 5. Evaluate
        sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
        log_evaluation(sdr, sir, sar, names=STEM_NAMES)

        # 6. Visualize
        logger.info("Generating plots...")
        plot_waveforms(X, "Mixed Channels", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "mixed_waveforms.png"))
        plot_waveforms(S_est, "Separated Channels", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "separated_waveforms.png"))
        plot_matrix(A_true, "True Mixing Matrix", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "true_mixing_matrix.png"))
        
        if mode == 'time':
            plot_matrix(np.linalg.inv(W_est), "Estimated Mixing Matrix", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "est_mixing_matrix.png"))

        logger.info("Done! Check the 'outputs' directory for results.")
        
    except MartiniError as e:
        logger.error(f"Separation failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    cli()
