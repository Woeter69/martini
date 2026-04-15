import click
import numpy as np
import os
from src.loader import load_stems, get_stem_matrix
from src.mixer import mix_stems, save_mixes
from src.preprocessor import compute_stft, reorder_stft_for_ica, reconstruct_stft_from_ica
from src.ica import fast_ica, solve_permutation
from src.postprocessor import reconstruct_audio_from_stft, save_separated_stems
from src.evaluate import evaluate_separation, print_evaluation
from src.visualize import plot_waveforms, plot_matrix, plot_spectrograms
from config import DATA_RAW_DIR, OUTPUTS_PLOTS_DIR, STEM_NAMES

@click.command()
@click.option('--duration', default=10.0, help='Duration of audio to process (seconds).')
@click.option('--mode', type=click.Choice(['time', 'frequency']), default='time', help='ICA domain (time or frequency).')
@click.option('--seed', default=42, help='Random seed for mixing matrix.')
@click.option('--contrast', type=click.Choice(['tanh', 'kurtosis']), default='tanh', help='Contrast function for ICA non-linearity.')
def main(duration, mode, seed, contrast):
    click.echo(f"Starting Music Source Separation ({mode} domain, {contrast} contrast)...")
    
    # 1. Load Stems
    try:
        stems_dict = load_stems(duration=duration)
        S_true = get_stem_matrix(stems_dict)
        click.echo(f"Loaded {S_true.shape[0]} stems ({duration}s).")
    except Exception as e:
        click.echo(f"Error loading stems: {e}")
        click.echo("Hint: Use a script to download stems to data/raw/ or place .wav files there.")
        return

    # 2. Mix Stems
    X, A_true = mix_stems(S_true, seed=seed)
    save_mixes(X)
    click.echo(f"Generated mixes using random matrix.")

    # 3. ICA Separation
    if mode == 'time':
        click.echo("Running FastICA on time-domain signals...")
        S_est, W_est, conv_info = fast_ica(X, contrast=contrast)

        # Print per-component convergence
        click.echo("Convergence summary:")
        for ci in conv_info:
            status = "\u2713" if ci['converged'] else "\u2717"
            click.echo(f"  Component {ci['component']}: {status}  ({ci['iterations']} iterations)")
        n_conv = sum(1 for ci in conv_info if ci['converged'])
        click.echo(f"  {n_conv}/{len(conv_info)} components converged.")
    else:
        click.echo("Running FastICA on STFT frequency bins (this may take a while)...")
        X_stft = compute_stft(X)
        X_reordered = reorder_stft_for_ica(X_stft)  # (BINS, SOURCES, FRAMES)

        n_bins, n_sources_stft, n_frames = X_reordered.shape
        Y_stft_reordered = np.zeros_like(X_reordered, dtype=complex)
        bin_convergence = []  # collect per-bin convergence info

        # Frequency-Domain ICA: apply FastICA independently to each frequency bin
        for b in range(n_bins):
            if b % 100 == 0:
                click.echo(f"Processing bin {b}/{n_bins}...")

            X_bin = X_reordered[b, :, :]  # (n_sources, n_frames) complex

            # Learn the unmixing matrix from concatenated real/imag parts.
            X_bin_real = np.hstack([np.real(X_bin), np.imag(X_bin)])
            _, W_bin, conv_info = fast_ica(X_bin_real, contrast=contrast)
            bin_convergence.append(conv_info)

            # Apply the learned unmixing matrix to the complex STFT data
            Y_stft_reordered[b, :, :] = W_bin @ X_bin

        # --- Bin-wise convergence summary ---
        total_bins = len(bin_convergence)
        fully_converged = sum(
            1 for bc in bin_convergence if all(c['converged'] for c in bc)
        )
        all_iters = [c['iterations'] for bc in bin_convergence for c in bc]
        avg_iters = np.mean(all_iters)
        max_iters = int(np.max(all_iters))

        failed_bins = [
            i for i, bc in enumerate(bin_convergence)
            if not all(c['converged'] for c in bc)
        ]

        click.echo(f"\nBin-wise convergence summary:")
        click.echo(f"  Fully converged: {fully_converged}/{total_bins} bins ({100 * fully_converged / total_bins:.1f}%)")
        click.echo(f"  Avg iterations:  {avg_iters:.1f}  (max {max_iters})")
        if failed_bins:
            preview = failed_bins[:10]
            click.echo(f"  Failed bins:     {len(failed_bins)} (first 10: {preview})")

        # Solve the permutation ambiguity across frequency bins
        click.echo("Solving permutation alignment across bins...")
        Y_stft_reordered = solve_permutation(Y_stft_reordered)

        # Transpose from (BINS, SOURCES, FRAMES) -> (SOURCES, BINS, FRAMES)
        S_stft = reconstruct_stft_from_ica(Y_stft_reordered)
        S_est = reconstruct_audio_from_stft(S_stft)

    # 4. Save Separated Stems
    save_separated_stems(S_est)

    # 5. Evaluate
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    print_evaluation(sdr, sir, sar, names=STEM_NAMES)

    # 6. Visualize
    click.echo("Generating plots...")
    plot_waveforms(X, "Mixed Channels", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "mixed_waveforms.png"))
    plot_waveforms(S_est, "Separated Channels", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "separated_waveforms.png"))
    plot_matrix(A_true, "True Mixing Matrix", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "true_mixing_matrix.png"))
    
    # Heatmap of estimated unmixing matrix (or its inverse)
    if mode == 'time':
        plot_matrix(np.linalg.inv(W_est), "Estimated Mixing Matrix", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "est_mixing_matrix.png"))

    click.echo("Done! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()
