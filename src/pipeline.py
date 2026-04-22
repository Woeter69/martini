import os
import logging
import numpy as np
from src.loader import load_stems, get_stem_matrix
from src.mixer import mix_stems, save_mixes
from src.preprocessor import compute_stft, reorder_stft_for_ica, reconstruct_stft_from_ica
from src.ica import fast_ica, solve_permutation
from src.postprocessor import reconstruct_audio_from_stft, save_separated_stems
from src.evaluate import evaluate_separation, log_evaluation
from src.visualize import plot_waveforms, plot_matrix
from src.exceptions import MartiniError, StemLoadError, ConvergenceError
from config import DATA_RAW_DIR, OUTPUTS_PLOTS_DIR, STEM_NAMES

logger = logging.getLogger(__name__)

def run_ica_on_mixtures(X, duration=10.0, mode='time', contrast='tanh', 
                        window='hann', overlap=0.75, normalize='peak'):
    """
    Runs ICA on a provided mixture matrix.
    Returns estimated sources and unmixing matrix.
    """
    if mode == 'time':
        S_est, W_est, conv_info = fast_ica(X, contrast=contrast)
    else:
        X_stft, hop_length = compute_stft(X, window=window, overlap_ratio=overlap)
        X_reordered = reorder_stft_for_ica(X_stft)
        
        n_bins, n_sources, n_frames = X_reordered.shape
        Y_stft_reordered = np.zeros_like(X_reordered, dtype=complex)
        
        for b in range(n_bins):
            X_bin = X_reordered[b, :, :]
            X_bin_real = np.hstack([np.real(X_bin), np.imag(X_bin)])
            try:
                _, W_bin, _ = fast_ica(X_bin_real, contrast=contrast)
                Y_stft_reordered[b, :, :] = W_bin @ X_bin
            except ConvergenceError:
                Y_stft_reordered[b, :, :] = X_bin 
        
        Y_stft_reordered = solve_permutation(Y_stft_reordered)
        S_stft = reconstruct_stft_from_ica(Y_stft_reordered)
        S_est = reconstruct_audio_from_stft(S_stft, hop_length=hop_length)
        W_est = None

    save_separated_stems(S_est, normalize=normalize)
    return S_est, W_est

def run_separation_pipeline(duration=10.0, mode='time', seed=42, contrast='tanh', 
                            channels='mono', window='hann', overlap=0.75, normalize='peak'):
    """
    Executes the full Martini separation pipeline (Load -> Mix -> Separate -> Evaluate).
    Returns a dictionary with results.
    """
    results = {}
    
    # 1. Load Stems
    stems_dict = load_stems(duration=duration, channel_strategy=channels)
    S_true = get_stem_matrix(stems_dict)
    
    # 2. Mix Stems
    X, A_true = mix_stems(S_true, seed=seed)
    save_mixes(X)
    
    # 3. ICA Separation
    S_est, W_est = run_ica_on_mixtures(X, duration, mode, contrast, window, overlap, normalize)

    # 4. Evaluate
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    
    results.update({
        'sdr': sdr,
        'sir': sir,
        'sar': sar,
        'avg_sdr': np.mean(sdr),
        'S_true': S_true,
        'S_est': S_est,
        'X': X,
        'A_true': A_true
    })
    
    # 5. Visualize (PNGs for backward compatibility/CLI)
    plot_waveforms(X, "Mixed Channels", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "mixed_waveforms.png"))
    plot_waveforms(S_est, "Separated Channels", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "separated_waveforms.png"))
    plot_matrix(A_true, "True Mixing Matrix", save_path=os.path.join(OUTPUTS_PLOTS_DIR, "true_mixing_matrix.png"))
    
    if mode == 'time' and W_est is not None:
        plot_matrix(np.linalg.inv(W_est), "Estimated Mixing Matrix", 
                   save_path=os.path.join(OUTPUTS_PLOTS_DIR, "est_mixing_matrix.png"))
        
    return results
