import os
import logging
import numpy as np
from src.loader import load_stems, get_stem_matrix
from src.mixer import mix_stems, save_mixes
from src.preprocessor import compute_stft, reorder_stft_for_ica, reconstruct_stft_from_ica
from src.ica import fast_ica, solve_permutation
from src.postprocessor import reconstruct_audio_from_stft, save_separated_stems
from src.evaluate import evaluate_separation
from src.visualize import plot_waveforms, plot_matrix
from config import OUTPUTS_PLOTS_DIR, STEM_NAMES

logger = logging.getLogger(__name__)

def run_ica_on_mixtures(X, duration=10.0, mode='time', contrast='tanh', 
                        window='hann', overlap=0.75, normalize='peak', n_fft=2048):
    if mode == 'time':
        S_est, W_est, _ = fast_ica(X, contrast=contrast)
    else:
        X_stft, hop_length = compute_stft(X, window=window, overlap_ratio=overlap, n_fft=n_fft)
        X_reordered = reorder_stft_for_ica(X_stft)
        n_bins, n_sources, _ = X_reordered.shape
        Y_st = np.zeros_like(X_reordered, dtype=complex)
        for b in range(n_bins):
            Xb = X_reordered[b]
            Xbr = np.hstack([np.real(Xb), np.imag(Xb)])
            try:
                _, Wb, _ = fast_ica(Xbr, contrast=contrast)
                Y_st[b] = Wb @ Xb
            except Exception:
                Y_st[b] = Xb 
        Y_st = solve_permutation(Y_st)
        S_est = reconstruct_audio_from_stft(reconstruct_stft_from_ica(Y_st), hop_length=hop_length)
        W_est = None
    save_separated_stems(S_est, normalize=normalize)
    return S_est, W_est

def run_separation_pipeline(duration=10.0, mode='time', seed=42, contrast='tanh', 
                            channels='mono', window='hann', overlap=0.75, normalize='peak', n_fft=2048):
    stems_dict = load_stems(duration=duration, channel_strategy=channels)
    S_true = get_stem_matrix(stems_dict)
    X, A_true = mix_stems(S_true, seed=seed)
    save_mixes(X)
    S_est, W_est = run_ica_on_mixtures(X, duration, mode, contrast, window, overlap, normalize, n_fft)
    
    # Source Alignment (Match S_est to S_true labels using correlation)
    from scipy.optimize import linear_sum_assignment
    n_sources = S_est.shape[0]
    cost = np.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            corr = np.corrcoef(S_true[i], S_est[j])[0, 1]
            cost[i, j] = -np.abs(corr)
    _, col_ind = linear_sum_assignment(cost)
    S_est = S_est[col_ind]
    
    sdr, sir, sar, _ = evaluate_separation(S_true, S_est)
    return {'sdr': sdr, 'sir': sir, 'sar': sar, 'S_true': S_true, 'S_est': S_est, 'X': X, 'A_true': A_true}
