"""
Separation Performance Evaluation Module.

This module provides functions for calculating blind source separation metrics
(SDR, SIR, SAR) using the mir_eval library.
"""

import mir_eval
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_separation(S_true, S_est):
    """
    Evaluate separation quality using BSS metrics (SDR, SIR, SAR).
    
    This function uses `mir_eval.separation.bss_eval_sources` which 
    handles permutation and scaling automatically.
    
    Parameters:
        S_true (np.ndarray): (N_SOURCES, N_SAMPLES) ground-truth signals.
        S_est (np.ndarray): (N_SOURCES, N_SAMPLES) estimated signals.
        
    Returns:
        tuple: (sdr, sir, sar, perm)
            - sdr: Signal-to-Distortion Ratio (higher is better).
            - sir: Signal-to-Interference Ratio.
            - sar: Signal-to-Artifact Ratio.
            - perm: The optimal permutation matching sources.
    """
    logger.info("Evaluating separation quality using mir_eval...")
    
    # mir_eval expects (n_sources, n_samples)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(S_true, S_est)
    
    return sdr, sir, sar, perm

def log_evaluation(sdr, sir, sar, names=None):
    """
    Log SDR/SIR/SAR for each source in a formatted table.
    
    Parameters:
        sdr (np.ndarray): Array of SDR values for each source.
        sir (np.ndarray): Array of SIR values for each source.
        sar (np.ndarray): Array of SAR values for each source.
        names (list): The names for each source channel.
    """
    n_sources = len(sdr)
    logger.info("--- Separation Quality Metrics (mir_eval) ---")
    logger.info(f"{'Source':<15} | {'SDR (dB)':<10} | {'SIR (dB)':<10} | {'SAR (dB)':<10}")
    logger.info("-" * 55)
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        logger.info(f"{name:<15} | {sdr[i]:<10.2f} | {sir[i]:<10.2f} | {sar[i]:<10.2f}")
    
    logger.info("-" * 55)
    # Calculate averages to show overall performance
    logger.info(f"{'Average':<15} | {np.mean(sdr):<10.2f} | {np.mean(sir):<10.2f} | {np.mean(sar):<10.2f}")

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    # Test evaluation with noise vs truth
    n_sources = 2
    n_samples = 1000
    S_true = np.random.randn(n_sources, n_samples)
    S_est = S_true + 0.1 * np.random.randn(n_sources, n_samples)
    
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    log_evaluation(sdr, sir, sar)
