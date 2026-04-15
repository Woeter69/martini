import mir_eval
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_separation(S_true, S_est):
    """
    Evaluate separation quality using mir_eval (SDR, SIR, SAR).
    S_true: (N_SOURCES, N_SAMPLES)
    S_est: (N_SOURCES, N_SAMPLES)
    Returns: (sdr, sir, sar, perm)
    """
    logger.info("Evaluating separation quality...")
    # mir_eval expects (n_sources, n_samples) and handles permutation/scaling
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(S_true, S_est)
    
    return sdr, sir, sar, perm

def log_evaluation(sdr, sir, sar, names=None):
    """
    Log SDR/SIR/SAR for each source.
    """
    n_sources = len(sdr)
    logger.info("--- Separation Quality Metrics (mir_eval) ---")
    logger.info(f"{'Source':<15} | {'SDR (dB)':<10} | {'SIR (dB)':<10} | {'SAR (dB)':<10}")
    logger.info("-" * 55)
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        logger.info(f"{name:<15} | {sdr[i]:<10.2f} | {sir[i]:<10.2f} | {sar[i]:<10.2f}")
    
    logger.info("-" * 55)
    logger.info(f"{'Average':<15} | {np.mean(sdr):<10.2f} | {np.mean(sir):<10.2f} | {np.mean(sar):<10.2f}")

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    # Test evaluation
    n_sources = 5
    n_samples = 1000
    S_true = np.random.randn(n_sources, n_samples)
    S_est = S_true + 0.1 * np.random.randn(n_sources, n_samples)
    
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    log_evaluation(sdr, sir, sar)
