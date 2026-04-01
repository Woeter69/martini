import mir_eval
import numpy as np

def evaluate_separation(S_true, S_est):
    """
    Evaluate separation quality using mir_eval (SDR, SIR, SAR).
    S_true: (N_SOURCES, N_SAMPLES)
    S_est: (N_SOURCES, N_SAMPLES)
    Returns: (sdr, sir, sar, perm)
    """
    # mir_eval expects (n_sources, n_samples) and handles permutation/scaling
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(S_true, S_est)
    
    return sdr, sir, sar, perm

def print_evaluation(sdr, sir, sar, names=None):
    """
    Print SDR/SIR/SAR for each source.
    """
    n_sources = len(sdr)
    print("\n--- Separation Quality Metrics (mir_eval) ---")
    print(f"{'Source':<15} | {'SDR (dB)':<10} | {'SIR (dB)':<10} | {'SAR (dB)':<10}")
    print("-" * 55)
    
    for i in range(n_sources):
        name = names[i] if names and i < len(names) else f"Source {i+1}"
        print(f"{name:<15} | {sdr[i]:<10.2f} | {sir[i]:<10.2f} | {sar[i]:<10.2f}")
    
    print("-" * 55)
    print(f"{'Average':<15} | {np.mean(sdr):<10.2f} | {np.mean(sir):<10.2f} | {np.mean(sar):<10.2f}\n")

if __name__ == "__main__":
    # Test evaluation
    n_sources = 5
    n_samples = 1000
    S_true = np.random.randn(n_sources, n_samples)
    S_est = S_true + 0.1 * np.random.randn(n_sources, n_samples)
    
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    print_evaluation(sdr, sir, sar)
