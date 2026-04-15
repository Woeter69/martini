import numpy as np
import pytest
from src.evaluate import evaluate_separation

def test_evaluate_separation_identity():
    """Evaluate signals that are nearly identical (perfect separation)."""
    n_sources = 2
    n_samples = 44100
    S_true = np.random.randn(n_sources, n_samples)
    S_est = S_true.copy()
    
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    
    # Perfect match should have high SDR/SIR/SAR
    assert np.all(sdr > 20)
    assert np.all(sir > 20)
    assert np.all(sar > 20)
    assert np.array_equal(perm, [0, 1])

def test_evaluate_separation_noise():
    """Evaluate signals that are pure noise (poor separation)."""
    n_sources = 2
    n_samples = 44100
    S_true = np.random.randn(n_sources, n_samples)
    S_est = np.random.randn(n_sources, n_samples)
    
    sdr, sir, sar, perm = evaluate_separation(S_true, S_est)
    
    # Near random noise match should have very low SDR
    assert np.mean(sdr) < 1.0
