import numpy as np
import pytest
from src.ica import fast_ica, center, whiten
from src.exceptions import UnderdeterminedError

def test_centering():
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    X_centered = center(X)
    assert np.allclose(np.mean(X_centered, axis=1), 0)

def test_whitening():
    X = np.random.randn(2, 1000)
    X_whitened, W_white, W_dewhite = whiten(X)
    cov = np.cov(X_whitened)
    assert np.allclose(cov, np.eye(2), atol=1e-2)

def test_fast_ica_simple():
    """Test if FastICA can recover independent sine waves."""
    t = np.linspace(0, 1, 1000)
    s1 = np.sin(2 * np.pi * 5 * t)
    s2 = np.sign(np.sin(2 * np.pi * 3 * t))
    S_true = np.stack([s1, s2])
    
    A = np.array([[0.5, 0.5], [0.3, 0.7]])
    X = A @ S_true
    
    S_est, W_est, conv_info = fast_ica(X, n_iter=500)
    
    assert all(ci['converged'] for ci in conv_info)
    assert S_est.shape == S_true.shape

def test_underdetermined_error():
    X = np.random.randn(2, 100)
    with pytest.raises(UnderdeterminedError):
        fast_ica(X, n_sources=3)
