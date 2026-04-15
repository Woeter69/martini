import numpy as np
import pytest
from src.mixer import generate_mixing_matrix, mix_stems

def test_mixing_matrix_conditioning():
    """Ensure singular values are in a reasonable range (well-conditioned)."""
    n_sources = 5
    A = generate_mixing_matrix(n_sources=n_sources)
    u, s, vh = np.linalg.svd(A)
    assert np.all(s >= 0.49)
    assert np.all(s <= 2.01)

def test_mixing_signal_shape():
    """Test if mixing produces correct output signal dimensions."""
    n_sources = 3
    n_samples = 1000
    S = np.random.randn(n_sources, n_samples)
    X, A = mix_stems(S)
    
    assert X.shape == (n_sources, n_samples)
    assert A.shape == (n_sources, n_sources)

def test_mixing_invalid_input():
    with pytest.raises(ValueError):
        mix_stems(np.random.randn(5)) # 1D instead of 2D
