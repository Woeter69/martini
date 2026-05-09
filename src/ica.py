"""
FastICA Engine Implementation.
"""
import numpy as np
import logging
from src.exceptions import ConvergenceError, UnderdeterminedError

logger = logging.getLogger(__name__)

def g_tanh(x): return np.tanh(x)
def g_prime_tanh(x): return 1 - np.square(np.tanh(x))
def g_kurtosis(x): return x ** 3
def g_prime_kurtosis(x): return 3 * np.square(x)

CONTRAST_FUNCTIONS = {'tanh': (g_tanh, g_prime_tanh), 'kurtosis': (g_kurtosis, g_prime_kurtosis)}

def center(X): return X - np.mean(X, axis=1, keepdims=True)

def whiten(X):
    cov = np.cov(center(X))
    d, E = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    W_white = D_inv_sqrt @ E.T
    return W_white @ X, W_white, E @ np.diag(np.sqrt(d + 1e-10))

def fast_ica(X, n_sources=None, n_iter=200, tol=1e-5, contrast='tanh'):
    g_func, g_prime_func = CONTRAST_FUNCTIONS[contrast]
    n_mixtures = X.shape[0]
    n_sources = n_sources or n_mixtures
    if n_mixtures < n_sources: raise UnderdeterminedError("Underdetermined problem")
    X_whitened, W_white, _ = whiten(X)
    W_ica = np.zeros((n_sources, n_mixtures))
    conv_info = []
    for i in range(n_sources):
        w = np.random.randn(n_mixtures); w /= np.linalg.norm(w)
        for j in range(n_iter):
            wtx = w @ X_whitened
            w_new = (X_whitened * g_func(wtx)).mean(axis=1) - g_prime_func(wtx).mean() * w
            if i > 0: w_new -= (w_new @ W_ica[:i].T) @ W_ica[:i]
            w_new /= np.linalg.norm(w_new)
            if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol: break
            w = w_new
        else: raise ConvergenceError(f"Component {i} failed to converge")
        W_ica[i, :] = w
        conv_info.append({'component': i, 'converged': True, 'iterations': j+1})
    W = W_ica @ W_white
    return W @ X, W, conv_info

def solve_permutation(Y_stft):
    from scipy.optimize import linear_sum_assignment
    n_bins, n_sources, n_frames = Y_stft.shape
    Y_aligned = Y_stft.copy()
    for b in range(1, n_bins):
        lookback = min(b, 3)
        ref_env = np.mean(np.abs(Y_aligned[b-lookback:b]), axis=0)
        cur_env = np.abs(Y_stft[b])
        cost = np.zeros((n_sources, n_sources))
        for i in range(n_sources):
            ri = ref_env[i] - np.mean(ref_env[i]); r_norm = np.linalg.norm(ri) + 1e-12
            for j in range(n_sources):
                cj = cur_env[j] - np.mean(cur_env[j]); c_norm = np.linalg.norm(cj) + 1e-12
                cost[i, j] = -np.dot(ri, cj) / (r_norm * c_norm)
        _, col_ind = linear_sum_assignment(cost)
        Y_aligned[b] = Y_stft[b, col_ind, :]
    return Y_aligned
