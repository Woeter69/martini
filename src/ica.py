import numpy as np

def g(x):
    """
    Contrast function g(u) = tanh(u).
    """
    return np.tanh(x)

def g_prime(x):
    """
    Derivative of tanh(u) = 1 - tanh^2(u).
    """
    return 1 - np.square(np.tanh(x))

def center(X):
    """
    Center the data (subtract mean).
    X: (N_SOURCES, N_SAMPLES)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean

def whiten(X):
    """
    Whiten the data using PCA.
    Returns: (X_whitened, whitening_matrix, dewhitening_matrix)
    """
    # Covariance matrix
    cov = np.cov(X)
    # Eigendecomposition
    d, E = np.linalg.eigh(cov)
    # Whitening matrix: D^(-1/2) * E.T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    W_white = D_inv_sqrt @ E.T
    # Dewhitening matrix: E * D^(1/2)
    W_dewhite = E @ np.diag(np.sqrt(d + 1e-10))
    
    return W_white @ X, W_white, W_dewhite

def fast_ica_step(X_whitened, n_iter=100, tol=1e-5):
    """
    FastICA for a single component.
    """
    n_sources, n_samples = X_whitened.shape
    # Initialize random weight vector
    w = np.random.randn(n_sources)
    w /= np.linalg.norm(w)
    
    for i in range(n_iter):
        # Update rule: w+ = E[X * g(w.T * X)] - E[g'(w.T * X)] * w
        # Efficiently compute w.T * X
        wtx = w @ X_whitened # (n_samples,)
        
        # E[X * g(w.T * X)]
        term1 = (X_whitened * g(wtx)).mean(axis=1) # (n_sources,)
        
        # E[g'(w.T * X)] * w
        term2 = g_prime(wtx).mean() * w
        
        w_new = term1 - term2
        
        # Decorrelate / Normalize
        w_new /= np.linalg.norm(w_new)
        
        # Check convergence (absolute value because sign doesn't matter)
        if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol:
            break
        
        w = w_new
        
    return w

def fast_ica(X, n_sources=None, n_iter=200, tol=1e-5):
    """
    FastICA algorithm (Deflation approach).
    X: (N_SOURCES, N_SAMPLES)
    Returns: (S, W) where S = W @ X
    """
    if n_sources is None:
        n_sources = X.shape[0]
        
    # 1. Center
    X_centered = center(X)
    
    # 2. Whiten
    X_whitened, W_white, W_dewhite = whiten(X_centered)
    
    # 3. Iterate (Deflation)
    W_ica = np.zeros((n_sources, n_sources))
    
    for i in range(n_sources):
        w = np.random.randn(n_sources)
        for j in range(n_iter):
            wtx = w @ X_whitened
            w_new = (X_whitened * g(wtx)).mean(axis=1) - g_prime(wtx).mean() * w
            
            # Orthogonalize against previously found components
            if i > 0:
                w_new -= (w_new @ W_ica[:i].T) @ W_ica[:i]
                
            w_new /= np.linalg.norm(w_new)
            
            if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol:
                break
            w = w_new
            
        W_ica[i, :] = w
        
    # Un-mixing matrix W = W_ica @ W_white
    W = W_ica @ W_white
    S = W @ X
    
    return S, W

if __name__ == "__main__":
    # Test ICA with simple signal
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 1, 1000)
    s1 = np.sin(2 * np.pi * 5 * t)
    s2 = np.sign(np.sin(2 * np.pi * 3 * t))
    S_true = np.stack([s1, s2])
    
    # Mix
    A = np.array([[0.5, 0.5], [0.3, 0.7]])
    X = A @ S_true
    
    # Recover
    S_rec, W_rec = fast_ica(X)
    
    print(f"Original mixing matrix A:\n{A}")
    print(f"Recovered W inverse (estimated A):\n{np.linalg.inv(W_rec)}")
    
    # Check if we recovered something similar (ignoring scale/perm)
    # The user wanted a script, this is for dev check.
