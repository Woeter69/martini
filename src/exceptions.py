class MartiniError(Exception):
    """Base exception class for Martini."""
    pass

class UnderdeterminedError(MartiniError):
    """Raised when n_mixtures < n_sources."""
    pass

class StemLoadError(MartiniError):
    """Raised when stems fail to load."""
    pass

class ConvergenceError(MartiniError):
    """Raised when FastICA doesn't converge within n_iter."""
    pass
