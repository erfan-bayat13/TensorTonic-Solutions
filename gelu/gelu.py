import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.atleast_1d(x).astype(float)  # scalar → 1D array
    erf_vec = np.vectorize(lambda v: math.erf(v / math.sqrt(2)))  # vectorize math.erf
    return 0.5 * x * (1 + erf_vec(x))
