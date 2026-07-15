import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    # compute mean
    mean = np.mean(x,axis=-1, keepdims=True)
    # var
    var = np.var(x,axis=-1, keepdims=True)
    # normalize 
    x_norm = (x - mean) / (np.sqrt(var + eps))

    output = gamma * x_norm + beta

    return output
    