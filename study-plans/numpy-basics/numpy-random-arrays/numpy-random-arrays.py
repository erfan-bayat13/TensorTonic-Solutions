import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    np.random.seed(seed)

    # Generate based on the requested distribution
    if kind == 'uniform':
        arr = np.random.random(shape)  # U[0, 1)
    elif kind == 'normal':
        arr = np.random.standard_normal(shape)  # N(0, 1)
    else:
        raise ValueError("kind must be 'uniform' or 'normal'")

    # Ensure dtype is float64 (np.random already returns float64, but this is explicit)
    return arr.astype(np.float64)