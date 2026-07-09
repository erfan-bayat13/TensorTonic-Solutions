import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    if kind == 'arange':
        # param is the step size (float)
        arr = np.arange(start, stop, param, dtype=np.float64)
    elif kind == 'linspace':
        # param is the number of points (cast to int)
        num = int(param)
        arr = np.linspace(start, stop, num=num, dtype=np.float64)
    else:
        raise ValueError("kind must be 'arange' or 'linspace'")

    return arr.astype(np.float64)