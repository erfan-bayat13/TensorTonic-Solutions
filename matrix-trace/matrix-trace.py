import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements) manually.
    
    Returns:
        A single scalar (int or float)
    """
    A = np.asarray(A)  # ensure input is a NumPy array
    n, m = A.shape
    
    if n != m:
        raise ValueError("Input must be a square matrix")
    
    trace_sum = 0
    for i in range(n):
        trace_sum += A[i, i]
    
    return trace_sum
