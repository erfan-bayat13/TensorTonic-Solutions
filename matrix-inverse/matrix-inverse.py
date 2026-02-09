import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A,dtype=float)
    if A.ndim != 2:
        return None
    if A.shape[0] != A.shape[1]:
        return None
    
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None
    
    # Compute the inverse
    try:
        A_inv = np.linalg.inv(A)
        return A_inv
    except np.linalg.LinAlgError:
        # Handle cases where inv() fails (singular or near-singular matrices)
        return None
