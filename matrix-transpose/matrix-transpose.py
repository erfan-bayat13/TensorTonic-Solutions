import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A=np.asarray(A)
    M, N = A.shape
    B = np.zeros((N, M), dtype=A.dtype)

    # Manual indexing
    for i in range(M):
        for j in range(N):
            B[j, i] = A[i, j]

    return B