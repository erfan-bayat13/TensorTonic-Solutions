import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    ## expand 
    h = x @ W1 + b1

    ## softmax
    a = np.maximum(0, h)

    ## contract 
    ffn = a @ W2 + b2
    return ffn