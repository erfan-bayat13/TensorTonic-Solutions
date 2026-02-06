import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    dot_product = np.dot(x1,x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    cos = dot_product / (norm_x1 * norm_x2)
    if label == 1:
        return 1 - cos
    else: 
        return max(0,cos - margin)
