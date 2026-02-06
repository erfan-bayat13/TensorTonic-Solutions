import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    # If single vector, add batch dimension
    if anchor.ndim == 1:
        anchor   = anchor[None, :]
        positive = positive[None, :]
        negative = negative[None, :]

    # Squared Euclidean distances (vectorized)
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    # Triplet loss per sample
    loss = np.maximum(0.0, d_ap - d_an + margin)

    # Mean over batch
    return np.mean(loss)
