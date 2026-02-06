import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Add eps to q to avoid log(0)
    q = q + eps

    # Mask where p > 0 (p[i] = 0 contributes 0)
    mask = p > 0

    # Elementwise computation and sum
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return kl

    