import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)

    # Handle empty input
    if N == 0:
        return np.array([]).reshape(0, 0)

    # Determine max length
    if max_len is None:
        L = max((len(seq) for seq in seqs), default=0)
    else:
        L = max_len

    # Create padded array
    out = np.full((N, L), pad_value)

    # Fill it
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        out[i, :length] = seq[:length]

    return out