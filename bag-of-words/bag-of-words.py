import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    counts = {word: 0 for word in vocab}

    # Count tokens
    for token in tokens:
        if token in counts:
            counts[token] += 1

    # Convert to numpy array in vocab order
    bow = np.array([counts[word] for word in vocab], dtype=int)

    return bow