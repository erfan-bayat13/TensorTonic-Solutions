import numpy as np
def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    values = np.asarray(values)
    unique, counts = np.unique(values, return_counts=True)
    freq_dict = dict(zip(unique, counts / len(values)))
    # Map each original value to its frequency proportion
    return np.array([freq_dict[v] for v in values]).tolist()