import numpy as np

def min_max_scaling(data, eps=1e-8):
    data = np.asarray(data, dtype=float)
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)

    scaled = (data - min_val) / (max_val - min_val + eps)
    return scaled.tolist()
