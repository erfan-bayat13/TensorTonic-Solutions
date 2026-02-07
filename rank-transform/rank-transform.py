import numpy as np

def rank_transform(values):
    """
    Replace each value with its rank (1 = smallest). 
    Handles ties by assigning average ranks, works for empty arrays and floats.
    """
    values = np.asarray(values)
    
    if values.size == 0:
        return np.array([]).tolist()  # empty input edge case
    
    # get sorted indices
    sorter = np.argsort(values)
    
    # initial ranks (1-based)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, len(values) + 1)
    
    # handle ties: average ranks for identical values
    unique_vals, first_indices, counts = np.unique(values, return_index=True, return_counts=True)
    if np.any(counts > 1):  # only process if there are ties
        for val, count in zip(unique_vals, counts):
            if count > 1:
                tie_indices = np.where(values == val)[0]
                avg_rank = ranks[tie_indices].mean()
                ranks[tie_indices] = avg_rank
    
    return ranks.tolist()