import numpy as np
def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    values = np.asarray(values)
    res = values[:,None] ** np.arange(0,degree+1)
    return res.tolist()