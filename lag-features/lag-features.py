import numpy as np
def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    # Write code here
    series = np.asarray(series)
    max_lag = max(lags)
    res = []
    for t in range(max_lag,len(series)):
        res.append([series[t-lag] for lag in lags])
    return res