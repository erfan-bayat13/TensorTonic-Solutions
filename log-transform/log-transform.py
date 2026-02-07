def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    for i in range(len(values)):
        values[i] = math.log(values[i]+1)
    return values
    ## or return [math.log1p(v) for v in values]