def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    min_val = min(values)
    max_val = max(values)

    # handle constant values
    if max_val == min_val:
        return [0] * len(values)

    width = (max_val - min_val) / num_bins
    bins = []

    for v in values:
        b = int((v - min_val) / width)
        if b == num_bins:  # handle max value edge case
            b -= 1
        bins.append(b)

    return bins