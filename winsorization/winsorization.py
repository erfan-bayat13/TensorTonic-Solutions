def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    # Write code here
    n = len(values)
    sorted_vals = sorted(values)
    def percentile(p):
        pos = p / 100 * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    lower_bound = percentile(lower_pct)
    upper_bound = percentile(upper_pct)

    res = []
    for v in values:
        if v < lower_bound:
            res.append(lower_bound)
        elif v > upper_bound:
            res.append(upper_bound)
        else:
            res.append(v)
    return res

