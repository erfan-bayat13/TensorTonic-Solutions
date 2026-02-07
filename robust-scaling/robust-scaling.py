def robust_scaling(values):
    n = len(values)
    s = sorted(values)

    # ---- MEDIAN ----
    if n % 2 == 0:
        median = (s[n//2 - 1] + s[n//2]) / 2.0
    else:
        median = s[n//2]

    # ---- LOWER / UPPER HALVES ----
    # Important: median NOT included in halves when odd
    lower = s[:n//2]
    upper = s[n//2:] if n % 2 == 0 else s[n//2+1:]

    # ---- MEDIAN FUNCTION ----
    def get_median(arr):
        m = len(arr)
        if m == 0:
            return median  # edge case n=1
        if m % 2 == 0:
            return (arr[m//2 - 1] + arr[m//2]) / 2.0
        return arr[m//2]

    q1 = get_median(lower)
    q3 = get_median(upper)

    iqr = q3 - q1

    # ---- SCALE ----
    if iqr == 0:
        return [float(x - median) for x in values]

    return [float((x - median) / iqr) for x in values]
