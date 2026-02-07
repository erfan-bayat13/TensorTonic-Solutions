def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    d = {}
    for i in range(len(ordering)):
        if ordering[i] in d:
            pass
        else:
            d[ordering[i]] = i
    res = []
    for i in range(len(values)):
        if values[i] in d:
            res.append(d[values[i]])
    return res