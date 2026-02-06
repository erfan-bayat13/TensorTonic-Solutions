def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    ans = []
    for i in range(len(values)):
        theta = (2*math.pi*values[i]) / period
        enc = [math.sin(theta),math.cos(theta)]
        ans.append(enc)
    return ans