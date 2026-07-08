def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    G_t = []
    G = 0

    for r in reversed(rewards):
        G = float(r) + gamma * G
        G_t.insert(0, round(G, 4))

    return G_t