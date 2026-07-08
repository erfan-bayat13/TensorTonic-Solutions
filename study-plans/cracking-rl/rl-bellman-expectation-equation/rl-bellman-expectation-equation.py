def bellman_expectation_backup(P, R, policy, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    S = len(V)
    A = len(policy[0])
    V_new = []
    
    for s in range(S):
        total = 0.0

        for a in range(A):
            action_total = 0.0

            for sp in range(S):
                action_total += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
            total += policy[s][a] * action_total
        V_new.append(round(total,4))
    return V_new
