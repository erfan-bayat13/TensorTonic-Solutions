def bellman_optimality_backup(P, R, gamma, V):
    S = len(V)
    A = len(P[0])

    V_new = []

    for s in range(S):
        best = float("-inf")

        for a in range(A):
            action_total = 0.0

            for sp in range(S):
                action_total += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])

            best = max(best, action_total)

        V_new.append(round(best, 4))

    return V_new