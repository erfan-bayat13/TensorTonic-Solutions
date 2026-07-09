def policy_iteration(P, R, gamma, eval_tol=1e-8, max_iters=200):
    S = len(P)
    A = len(P[0])
    V = [0.0] * S

    def policy_eval(policy, V):
        while True:
            V_new = [0.0] * S

            for s in range(S):
                a = policy[s]
                total = 0.0

                for sp in range(S):
                    total += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])

                V_new[s] = total

            delta = max(abs(V_new[s] - V[s]) for s in range(S))
            V = V_new

            if delta < eval_tol:
                break

        return V

    def policy_imp(V):
        new_policy = [0] * S

        for s in range(S):
            best_q = float("-inf")
            best_a = 0

            for a in range(A):
                q = 0.0

                for sp in range(S):
                    q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])

                if q > best_q:
                    best_q = q
                    best_a = a

            new_policy[s] = best_a

        return new_policy

    policy = [0] * S

    counter = 0
    while counter < max_iters:
        V = policy_eval(policy, V)
        new_policy = policy_imp(V)

        if new_policy == policy:
            break

        policy = new_policy
        counter += 1

    return [round(v, 4) for v in V], policy