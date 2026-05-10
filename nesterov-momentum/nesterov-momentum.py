import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    Return updated parameters and velocity.
    """

    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: look-ahead position
    w_look = w - momentum * v

    # Step 2: update velocity
    v_new = momentum * v + lr * grad

    # Step 3: update weights
    w_new = w - v_new

    return w_new, v_new