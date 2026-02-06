import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)

    loss_pos = -y * (1 - p)**gamma * np.log(p)
    loss_neg = -(1 - y) * p**gamma * np.log(1 - p)
    loss = loss_pos + loss_neg
    return float(np.mean(loss))
    