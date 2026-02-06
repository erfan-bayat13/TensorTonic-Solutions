import numpy as np


def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.array(y_true)  # convert to NumPy array
    y_pred = np.array(y_pred)  # convert to NumPy array
    
    e = y_true - y_pred
    total_loss = 0.0
    
    for i in range(len(e)):
        if abs(e[i]) <= delta:
            total_loss += 0.5 * e[i]**2
        else:
            total_loss += delta * (abs(e[i]) - 0.5 * delta)
    
    return total_loss / len(e)
    