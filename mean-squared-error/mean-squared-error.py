import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    temp = 0
    c = 0
    for i in range(len(y_pred)):
        temp += (y_pred[i] - y_true[i])**2
        c +=1
    return temp / c
    

