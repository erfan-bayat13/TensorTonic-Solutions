import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    temp = np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b
    h_t = np.tanh(temp)
    return h_t
