import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    ## my notes this is just one step
    # Update biased first moment estimate
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    
    m_t = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v_t = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m_t / (1 - beta1 ** t)
    v_hat = v_t / (1 - beta2 ** t)

    # Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_t, v_t