import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)  # make it 2D if it was 1D

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for step in range(steps):
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)
        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.sum(y_pred - y) / n_samples
        w -= lr * dw
        b -= lr * db

    return w, b
