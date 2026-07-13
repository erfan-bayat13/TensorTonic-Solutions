import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    Scaled dot-product attention.
    """
    d_k = Q.shape[-1]

    scores = Q @ np.swapaxes(K, -2, -1)
    scores = scores / np.sqrt(d_k)

    weights = softmax(scores, axis=-1)

    return weights @ V


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int
) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    # Project into the full d_model space
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    d_model = Q_proj.shape[-1]

    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads

    heads = []

    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim

        Q_head = Q_proj[..., start:end]
        K_head = K_proj[..., start:end]
        V_head = V_proj[..., start:end]

        head_output = attention(Q_head, K_head, V_head)
        heads.append(head_output)

    # Concatenate along the feature dimension
    concatenated = np.concatenate(heads, axis=-1)

    # Final output projection
    return concatenated @ W_o