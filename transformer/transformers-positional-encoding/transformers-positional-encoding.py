import numpy as np
import torch
import math

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = torch.zeros(seq_length, d_model)

    position = torch.arange(seq_length).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )

    angles = position * div_term

    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe.numpy()