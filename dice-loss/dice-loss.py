import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    temp = np.sum(p * y)
    temp1,temp2 = np.sum(p), np.sum(y)

    dice = (2*temp + eps) / (temp1+temp2+eps)
    return 1-dice
