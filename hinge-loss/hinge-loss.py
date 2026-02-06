import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    ans = []
    for i in range(len(y_score)):
        temp = max(0,margin-y_score[i] * y_true[i])
        ans.append(temp)
    if reduction == "mean":
        return float(np.mean(ans))
    elif reduction == "sum":
        return float(np.sum(ans))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")