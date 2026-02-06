import math

def binary_focal_loss(predictions, targets, alpha=0.25, gamma=2.0, eps=1e-8):
    """
    Compute mean binary focal loss for a list of predictions and targets.
    
    predictions: list of predicted probabilities for class 1
    targets: list of true labels (0 or 1)
    alpha: weighting factor for positive class
    gamma: focusing parameter
    eps: numerical stability to avoid log(0)
    """
    total_loss = 0.0
    n = len(predictions)

    for p, y in zip(predictions, targets):
        # Clip probability to avoid log(0)
        p = min(max(p, eps), 1.0 - eps)
        
        # Compute p_t
        p_t = p if y == 1 else 1.0 - p
        
        # Compute focal loss for this sample
        fl = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)
        total_loss += fl

    # Return mean loss
    return total_loss / n
