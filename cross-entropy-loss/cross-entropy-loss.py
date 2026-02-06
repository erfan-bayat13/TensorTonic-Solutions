import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get number of samples
    N = len(y_true)
    
    # Use advanced indexing to extract probabilities for true classes
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute negative log likelihood
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss