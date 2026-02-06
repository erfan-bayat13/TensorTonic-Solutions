def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    K = len(predictions)

    loss = 0.0
    for i in range(K):
        if i == target:
            q = 1 - epsilon + (epsilon / K)
        else:
            q = epsilon / K

        loss += q * math.log(predictions[i])

    return -loss