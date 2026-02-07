import numpy as np
def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    # Write code here
    X = np.asarray(X)
    n_samples, n_features = X.shape

    ## all pairwiese multis
    interactions_matrix = np.einsum('ij,ik->ijk', X, X) # shape: (n_samples, n_features, n_features)
    # Take only upper triangle (i < j)
    iu = np.triu_indices(n_features, k=1)
    interactions = interactions_matrix[:, iu[0], iu[1]]  # shape: (n_samples, n_interactions)

    # Append interactions to original features
    X_new = np.hstack([X, interactions])
    return X_new.tolist()
