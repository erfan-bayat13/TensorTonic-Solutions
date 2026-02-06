import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_mean = np.mean(real_scores)
    fake_mean = np.mean(fake_scores)
    out = fake_mean - real_mean
    return float(out)
