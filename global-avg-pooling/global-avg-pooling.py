import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here 
    if x.ndim == 3:          # (C, H, W)
        return x.mean(axis=(1, 2))
    
    elif x.ndim == 4:        # (N, C, H, W)
        return x.mean(axis=(2, 3))
    
    else:
        raise ValueError("Input must be 3D or 4D tensor")