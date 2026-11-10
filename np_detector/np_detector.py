import numpy as np

def np_detector(x, beta_fa, mu, xi, lambda_delta):
    threshold = beta_fa * np.sqrt( mu*xi/(lambda_delta**2) )
    return np.abs(x) > threshold