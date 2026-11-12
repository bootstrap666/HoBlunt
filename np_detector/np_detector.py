import numpy as np

def np_detector(x, beta_fa, mu, xi, lambda_delta):
    threshold = beta_fa * np.sqrt( mu*xi/(lambda_delta) )
    return np.abs(x) > threshold