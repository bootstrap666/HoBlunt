import numpy as np

class Power_Estimator:
    def __init__(self, alpha, initial_value=1.0):

        if not np.isscalar(alpha) or not isinstance(alpha, (int, float, np.number)):
            raise TypeError("alpha must be a numeric scalar")

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        
        self.alpha = alpha
        self.initial_value = initial_value
        self.reset()

    def update(self, new_sample):
        self._memory = (1.0-self.alpha)*self._memory + self.alpha*(new_sample**2)
        
    def read_estimate(self):
        return self._memory

    def update_alpha(self, alpha):
        if not np.isscalar(alpha) or not isinstance(alpha, (int, float, np.number)):
            raise TypeError("alpha must be a numeric scalar")

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha
        
    def reset(self):
        self._memory = self.initial_value
