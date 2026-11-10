import numpy as np


class RegressionVector:
    """
    Regression vector

        x(n) = [x[n], x[n-1], ..., x[n-M+1]]ᵀ
    """

    def __init__(self, length, dtype=np.float64):
        self.dtype=dtype
        if length <= 0:
            raise ValueError("length must be positive.")
        self.length = length
        self.vector = np.zeros(length, dtype=dtype)

    def reset(self):
        """Reset the regression vector."""
        self.vector.fill(0)
        
    def initialize(self, x):
        """
        Initialize the regression vector.

        Parameters
        ----------
        x : array_like
            Initial regression vector. The first element must
            correspond to the most recent sample.
        """
        x = np.asarray(x, dtype=self.dtype).ravel()

        if x.size != self.vector.size:
            raise ValueError(f"Received {x.size} but expected {self.vector.size}")

        self.vector[:] = x

    def update(self, sample):
        """
        Update the regression vector.

        Parameters
        ----------
        sample : float
            New sample inserted at the beginning of the
            regression vector.

        Returns
        -------
        ndarray
            Updated regression vector.
        """
        self.vector[1:] = self.vector[:-1]
        self.vector[0] = sample

        return self.vector
    
    def copy(self):
        """Return a copy of the regression vector."""
        return self.vector.copy()