from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from .regression_vector import RegressionVector

class SignalGenerator(ABC):

    @abstractmethod
    def generate(self, n_samples):
        pass

    @abstractmethod
    def autocorrelation(self, lag):
        """
        Return the autocorrelation r(lag).
        """
        pass

    @abstractmethod
    def generate_sample(self):
        pass        
    
class ARGenerator(SignalGenerator):
    """
    Autoregressive (AR) signal generator.

    The generated process is

        x(n) = -aᵀx(n-1) + u(n)

    where u(n) is a white Gaussian innovation process.
    """
    def __init__(
        self,
        denominator,
        process_variance=1.0,
        warmup_samples=1000,
        rng=None
    ):
        """
        Parameters:
            denominator: AR denominator coefficients.
            process_variance: Output variance.
            warmup_samples: Number of discarded initial samples.
            rng: NumPy random number generator.
        """
        self.denominator = np.asarray(
            denominator,
            dtype=np.float64
        ).ravel()
        if self.denominator.size == 0:
            raise ValueError("denominator length must be positive.")
        
        if process_variance < 0:
            raise ValueError("process_variance must be non-negative.")        
        
        if warmup_samples < 0:
            raise ValueError("warmup_samples must be non-negative.")
        
        self.process_variance = float(process_variance)

        self.warmup_samples = warmup_samples

        self.regressor = RegressionVector(
            len(self.denominator)
        )
        self._autocorrelation = None
   
        # Computes self.innovation_variance and self._autocorrelation 
        # up to max(len(self.denominator) + 1,len(self.regressor))
        self._initialize_statistics()
    
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def generate(self, n_samples):
        """
    Generate one realization of the AR process.

    Returns:
        AR signal with the warm-up samples discarded.
    """
        self.regressor.reset()
        
        total_samples = n_samples + self.warmup_samples

        x = np.zeros(total_samples, dtype=np.float64)

        innovation = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.innovation_variance),
            size=total_samples
        )
        
        for n in range(total_samples):

            sample = (
                -self.regressor.vector @ self.denominator
                + innovation[n]
            )

            self.regressor.update(sample)

            x[n] = sample

        return x[self.warmup_samples:]
        
    def generate_sample(self):
        innovation = self.rng.normal(
            0.0,
            np.sqrt(self.innovation_variance)
        )
    
        sample = -self.regressor.vector @ self.denominator + innovation
        self.regressor.update(sample)
    
        return sample        
        
    def _initialize_statistics(self):
        """
        Initialize the statistical description of the AR process.
    
        This method computes the innovation variance and initializes
        the autocorrelation cache used by analytical models.
        """        
        # Resolve Yule-Walker for r(0), ..., r(p)
        self._autocorrelation = self._solve_yule_walker().tolist()
    
        # Compute innovation variance
        self.innovation_variance = (
            self.process_variance
            + np.dot(
                self.denominator,
                self._autocorrelation[1:]
            )
        )
    
        # Extend the autocorrelation cache if necessary
        required_lag = self.regressor.length - 1
    
        if required_lag >= len(self._autocorrelation):
            self._extend_autocorrelation(required_lag)
            
    def _solve_yule_walker(self):
        """
        Solve the Yule-Walker equations.
    
        Returns
        -------
        numpy.ndarray
            Autocorrelation sequence
            [r(0), r(1), ..., r(p)].
        """
    
        p = len(self.denominator)
    
        # Unknowns:
        # [r(1), r(2), ..., r(p)]
    
        A = np.zeros((p, p), dtype=np.float64)
        b = np.zeros(p, dtype=np.float64)
    
        for k in range(1, p + 1):
    
            b[k - 1] = -self.process_variance * self.denominator[k - 1]
    
            for i in range(1, p + 1):
    
                value = 0.0
    
                if k == i:
                    value += 1.0
    
                lag = abs(k - i)
    
                if lag != 0:
                    value += self.denominator[lag - 1]
    
                A[k - 1, i - 1] = value
    
        r = np.empty(p + 1, dtype=np.float64)
    
        r[0] = self.process_variance
        r[1:] = np.linalg.solve(A, b)
    
        return r
        
    def _extend_autocorrelation(self, max_lag):
        """
        Extend the autocorrelation cache up to max_lag.
        """
        while len(self._autocorrelation) <= max_lag:
        
            k = len(self._autocorrelation)
        
            rk = 0.0
        
            for i, a in enumerate(self.denominator, start=1):
                rk -= a * self._autocorrelation[k - i]
        
            self._autocorrelation.append(rk)

    def autocorrelation(self,lag):

        lag = abs(lag)

        if len(self._autocorrelation) <= lag:
            self._extend_autocorrelation(lag)
            
        return self._autocorrelation[:lag]
            
class GaussianNoiseGenerator(SignalGenerator):

    def __init__(self, process_variance=1.0, rng=None):
        
        if process_variance < 0:
            raise ValueError("process_variance must be non-negative.")        

        self.process_variance = process_variance
        self.innovation_variance = process_variance
        
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate(self, n_samples):
        return self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.process_variance),
            size=n_samples
        )
        
    def autocorrelation(self, max_lag):

        r = np.zeros(max_lag + 1)
        r[0] = self.process_variance

        return r

    def generate_sample(self):
        return self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.process_variance),
        )


class SpeechFileGenerator(ARGenerator):
    def __init__(
        self,
        filename,
        coefficients_filename,
        process_variance=1.0,
        warmup_samples=1000,
        rng=None,
    ):
        if not Path(filename).is_file():
            raise FileNotFoundError(f"WAV file not found: {filename}")

        if not Path(coefficients_filename).is_file():
            raise FileNotFoundError(
                f"Coefficients file not found: {coefficients_filename}"
            )
            
        with np.load(coefficients_filename) as data:
            denominator = data["ar_coefficients"]

        super().__init__(
            denominator,
            process_variance,
            warmup_samples,
            rng,
        )

        self.samples, _ = librosa.load(filename, sr=None, mono=True)
        self.cursor = 0

    def generate(self, n_samples):
        n_samples = int(n_samples)
    
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
    
        if n_samples == 0:
            return np.empty(0, dtype=self.samples.dtype)
    
        file_length = len(self.samples)
        indices = (self.cursor + np.arange(n_samples)) % file_length
        result = self.samples[indices]
    
        self.cursor = (self.cursor + n_samples) % file_length
    
        return result

    def generate_sample(self):
        file_length = len(self.samples)
        
        result = self.samples[self.cursor]
    
        self.cursor = (self.cursor + 1) % file_length
    
        return result

        