import numpy as np
import scipy as sp

import random
from dataclasses import dataclass
from typing import Optional
from collections.abc import Callable

import warnings

# TODO: bug with job number mismatch under bad colored noises

@dataclass
class StochasticSource:
    seed: Optional[int] = None
    _rng: Optional[np.random.Generator] = None
            
    def sample(self, n: int, rng: Optional[np.random.Generator] = None):
        raise NotImplementedError("Sampling method not implemented.")  
    
    def reset(self, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        self.seed = seed
        self._rng = rng
        
    def _get_rng(self, rng: Optional[np.random.Generator]):
        if rng is not None:
            return rng
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)            
        return self._rng
   

class IID(StochasticSource):
    """
    Wrap any vectorized sampler that takes (n, rng) and returns an array of length n.

    Parameters
    ----------
    sampler : Callable[[int, np.random.Generator], np.ndarray]

    seed : Optional[int]

    Examples
    --------
    Plugging in any NumPy distribution:

    >>> src = IID(lambda n, r: r.normal(10.0, 2.0, size=n), seed=42)
    """
    def __init__(self, sampler: Callable = lambda n, r: r.normal(0.0, 1.0, size=n), seed: Optional[int] = None):
        super().__init__(seed)
        self.sampler = sampler
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None):
        rng = self._get_rng(rng)
        return self.sampler(n, rng)


# TODO: `Discrete` class is redundant? 
# I can pass IID(lambda n, r: r.choice(xs, p=ps, size=n)) to IID
class Discrete(IID):
    """_summary_

    Args:
        distribution (dict[float, float]): pass mapping of value to its probability
    """
    def __init__(self, distribution: dict[float, float]):
        self.values, self.probabilities = zip(*distribution.items())
        self.probabilities = np.asanyarray(self.probabilities)
        self.values = np.asanyarray(self.values)
        
        if not np.isclose(self.probabilities.sum(), 1.0):
            warnings.warn(f"Discrete distribution probabilities does sum to 1 ({self.probabilities.sum():.5f} instead). They will be renormalized.")
            self.probabilities /= self.probabilities.sum()

    def sample(self, n: int, rng: Optional[np.random.Generator] = None):
        rng = self._get_rng(rng)
        return rng.choice(self.values, size=n, p=self.probabilities).astype(float)


class AR1(StochasticSource):
    r"""
    First-order autoregressive process.
    
    $x_t = \rho x_{t-1} + \epsilon_t$
    
    Parameters
    ----------
    noise : StochasticSource
    """
    def __init__(
        self, 
        rho: float = 0.5, 
        x0: float = 0.0,
        noise: Optional[StochasticSource] = IID(lambda n, r: r.normal(0.0, 1.0, size=n)), 
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.rho = rho
        self._x0 = x0
        self.noise = noise

    # TODO: keep last state?        
    def sample(self, n: int, rng: Optional[np.random.Generator] = None):
        rng = self._get_rng(rng)
        noise_samples = (self.noise.sample(n, rng) if self.noise else np.zeros(n))
        samples = [self._x0 * self.rho + noise_samples[0]]
        for i in range(1, n):
            samples.append(self.rho * samples[i-1] + noise_samples[i])
        samples = np.array(samples).astype(float)
        return samples
    
# One-class colored noise with exponent alpha
class ColoredNoise(StochasticSource):   
    r"""
    Colored noise with power spectral density ~ 1 / f^{alpha}.

    Parameters
    ----------
    alpha : float
        Spectral exponent (0=white, 1=pink, 2=brown, -1=blue, -2=violet).
    mean : float
        Mean of the white noise before filtrations.
    std : float
        Std of the white noise before filtrations.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        mean: float = 0.0,
        std: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.alpha = float(alpha)
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None):
        rng = self._get_rng(rng)

        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1
        white = rng.normal(0, self.std, n)
        f_transform = np.fft.rfft(white)
        f_transform = f_transform / (freqs**self.alpha)
        samples = np.fft.irfft(f_transform)

        samples = samples + self.mean

        return samples.astype(float)
