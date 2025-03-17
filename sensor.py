import math
from jax import random
import jax.numpy as jnp
from jax_f16.f16_utils import f16state
from abc import ABC, abstractmethod

class Sensor(ABC):
    @abstractmethod
    def get_noise(self, key, x_f16_state, i):
        # Abstract method, no implementation here
        pass

class GaussianNoisySensor(Sensor):
    def __init__(self, mean, cov, seed=0, eps=1e-8):
        self.mean = mean
        self.cov = cov
        self.eps = eps
        # Instead of storing a mutable key, we store a base key.
        self.base_key = random.PRNGKey(seed)
        self.cov_sqrt = self.compute_cov_sqrt(cov)

    def compute_cov_sqrt(self, cov):
        # Check if cov is diagonal.
        diag_cov = jnp.diag(jnp.diag(cov))
        if jnp.allclose(cov, diag_cov):
            # For a diagonal matrix, the square-root is just the sqrt of the diagonal elements.
            diag_sqrt = jnp.sqrt(jnp.diag(cov))
            return jnp.diag(diag_sqrt)
        else:
            # Otherwise, add a small epsilon to the diagonal to ensure positive definiteness.
            cov_pd = cov #+ self.eps * jnp.eye(cov.shape[0])
            return jnp.linalg.cholesky(cov_pd)

    def get_noise(self, key, x_f16: f16state, i):
        """
        Given an input key and a flight state x_f16, split the key,
        generate a noise sample, and return the updated key along with the
        noisy state.
        """
        key, subkey = random.split(key)
        z = random.normal(subkey, x_f16.shape)
        noise = self.mean + jnp.dot(z, self.cov_sqrt.T)
        return key, noise
    
class DetermNoiseSensor(Sensor):
    def __init__(self, noise_arr):
        self.noise_arr = noise_arr
        
    def get_noise(self, key, x_f16: f16state, ind):
        key, subkey = random.split(key)
        full_noise = jnp.zeros(16)
        full_noise = full_noise.at[6].set(self.noise_arr[ind,0])   # Roll rate noindse
        full_noise = full_noise.at[3].set(self.noise_arr[ind,1])   # Roll angle noise
        return key, full_noise
