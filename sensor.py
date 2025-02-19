"""
Implement sensor to simulate noise and inject into the F16 GCAS controller
"""

import math
from jax import random
import jax.numpy as jnp
from jax_f16.f16_utils import f16state

# Additive Gaussian noise
class GaussianNoisySensor:
    def __init__(self, mean, cov, seed=0):
        self.mean = mean
        self.cov = cov
        # Cache Cholesky decomposition of the covariance matrix
        self.cov_sqrt = jnp.linalg.cholesky(cov)

        # Define the random seed
        self.key = random.PRNGKey(seed)

    def apply_noise(self, x_f16 : f16state):
        key, subkey = random.split(self.key)  # Split the key for independent random generation
        z = random.normal(subkey, x_f16.shape)  # Generate one random sample
        noise = self.mean + jnp.dot(z, self.cov_sqrt.T)  # Apply transformation to match desired covariance
        return x_f16 + noise
    

