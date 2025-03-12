import math
from jax import random
import jax.numpy as jnp
from jax_f16.f16_utils import f16state

class GaussianNoisySensor:
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
            cov_pd = cov + self.eps * jnp.eye(cov.shape[0])
            return jnp.linalg.cholesky(cov_pd)

    def get_noise(self, key, x_f16: f16state):
        """
        Given an input key and a flight state x_f16, split the key,
        generate a noise sample, and return the updated key along with the
        noisy state.
        """
        key, subkey = random.split(key)
        z = random.normal(subkey, x_f16.shape)
        noise = self.mean + jnp.dot(z, self.cov_sqrt.T)
        return key, noise
