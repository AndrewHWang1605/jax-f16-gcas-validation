
import math
import jax
import jax.numpy as jnp
from numpy import deg2rad

from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16

from gcas import GcasAutopilot


def euler_integration(autopilot, x, dt, steps=10):
    for _ in range(steps):
        xdot = controlled_f16(x, autopilot.get_u_ref(x)).xd
        x = x + xdot * dt
    return x


def step(x: jnp.ndarray, i: int, autopilot: GcasAutopilot, dt: float, steps: int) -> jnp.ndarray:
    return euler_integration(autopilot, x, dt, steps=steps), x


def sim_gcas(
    T: int = 1000,
    dt: float = 1/500,
    euler_steps: int = 10,
    power: int = 9,
    alpha: float = deg2rad(2.1215),
    beta: float = 0.0,
    alt: float = 600.0,
    vt: float = 540.0,
    phi: float = -math.pi/8,
    theta: float = (-math.pi/2) * 0.3,
    psi: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
) -> jnp.ndarray:
    
    '''Simulate GCAS over time T with given parameters.'''
    ap = GcasAutopilot()
    x = f16state(vt, [alpha, beta], [phi, theta, psi], [p, q, r], [0, 0, alt], power, [0, 0, 0])

    x, xs = jax.lax.scan(lambda carry, i: step(carry, i, ap, dt, euler_steps), x, jnp.arange(T))
    return xs


xs = sim_gcas()