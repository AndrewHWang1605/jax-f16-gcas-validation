
import math
import jax
import jax.numpy as jnp
from numpy import deg2rad

from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16

from gcas import GcasAutopilot
from sensor import GaussianNoisySensor


def euler_integration(autopilot, sensor, x, dt, steps=10):
    for _ in range(steps):
        x_meas = sensor.apply_noise(x)
        xdot = controlled_f16(x, autopilot.get_u_ref(x_meas)).xd
        x = x + xdot * dt
    return x


def step(x: jnp.ndarray, i: int, autopilot: GcasAutopilot, sensor: GaussianNoisySensor, dt: float, steps: int) -> jnp.ndarray:
    return euler_integration(autopilot, sensor, x, dt, steps=steps), x


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
    ''' 
    States: 
    0. vt: Air Speed
    1. alpha: Angle of Attack
    2. beta: Sideslip Angle
    3. phi: Roll
    4. theta: Pitch
    5. psi: Yaw
    6. P: Roll rate
    7. Q: Pitch rate
    8. R: Yaw rate
    9. Pn: Northward Displacement
    10.Pe: Eastward Displacement
    11.alt: Altitude
    12.pow: Engine power lag
    13.Nz: Upward Accel
    14.Ps: Stability Roll rate
    15.Ny+r: Side accel and yaw rate
    
    Simulate GCAS over time T with given parameters.
    '''
    ap = GcasAutopilot()

    noise_mean = jnp.zeros(16)
    noise_mean.at[11].set(10)
    noise_cov = jnp.diag(jnp.square(jnp.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 10, 10, 10, 1, 1, 1, 1])))
    sensor = GaussianNoisySensor(noise_mean, noise_cov)
    x = f16state(vt, [alpha, beta], [phi, theta, psi], [p, q, r], [0, 0, alt], power, [0, 0, 0])

    x, xs = jax.lax.scan(lambda carry, i: step(carry, i, ap, sensor, dt, euler_steps), x, jnp.arange(T))
    return xs

# xs = sim_gcas()