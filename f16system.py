import math
import jax
import jax.numpy as jnp
from numpy import deg2rad

from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16

from gcas import GcasAutopilot
from sensor import GaussianNoisySensor


CRASH_ALT = 100

def euler_integration(autopilot, sensor, x, dt, steps=10):
    for _ in range(steps):
        x_meas = sensor.apply_noise(x)
        xdot = controlled_f16(x, autopilot.get_u_ref(x_meas)).xd
        x = x + xdot * dt
    return x

"""
class TrajectoryDistribution:
    def __init__(self, Ps, D, d=1):
        self.Ps = Ps    # initial state distribution, leave deterministic for now
        self.D = D      # should be based on sensor noise
        self.d = d  # depth
"""



class F16System:
    def __init__(self, T: int = 1000,
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
                ):
        
        self.T = T
        self.dt = dt
        self.euler_steps = euler_steps
        self.power = power
        self.alpha = alpha
        self.beta = beta
        self.alt = alt
        self.vt = vt
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.p = p
        self.q = q
        self.r = r
    
        self.trajectories = []
        self.ap = GcasAutopilot()
        self.state = f16state(vt, [alpha, beta], [phi, theta, psi], [p, q, r], [0, 0, alt], power, [0, 0, 0])
        
    def step(self, x: jnp.ndarray, i: int, autopilot: GcasAutopilot, 
             sensor: GaussianNoisySensor, dt: float, steps: int) -> jnp.ndarray:
        return euler_integration(autopilot, sensor, x, dt, steps=steps), x

    # real-valued function that (for now) returns altitude of the state
    def mu(self, state):
        return state.alt
    
    # for each state x in trajectory xs, check to see if it is above c
    def isSuccess(self, xs, c=0):
        for x in xs:
            if self.mu(x) <= c:
                return False
        return True

    def robustness(self, state, c):
        return self.mu(state) - c
    
    # depth based on self.T
    def rollout(self):
        x, xs = jax.lax.scan(lambda carry, i: self.step(carry, i, self.ap, self.dt, self.euler_steps), x, jnp.arange(self.T))
        return xs
    
"""
TODO: Calculate trajectory likelihood.
TODO: run to make sure it works (this is untested)
"""
