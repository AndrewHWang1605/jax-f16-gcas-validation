import math
import jax
import jax.numpy as jnp
from numpy import deg2rad
from dataclasses import dataclass

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

@dataclass
class FlightStep:
    time: float
    state: jnp.ndarray
    disturbance: jnp.ndarray

class FlightTrajectory:
    def __init__(self):
        self.steps = []

    def add_step(self, time, state, disturbance):
        self.steps.append(FlightStep(time, state, disturbance))

    def get_trajectory(self):
        return self.steps

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
        
        # Define the sensor noise distribution - 16-dimensional Gaussian noise
        self.noise_mean = jnp.zeros(16)
        self.noise_mean = self.noise_mean.at[11].set(10)
        self.noise_std = jnp.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 10, 10, 10, 1, 1, 1, 1])
        self.noise_cov = jnp.diag(jnp.square(self.noise_std))
        
        self.sensor = GaussianNoisySensor(self.noise_mean, self.noise_cov)
        
        self.state = f16state(vt, [alpha, beta], [phi, theta, psi], [p, q, r], [0, 0, alt], power, [0, 0, 0])
        
    def step(self, x: jnp.ndarray, i: int, autopilot: GcasAutopilot, 
             sensor: GaussianNoisySensor, dt: float, steps: int):
        """
        Perform one simulation step
        We sample the sensor disturbance once at the start, record the current state,
        then use the noisy measurement to compute the control input
        """
        # Sample the sensor disturbance once
        disturbance = sensor.apply_noise(x)
        # Record the state before integration
        state_snapshot = x
        # Compute control input using the disturbance
        u_ref = autopilot.get_u_ref(disturbance)
        new_x = x
        for _ in range(steps):
            xdot = controlled_f16(new_x, u_ref).xd
            new_x = new_x + xdot * dt
        # Return the updated state along with the recorded snapshot and disturbance
        return new_x, (state_snapshot, disturbance)
    
    # real-valued function that (for now) returns altitude of the state
    def mu(self, state):
        return state[11]  # altitude is index 11
    
    # for each state x in trajectory xs, check to see if it is above c
    def isSuccess(self, xs, c=0):
        for x in xs:
            if self.mu(x) <= c:
                return False
        return True

    def robustness(self, state, c):
        return self.mu(state) - c
    
    def rollout(self) -> FlightTrajectory:
        """
        Run the simulation over self.T timesteps using jax.lax.scan
        and record at each step the state snapshot and sensor disturbance
        """
        def scan_step(carry, i):
            new_state, record = self.step(carry, i, self.ap, self.sensor, self.dt, self.euler_steps)
            return new_state, record
        
        init_state = self.state
        final_state, records = jax.lax.scan(scan_step, init_state, jnp.arange(self.T))
        states, disturbances = records
        trajectory = FlightTrajectory()
        for i, (state, disturbance) in enumerate(zip(states, disturbances)):
            time = i * self.dt
            trajectory.add_step(time, state, disturbance)
        return trajectory
    
    def trajectory_log_likelihood(self, trajectory: FlightTrajectory) -> float:
        """
        Compute the log-likelihood of the given trajectory
        Good for numerical stability!
        """
        def gaussian_log_pdf(x, mean, std):
            log_coeff = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(std)
            log_exponent = -0.5 * (((x - mean) / std) ** 2)
            return jnp.sum(log_coeff + log_exponent)
        
        log_likelihood = 0.0
        for step in trajectory.get_trajectory():
            log_p = gaussian_log_pdf(step.disturbance, self.noise_mean, self.noise_std)
            log_likelihood += log_p
        return float(log_likelihood)
