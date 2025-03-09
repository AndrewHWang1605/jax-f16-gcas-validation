import math
import jax
import jax.numpy as jnp
from numpy import deg2rad
from dataclasses import dataclass
from tqdm import tqdm

from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16

from gcas import GcasAutopilot
from sensor import GaussianNoisySensor

# Using altitude of 0 as the crash threshold
CRASH_ALT = 0

def euler_integration(autopilot, sensor, x, dt, steps=10):
    for _ in range(steps):
        # Not used in our rollout since step() does the integration.
        x_meas = sensor.apply_noise(x)[1]
        xdot = controlled_f16(x, autopilot.get_u_ref(x_meas)).xd
        x = x + xdot * dt
    return x

@dataclass
class FlightStep:
    time: float
    state: jnp.ndarray
    disturbance: jnp.ndarray  # Array of shape (num_substeps, state_dim)

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
                    sensor_seed: int = 0):
        
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
        
        # Define the sensor noise distribution.
        # Only the IMU channels (indices 6,7,8) have noise
        self.noise_mean = jnp.zeros(16)
        noise_std = jnp.zeros(16)
        noise_std = noise_std.at[6].set(0.05)   # Roll rate noise
        noise_std = noise_std.at[7].set(0.1)   # Pitch rate noise
        noise_std = noise_std.at[8].set(0.1)   # Yaw rate noise
        self.noise_std = noise_std
        self.noise_cov = jnp.diag(jnp.square(self.noise_std))
        # Create a stateless sensor (its apply_noise will accept a key).
        self.sensor = GaussianNoisySensor(self.noise_mean, self.noise_cov)
        
        # Save the initial flight state.
        self.initial_state = f16state(vt, [alpha, beta], [phi, theta, psi], [p, q, r], [0, 0, alt], power, [0, 0, 0])
        # Initialize the sensor key.
        self.sensor_key = jax.random.PRNGKey(sensor_seed)
        
    def step(self, carry, i, autopilot, sensor, dt, steps):
        """
        Perform one simulation step.
        The carry is a tuple (state, sensor_key).
        At each substep we sample a new disturbance (and update the sensor key),
        apply the disturbance, and update the state.
        We record all substep disturbances.
        """
        state, sensor_key = carry
        state_snapshot = state  # Record the state at the beginning of the time step.
        disturbances_list = []  # To store disturbance at each substep.
        new_state = state
        for _ in range(steps):
            sensor_key, noisy_state = sensor.apply_noise(sensor_key, new_state)
            # The disturbance is the difference between the noisy measurement and the true state.
            disturbance = noisy_state - new_state
            disturbances_list.append(disturbance)
            u_ref = autopilot.get_u_ref(noisy_state)
            xdot = controlled_f16(new_state, u_ref).xd
            new_state = new_state + xdot * dt
        disturbances_arr = jnp.stack(disturbances_list, axis=0)
        new_carry = (new_state, sensor_key)
        return new_carry, (state_snapshot, disturbances_arr)
    
    def mu(self, state):
        # Returns the altitude; index 11 holds the altitude.
        return state[11]
    
    def isSuccess(self, steps, c=CRASH_ALT):
        for step in steps:
            if self.mu(step.state) < c:
                return False
        return True

    def robustness(self, state, c):
        return self.mu(state) - c
    
    def rollout(self) -> FlightTrajectory:
        """
        Run the simulation over self.T timesteps using jax.lax.scan.
        The simulation always starts from the same deterministic initial state,
        but the sensor_key is carried through so that disturbances vary.
        After the rollout, we update self.sensor_key so subsequent rollouts
        are independent.
        """
        def scan_step(carry, i):
            return self.step(carry, i, self.ap, self.sensor, self.dt, self.euler_steps)
        
        # Each rollout starts from the same initial state.
        init_carry = (self.initial_state, self.sensor_key)
        final_carry, records = jax.lax.scan(scan_step, init_carry, jnp.arange(self.T))
        # Update the sensor key for future rollouts.
        self.sensor_key = final_carry[1]
        states, disturbances = records
        trajectory = FlightTrajectory()
        for i, (state, disturbance) in enumerate(zip(states, disturbances)):
            time = i * self.dt
            trajectory.add_step(time, state, disturbance)
        return trajectory
    
    def trajectory_log_likelihood(self, trajectory: FlightTrajectory) -> float:
        """
        Compute the log-likelihood of the given trajectory.
        We sum over all substeps recorded in each FlightStep.
        Only dimensions with nonzero noise (here, the IMU channels: indices 6, 7, 8)
        are used in the calculation.
        """
        def gaussian_log_pdf(x, mean, std):
            mask = std > 0
            masked_x = x[mask]
            masked_mean = mean[mask]
            masked_std = std[mask]
            log_coeff = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(masked_std)
            log_exponent = -0.5 * (((masked_x - masked_mean) / masked_std) ** 2)
            return jnp.sum(log_coeff + log_exponent)
        
        log_likelihood = 0.0
        for step in trajectory.get_trajectory():
            for substep in step.disturbance:
                log_likelihood += gaussian_log_pdf(substep, self.noise_mean, self.noise_std)
        return float(log_likelihood)

def direct_estimation(system: F16System, num_trials: int) -> float:
    """
    Run num_trials independent simulations using system.rollout() and compute
    the failure probability as the fraction of trajectories that fail (i.e.,
    at least one state has altitude < CRASH_ALT).
    For each trajectory, print its log likelihood and its minimum altitude.
    """
    failure_count = 0
    for i in tqdm(range(num_trials)):
        trajectory = system.rollout()
        log_likelihood = system.trajectory_log_likelihood(trajectory)
        # Compute the minimum altitude across the trajectory.
        min_altitude = min(system.mu(step.state) for step in trajectory.get_trajectory())
        print(f"Trajectory {i} log likelihood: {log_likelihood}, min altitude: {min_altitude}")
        
        states = trajectory.get_trajectory()
        if not system.isSuccess(states, c=CRASH_ALT):
            failure_count += 1
    return failure_count / num_trials
