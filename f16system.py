import math
import jax
import jax.numpy as jnp
import numpy as np
from numpy import deg2rad
from jax.scipy.stats.multivariate_normal import logpdf as multivar_gauss_logpdf
from dataclasses import dataclass
from tqdm import tqdm

from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16

from gcas import GcasAutopilot
from sensor import GaussianNoisySensor

CRASH_ALT = 0

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
                    sensor_seed: int = 0,
                    prop_noise_mean: jnp.array = None,
                    prop_noise_std: jnp.array = None):
        
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
        self.noise_mean = jnp.zeros(16)
        noise_std = jnp.zeros(16) + 1e-10
        noise_std = noise_std.at[6].set(0.03)   # Roll rate noise
        noise_std = noise_std.at[3].set(0.01)     # Roll angle noise
        self.noise_std = noise_std
        self.noise_cov = jnp.diag(jnp.square(self.noise_std))
        if prop_noise_std is None or prop_noise_mean is None:
            self.sensor = GaussianNoisySensor(self.noise_mean, self.noise_cov)
        else:   
            prop_noise_cov = jnp.diag(jnp.square(prop_noise_std))
            self.sensor = GaussianNoisySensor(prop_noise_mean, prop_noise_cov)
        
        # Save the initial flight state.
        self.initial_state = f16state(vt, [alpha, beta], [phi, theta, psi],
                                      [p, q, r], [0, 0, alt], power, [0, 0, 0])
        # Initialize the sensor key.
        self.sensor_key = jax.random.PRNGKey(sensor_seed)
        
    def step(self, carry, i, autopilot, sensor, dt, steps):
        """
        Perform one overall simulation step.
        The carry is a tuple: (state, sensor_key).
        At each Euler substep, we update the state with a newly sampled disturbance.
        We record all substep disturbances.
        """
        state, sensor_key = carry
        state_snapshot = state  # record the state at the beginning of the overall step
        disturbances_list = []
        new_state = state
        for _ in range(steps):
            sensor_key, noisy_state = sensor.apply_noise(sensor_key, new_state)
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
    
    def rollout_min_altitude_and_loglik(self):
        """
        A memory-optimized rollout that computes both the minimum altitude
        and the cumulative log likelihood over the trajectory.
        The carry is now a tuple: (state, sensor_key, min_alt, cum_log_like).
        We record at each overall step the current (min_alt, cum_log_like).
        """

        def scan_step(carry, i):
            state, sensor_key, min_alt, cum_log_like = carry
            new_carry, record = self.step((state, sensor_key), i, self.ap, self.sensor, self.dt, self.euler_steps)
            new_state, new_sensor_key = new_carry
            current_alt = self.mu(new_state)
            new_min_alt = jnp.minimum(min_alt, current_alt)
            _, disturbances_arr = record
            step_log_like = gaussian_log_pdf_traj(disturbances_arr, self.noise_mean, self.noise_cov)
            new_cum_log_like = cum_log_like + step_log_like
            return (new_state, new_sensor_key, new_min_alt, new_cum_log_like), (new_min_alt, new_cum_log_like, disturbances_arr)
        
        init_min_alt = self.mu(self.initial_state)
        init_cum_log_like = 0.0
        init_carry = (self.initial_state, self.sensor_key, init_min_alt, init_cum_log_like)
        final_carry, outputs = jax.lax.scan(scan_step, init_carry, jnp.arange(self.T))
        self.sensor_key = final_carry[1]
        final_min_alt = final_carry[2]
        final_cum_log_like = final_carry[3]
        return final_min_alt, final_cum_log_like, outputs


def direct_estimation(system: F16System, num_trials: int) -> float:
    """
    Run num_trials independent simulations.
    For each trajectory, print its final cumulative log likelihood,
    the progression of log likelihood (if desired), and the minimum altitude.
    Compute the failure probability as the fraction of trajectories that fail
    (i.e., at least one state has altitude < CRASH_ALT).
    """
    failure_count = 0
    min_alts = []
    log_likes = []
    for i in tqdm(range(num_trials), desc="Running Rollouts"):
        # Use the memory-optimized rollout that computes min altitude and cumulative log likelihood.
        min_alt, cum_log_like, progression = system.rollout_min_altitude_and_loglik()
        #print(f"Trajectory {i}: final cumulative log likelihood = {cum_log_like}, min altitude = {min_alt}")
        # Optionally, you could also print or plot the progression array.
        if min_alt < CRASH_ALT:
            failure_count += 1
        min_alts.append(min_alt)
        log_likes.append(cum_log_like)
    return failure_count / num_trials, min_alts, log_likes

def gaussian_log_pdf(x, mean, cov):
    return multivar_gauss_logpdf(x, mean, cov)

def gaussian_log_pdf_traj(disturbances_arr, mean, cov):
    step_log_like = 0.0
    for j in range(disturbances_arr.shape[0]):
        step_log_like += gaussian_log_pdf(disturbances_arr[j], mean, cov)
    return step_log_like

    

def generate_proposal_distributions(p_mean, p_std, num_proposals=1, scale=1):
    """
    generate proposal distributions by taking the nominal distribution
    parameters and perturbing them by a random value
    """
    qs = [] # proposal distributions
    mask = np.zeros(16) # which components we are going to change for proposal
    mask[3] = 1
    mask[6] = 1
    # create proposal distributions by slightly perturbing the parameters of nominal dist
    for i in range(num_proposals):
        # use mask to only change the noisy features
        q_mean = p_mean + (scale * np.random.randn(16) * mask)
        q_std = p_std # only change the mean for now
        # qs.append(jsp.stats.multivariate_normal(q_mean, system.noise_std))
        qs.append((q_mean, q_std))
    return qs

def imp_sampl_fail_est(p, qs, num_rollouts, DMMIS=False):
    q_systems = []
    for q in qs:
        q_system = F16System(T=300, prop_noise_mean=q[0], prop_noise_std=q[1])
        q_systems.append(q_system)
    rollouts = []
    ws = []

    if DMMIS:
        print("Not implemented")
        return 0.0
        # for q in tqdm(q_systems, desc="Running Rollouts"):
        #     for j in range(num_rollouts):
        #         rollouts.append(q.rollout_min_altitude_and_loglik())
        # for rollout in tqdm(rollouts, desc="Calculating weights"):
        #     traj = rollout[2][2].reshape(-1, 16)
        #     p_i = gaussian_log_pdf_traj(traj, p.noise_mean, p.noise_cov)
        #     print("\np_i:", p_i)
        #     denom = 0
        #     for q in qs:
        #         denom += gaussian_log_pdf_traj(traj, q[0], np.diag(np.square(q[1])))
        #     # print(denom)
        #     denom /= len(qs)
        #     # else:
        #     #     denom = rollout[1]
        #     print("\nDenom:", denom)
        #     ws.append(p_i / denom)

    else:
        for q in tqdm(q_systems, desc="Running Rollouts"):
            for j in range(num_rollouts):
                rollout = q.rollout_min_altitude_and_loglik()
                rollouts.append(rollout)
                traj = rollout[2][2].reshape(-1, 16)
                p_i = gaussian_log_pdf_traj(traj, p.noise_mean, p.noise_cov)
                q_i = gaussian_log_pdf_traj(traj, q.sensor.mean, q.sensor.cov)
                ws.append(np.exp(p_i - q_i))
    print("Finished proposal distribution likelihood")
    print(ws)
    weighted_sum = 0
    for i in range(len(rollouts)):
        if rollouts[i][0] < CRASH_ALT:
            print("Found failure!")
            weighted_sum += ws[i].item()
    return weighted_sum / len(rollouts)