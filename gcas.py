import math
import jax
import numpy as np
from numpy import deg2rad


import jax.numpy as jnp
from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16


from jax_f16.f16_types import S


class GcasAutopilot:
    '''ground collision avoidance autopilot'''

    def __init__(self, stdout=False):

        # config
        self.cfg_eps_phi = deg2rad(5)       # Max abs roll angle before pull
        self.cfg_eps_p = deg2rad(10)        # Max abs roll rate before pull
        self.cfg_path_goal = deg2rad(0)     # Min path angle before completion
        self.cfg_k_prop = 4                 # Proportional control gain
        self.cfg_k_der = 2                  # Derivative control gain
        self.cfg_flight_deck = 1000         # Altitude at which GCAS activates
        self.cfg_min_pull_time = 2          # Min duration of pull up

        self.cfg_nz_des = 5

        self.pull_start_time = 0
        self.stdout = stdout

        self.waiting_cmd = jnp.zeros(4)
        self.waiting_time = 2

    def log(self, s):
        'print to terminal if stdout is true'

        if self.stdout:
            print(s)

    def are_wings_level(self, x_f16):
        'are the wings level?'

        phi = x_f16[S.PHI]

        radsFromWingsLevel = jnp.round(phi / (2 * jnp.pi))

        return jnp.abs(phi - (2 * jnp.pi)  * radsFromWingsLevel) < self.cfg_eps_phi

    def is_roll_rate_low(self, x_f16):
        'is the roll rate low enough to switch to pull?'

        p = x_f16[S.P]

        return abs(p) < self.cfg_eps_p

    def is_above_flight_deck(self, x_f16):
        'is the aircraft above the flight deck?'

        alt = x_f16[S.ALT]

        return alt >= self.cfg_flight_deck

    def is_nose_high_enough(self, x_f16):
        'is the nose high enough?'

        theta = x_f16[S.THETA]
        alpha = x_f16[S.ALPHA]

        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromNoseLevel = jnp.round((theta-alpha)/(2 * jnp.pi))

        # Evaluate boolean
        return ((theta-alpha) - 2 * jnp.pi * radsFromNoseLevel) > self.cfg_path_goal

    def get_u_ref(self, x_f16):
        '''get the reference input signals'''
        def roll_or_pull():
            roll_condition = jnp.logical_and(self.is_roll_rate_low(x_f16), self.are_wings_level(x_f16))
            return jax.lax.cond(roll_condition, lambda _: self.pull_nose_level(), lambda _: self.roll_wings_level(x_f16), None)

        def standby_or_roll():
            standby_condition = jnp.logical_and(jnp.logical_not(self.is_nose_high_enough(x_f16)), jnp.logical_not(self.is_above_flight_deck(x_f16)))
            return jax.lax.cond(standby_condition, lambda _: roll_or_pull(), lambda _: jnp.zeros(4), None)

        pull_condition = jnp.logical_and(self.is_nose_high_enough(x_f16), True)
        return jax.lax.cond(pull_condition, lambda _: jnp.zeros(4), lambda _: standby_or_roll(), None)
    

    def pull_nose_level(self):
        'get commands in mode PULL'
        rv = jnp.array([self.cfg_nz_des, 0.0, 0.0, 0.0]) 

        return rv

    def roll_wings_level(self, x_f16):
        'get commands in mode ROLL'

        phi = x_f16[S.PHI]
        p = x_f16[S.P]

        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromWingsLevel = jnp.round(phi / (2 * jnp.pi))

        # PD Control until phi == pi * radsFromWingsLevel
        ps = -(phi - (2 * jnp.pi) * radsFromWingsLevel) * self.cfg_k_prop - p * self.cfg_k_der

        # Build commands to roll wings level
        rv = jnp.array([0.0, ps, 0.0, 0.0])

        return rv