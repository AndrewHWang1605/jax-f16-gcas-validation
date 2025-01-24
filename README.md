<div align="center">

# Jax F16 GCAS


</div>

This repository contains a simple simulation of the F16 Ground Collision Avoidance System (GCAS). The GCAS autopilot is based on [this paper](https://stanleybak.com/papers/heidlauf2018arch.pdf) by Heidlauf et al. and the JAX implementation of the F-16 dynamics from [this package](https://github.com/MIT-REALM/jax-f16).

Thanks to JAX, the the autopilot and the dynamics are differentiable. Note that some minor changes were made to the original python implementation of the GCAS autopilot to make it compatible with JAX. Notably, the GCAS does not depend on the current time, but only the current state of the aircraft.

Check out [this video](https://www.youtube.com/watch?v=WkZGL7RQBVw) of the real GCAS system saving a pilot's life.



## Installation

First, clone the repository:

```bash
git clone git@github.com:hdelecki/jax-f16-gcas.git
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
See the [example](simulate.py) for a simple example of simulating the GCAS.



