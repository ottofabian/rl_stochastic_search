### Policy Updates for stochastic search

This repository contains several implementations of popular algorithms. 
However, they are written for stochastic search to show there true performance on optimization tasks. 
It is not the greatest code, but tries to put everything you need in one file to make it better understandable. 

## Installation
See the requirements.txt, but be warned it might install some other things, because I just copied my testing env. 

Manually, just go with, that should suffice:
- tensorflow > 2.0
- tensorflow_probability
- numpy 
- hips/autograd

## Run some algorithms

Main methods of each algortihm can be found in their respective folder. 

Anyway, one example for running REPS.

```python
import numpy as np
from objective_functions.rosenbrock import Rosenbrock
from models.policy.gaussian_policy import GaussianPolicyNumpy
from algorithms.stochastic_search.reps.reps import EpisodicREPS

np.random.seed(20)

num_iter = 1000

# specify dimensions
input_dim = 10
# target_dim = 1

# generate objective
objective = Rosenbrock(input_dim)

# init for Gaussian
mean_init = np.random.randn(input_dim)
cov_init = 1 * np.eye(input_dim)

policy = GaussianPolicyNumpy(mean_init, cov_init)

algo = EpisodicREPS(objective=objective, policy=policy, max_samples=1000, n_samples=1000, epsilon=0.05)
# start = time.time()
algo.learn(num_iter=num_iter)
# print(time.time() - start)
```

