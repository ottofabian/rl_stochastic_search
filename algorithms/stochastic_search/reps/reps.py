import copy
import logging
import os
from collections import deque

import autograd.numpy as np
from autograd import grad
from scipy import optimize

from algorithms.abstract_policy_update import AbstractPolicyUpdate
from models.policy.gaussian_policy import GaussianPolicyNumpy
from objective_functions.f_base import BaseObjective
from objective_functions.f_rosenbrock import Rosenbrock

os.environ['OPENBLAS_NUM_THREADS'] = '4'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('reps')


class EpisodicREPS(AbstractPolicyUpdate):

    def __init__(self, objective: BaseObjective, policy: GaussianPolicyNumpy, max_samples=100, n_samples=10,
                 epsilon=0.4):
        """

        Args:
            objective: target function
            policy: policy instance
            max_samples: number of samples in first iteration
            n_samples: number of resamples in following interations
            epsilon: kl bound
        """
        super().__init__(policy)
        self.objective = objective
        self.input_dim = self.objective.dim

        self.max_samples = max_samples
        self.num_resample = n_samples

        self.epsilon = epsilon

        # keep only max_samples most recent samples
        self._sample_q = deque(maxlen=max_samples)
        self._reward_q = deque(maxlen=max_samples)

    def sample(self, mean, std, n, context=None, **kwargs):
        """
        Generate new samples and add them to Q
        Args:

            n: number of samples to generate from current policy

        Returns: None

        """
        if self._sample_q:
            samples = self.policy.sample(self.num_resample)
        else:
            samples = self.policy.sample(self.max_samples)
        self._sample_q.extend(samples)

        self._reward_q.extend(- self.objective(samples))

    def dual_function(self, eta, r):
        """
        Compute dual function
        Args:
            eta: lagrangian multiplier
            r: rewards

        Returns:

        """
        adv = r - np.max(r)
        g = eta * self.epsilon + np.max(r) + eta * np.log(np.mean(np.exp(adv / eta), axis=0))
        return g

    def update_policy(self, policy, eta, x, r):
        """
        Update given policy based on optimal lagranian and samples.
        Args:
            policy: policy instance to update
            eta: lagrangian multiplier (found by optimization)
            x: samples
            r: rewards

        Returns:

        """

        adv = r - np.max(r)
        w = np.exp(adv / eta)

        mean = np.sum(x * w[:, None], axis=0) / np.sum(w, axis=0)
        diff = x - mean
        cov = np.einsum('nk,n,nh->kh', diff, w, diff) / np.sum(w, axis=0)

        policy.set_params([mean, cov])

    def step(self, policy_new, sample_dict={}):
        """
        compute one step of the update
        Args:
            policy_new: policy instance to make the step for
            sample_dict: dict with samples (currently not used)

        Returns:

        """

        self.sample(None, None, None)
        reward = np.vstack(self._reward_q)

        res = optimize.minimize(self.dual_function, 1.0,
                                method='SLSQP',
                                # method='L-BFGS-B',
                                jac=grad(self.dual_function),
                                args=(reward,),
                                bounds=((1e-8, 1e8),))
        eta = res.x

        kl_samples = self.kl_divergence(eta, reward)

        r = np.asarray(self._reward_q)
        x = np.stack(self._sample_q)
        self.update_policy(policy_new, eta=eta, x=x, r=r)

        # maintain old policy to sample from for later use
        self.policy.set_params(policy_new.params())

        return {'epsilon': self.epsilon,
                #     'beta': self.beta,
                'eta': eta.item(),
                'kl': kl_samples.item(),
                # 'entropy_diff': entropy_diff,
                'entropy': self.policy.entropy(),
                'reward': (self.objective(policy_new.mean) - self.objective.f_opt).item(),
                }

    def kl_divergence(self, eta, r):
        """
        Compute sample based kl
        Args:
            eta: lagrangian multiplier
            r: rewards

        Returns:

        """
        adv = r - np.max(r)

        # compute normalized weight
        w = np.exp(adv / eta)
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)

        # compute weighted KL
        return np.mean(w * np.log(w), axis=0)

    def learn(self, num_iter, **kwargs):
        """
        Fully execute REPS for n steps
        Args:
            num_iter: number of iteration steps
            **kwargs:

        Returns:

        """

        policy_new = copy.deepcopy(self.policy)

        for i in range(num_iter):

            d = self.step(policy_new, None)
            if i % 10 == 0:
                logger.info("Iteration {}".format(i) + "-" * 100)
                logger.info(d)

            # end early in case cov collapses
            if np.linalg.det(self.policy.cov) < 1e-150:
                logger.info("Minimal determinant.")
                print(np.linalg.det(self.policy.cov))
                break


if __name__ == "__main__":
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
