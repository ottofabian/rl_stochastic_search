import copy
import logging
import os
import time
from collections import defaultdict
from typing import Union

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from algorithms.abstract_policy_update import AbstractPolicyUpdate
from objective_functions.f_rosenbrock import RosenbrockRawTensorflow
from models.policy.abstract_policy import AbstractPolicy
from models.policy.gaussian_policy import GaussianPolicy
from utils.tf_utils import get_optimizer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('repa')


class REPA(AbstractPolicyUpdate):

    def __init__(self, policy: AbstractPolicy, objective: callable, n_samples: int,
                 learning_rate: Union[float, callable] = 2.5e-4, entropy_coef=0.):
        """
        Policy update via Reparametrization trick
        :param policy: Policy Instance
        :param objective: Objective to optimize
        :param n_samples: max number of samples kept
        :param learning_rate: learning rate
        """

        super().__init__(policy)

        self.objective = objective
        self.n_samples = n_samples

        self.entropy_coef = 0.0

        # learning rate value or schedule
        self.learning_rate_handle = learning_rate

        # TODO Fix this for actual schedules and not constant lrs
        self.optimizer = get_optimizer("adam", learning_rate=learning_rate, epsilon=1e-5)

        self.mb_loss_vals = defaultdict(list)
        self._global_iters = 1

    @tf.function
    def sample(self, mean, std, n, context=None):
        pass

    @tf.function
    def compute_loss(self, samples, policy):
        with tf.GradientTape() as tape:
            samples = samples if samples else policy.sample(self.n_samples)
            loss = tf.reduce_mean(self.objective(samples))

        grads = tape.gradient(loss, policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        kl = self.policy.kl_divergence(policy, None)
        entropy = policy.entropy()

        loss += self.entropy_coef * entropy

        return loss, kl, entropy

    def step(self, policy_new, sample_dict={}):

        t = time.time()
        loss, kl, entropy = self.compute_loss(sample_dict['x'], policy_new)

        kl = kl.numpy().item()
        kl = -1 if np.isnan(kl) or np.isinf(kl) else kl

        self.policy.set_params(policy_new.params())

        out = {'loss': loss.numpy().item(),
               'kl': kl,
               # 'entropy_diff': entropy_diff,
               'entropy': entropy.numpy().item(),
               'reward': (self.objective(policy_new.mean) - self.objective.f_opt).numpy().item(),
               }

        [self.mb_loss_vals[k].append(v) for k, v in out.items()]
        self.mb_loss_vals['mean'].append(policy_new.mean)
        self.mb_loss_vals['cov'].append(policy_new.cov)

        return out

    def learn(self, num_iter, **kwargs):
        # Transform to callable if needed
        # self.learning_rate_handle = get_schedule_fn(self.learning_rate_handle)

        policy = copy.deepcopy(self.policy)

        for i in range(1, num_iter + 1):

            if i % 50 == 1:
                sample_dict = {
                    'x': None  # tf.random.normal((self.n_samples, self.objective.dim))
                }
            # t = time.time()
            d = self.step(policy, sample_dict)
            d.update({'lr': (
                self.optimizer.lr(i) if isinstance(
                    self.optimizer.lr, tf.optimizers.schedules.LearningRateSchedule) else self.optimizer.lr).numpy()})

            self._global_iters += 1
            if i % 10 == 0:
                # print(time.time() - t)
                logger.info("Iteration {}".format(i) + "-" * 100)
                logger.info(d)


if __name__ == "__main__":
    np.random.seed(20)
    tf.random.set_seed(1)
    dtype = tf.float32

    num_iter = 10000

    # specify dimensions
    input_dim = 30
    # target_dim = 1

    # generate objective
    # objective = RosenbrockTensorflow(input_dim)
    objective = RosenbrockRawTensorflow(input_dim)

    # init for Gaussian
    mean_init = np.random.randn(input_dim)
    # mean_init = np.array([-1., 1.])
    cov_init = 1 * np.eye(input_dim)

    policy = GaussianPolicy(mean_init, cov_init)

    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=100, decay_rate=0.96)
    learning_rate = 1e-3

    algo = REPA(objective=objective, policy=policy, n_samples=5000, learning_rate=learning_rate)
    algo.learn(num_iter=num_iter)
