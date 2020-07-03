import copy
import logging
import os
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as k

from algorithms.abstract_policy_update import AbstractPolicyUpdate
from objective_functions.f_rosenbrock import RosenbrockTensorflow
from models.policy.abstract_policy import AbstractPolicy
from models.policy.gaussian_policy import GaussianPolicy
from utils.tf_utils import get_optimizer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ppo')


class PPO(AbstractPolicyUpdate):

    def __init__(self, policy: AbstractPolicy, objective: callable, n_samples: int, ent_coef: float = 0.00,
                 learning_rate: Union[float, callable] = 2.5e-4, max_grad_norm: float = 0.5,
                 steps_per_sample: int = 4, gradient_clipping: Union[float, callable] = 0.2,
                 reward_clipping: float = None):
        """

        :param policy: Policy Instance
        :param objective: Objective to optimize
        :param n_samples: max number of samples kept
        :param ent_coef: entropy loss scaling
        :param learning_rate: learning rate or learning rate schedule
        :param max_grad_norm: max global l2 norm
        :param steps_per_sample: number of optimization steps per sample
        :param gradient_clipping: max gradient value
        """

        super().__init__(policy)

        self.objective = objective
        self.max_samples = n_samples

        self.optimizer = None

        # learning rate value or schedule
        self.learning_rate = learning_rate
        # self.learning_rate_handle = get_schedule_fn(learning_rate)
        self.gradient_clipping = gradient_clipping
        self.max_grad_norm = max_grad_norm
        self.reward_clipping = reward_clipping

        # TODO Fix this for actual schedules and not constant lrs
        self.optimizer = get_optimizer("adam", learning_rate=self.learning_rate, clipnorm=self.max_grad_norm,
                                       epsilon=1e-5)

        self.entropy_coef = ent_coef

        # number of updates
        self.steps_per_sample = steps_per_sample

        # containers
        self.samples = None

        self.actions = None
        self.logprobs = None

        self.advantages = None
        self.rewards = None

    @tf.function
    def sample(self, mean, std, n, context=None):
        samples = self.policy.sample(self.max_samples)  # actions
        logprobs = self.policy.log_prob(samples)

        rewards = - self.objective(samples)

        return {"x": samples, "r": rewards, "logp": logprobs}

    @tf.function
    def compute_loss(self, policy_new, samples, rewards, logprobs):
        new_logprobs = policy_new.log_prob(samples)
        entropy = tf.reduce_mean(policy_new.entropy())

        if self.reward_clipping:
            rewards = tf.clip_by_value(rewards, -self.reward_clipping, self.reward_clipping)

        # TODO check which baseline works best here, maybe use mean or something else?
        # compute advantages
        # advantages = rewards - tf.reduce_max(rewards)
        advantages = rewards - tf.reduce_mean(rewards)

        advs = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-16)
        # advs = self.advantages

        # importance ratio and surrogate objectives
        ratio = tf.exp(new_logprobs - tf.stop_gradient(logprobs))
        surrogate1 = ratio * advs
        surrogate2 = tf.clip_by_value(ratio, 1.0 - self.gradient_clipping, 1.0 + self.gradient_clipping) * advs

        # compute clipped objective
        policy_loss = - tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        loss = policy_loss - self.entropy_coef * entropy

        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), self.gradient_clipping), tf.float32))

        kl = policy_new.kl_divergence(self.policy, None)

        return loss, policy_loss, entropy, kl, clipfrac

    def step(self, policy_new, sample_dict={}):

        sample_dict = self.sample(None, None, None)

        samples, rewards, old_logprobs = sample_dict['x'], sample_dict['r'], sample_dict['logp']

        mb_loss_vals = []

        # make n update steps per sample
        for epoch_num in range(self.steps_per_sample):
            with tf.GradientTape() as tape:
                # passing samples and advantages because of tf.function
                surrogate_loss, policy_loss, entropy, kl, clipfrac = self.compute_loss(policy_new, samples, rewards,
                                                                                       old_logprobs)

            grads = tape.gradient(surrogate_loss, policy_new.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, policy_new.trainable_variables))

            mb_loss_vals.append([surrogate_loss, policy_loss, entropy, kl, clipfrac])

        self.policy.set_params(policy_new.params())

        scores = np.mean(mb_loss_vals, axis=0)

        return {
            'loss': scores[0],
            'policy_loss': scores[1],
            'entropy_loss': scores[2],
            'entropy': policy_new.entropy().numpy(),
            'kl': scores[3],
            'clip_factor': scores[4],
            'reward': (self.objective(policy_new.mean) - self.objective.f_opt).numpy().item(),
        }

    def learn(self, num_iter, **kwargs):

        policy_new = copy.deepcopy(self.policy)

        for i in range(1, num_iter + 1):

            # sample based on current policy
            # sample_dict = self.sample()

            d = self.step(policy_new, None)
            d.update({'lr': (
                self.optimizer.lr(
                    i * self.steps_per_sample) if isinstance(
                    self.optimizer.lr, tf.optimizers.schedules.LearningRateSchedule) else self.optimizer.lr).numpy()
                      })

            if i % 10 == 0:
                logger.info("Iteration {}".format(i) + "-" * 100)
                logger.info(d)


if __name__ == "__main__":
    np.random.seed(20)

    num_iter = 10000

    # specify dimensions
    input_dim = 10
    # target_dim = 1

    # generate objective
    objective = RosenbrockTensorflow(input_dim, int_opt=(0, 0))
    # objective = Quad(input_dim)

    # init for Gaussian
    mean_init = np.random.randn(input_dim)
    cov_init = 1 * np.eye(input_dim)

    policy = GaussianPolicy(mean_init, cov_init)

    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=500, decay_rate=0.96)
    learning_rate = 1e-3

    algo = PPO(objective=objective, policy=policy, n_samples=2000, learning_rate=learning_rate,
               gradient_clipping=0.2, max_grad_norm=.5, steps_per_sample=4, reward_clipping=None)
    algo.learn(num_iter=num_iter)
