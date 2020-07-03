import copy
import logging
import os
from functools import partial

import numpy as np
import tensorflow as tf

from algorithms.abstract_policy_update import AbstractPolicyUpdate
from algorithms.stochastic_search.trpo.conjugate_gradient import conjugate_gradients
from objective_functions.f_rosenbrock import RosenbrockTensorflow
from models.policy.abstract_policy import AbstractPolicy
from models.policy.gaussian_policy import GaussianPolicy

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('trpo')
logger.setLevel(logging.INFO)


class TRPO(AbstractPolicyUpdate):

    def __init__(self, policy: AbstractPolicy,
                 objective: callable,
                 n_samples: int,
                 ent_coef: float = 0.0,
                 entropy_bound=1.0,
                 kl_bound=0.01,
                 damping_coeff=0.0,
                 max_backtrack_iter=10,
                 backtrack_coeff=0.5,
                 cg_iterations=10
                 ):
        """

        :param policy: Policy instance
        :param objective: Objective function
        :param n_samples: number of generated samples
        :param ent_coef: entropy loss weight
        :param entropy_bound: adaptive KL penalty coefficient (not used currently)
        :param kl_bound: KL threshold
        :param damping_coeff: dampening for Hessian vector product
        :param max_backtrack_iter: number of backtracking steps for line search
        :param backtrack_coeff: exponentially scaled coefficient to reduce gradient
        :param cg_iterations: number of CG iterations, if 0, actual hessian is computed.
        """

        super().__init__(policy)

        self.objective = objective
        self.max_samples = n_samples

        self.entropy_coef = ent_coef
        self.beta = entropy_bound
        self.delta = kl_bound
        self.damping_coeff = damping_coeff

        # backtracking line search
        self.backtrack_iters = max_backtrack_iter
        self.backtrack_coeff = backtrack_coeff

        # inverse hessian computation with CG
        self.cg_iterations = cg_iterations

    @tf.function
    def sample(self, mean, std, n, context=None):
        samples = tf.squeeze(self.policy.sample(self.max_samples))  # actions
        logprobs = self.policy.log_prob(samples)

        rewards = - self.objective(samples)

        return {"x": samples, "r": rewards, "logp": logprobs}

    @tf.function
    def compute_loss(self, policy_new, sample_dict):
        """
        Compute policy gradient loss
        :return: loss value
        """

        samples, rewards, old_logprobs = sample_dict['x'], sample_dict['r'], sample_dict['logp']

        new_logprobs = policy_new.log_prob(samples)
        entropy = tf.reduce_mean(policy_new.entropy())

        # TODO check which baseline works best here, maybe use mean or something else?
        # advantages = rewards - tf.reduce_max(rewards)
        advantages = rewards - tf.reduce_mean(rewards)

        # normalize advantages
        advs = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        ratio = tf.exp(new_logprobs - tf.stop_gradient(old_logprobs))
        policy_loss = - tf.reduce_mean(ratio * advs)

        surrogate_loss = policy_loss + self.entropy_coef * entropy

        kl = policy_new.kl_divergence(self.policy, None)

        return surrogate_loss, policy_loss, entropy, kl

    # @tf.function
    def line_search(self, policy_new, sample_dict, alpha, step_dir, starting_loss):
        """
        backtracking line search to find step size for given direction
        :param alpha:
        :param step_dir:
        :param starting_loss:
        :return:
        """

        kl = tf.constant(0.0)
        policy_loss = 0.0
        new_loss = tf.constant(0.0)
        # starting_loss = tf.constant(starting_loss)
        entropy = 0.0
        success = tf.constant(False)

        for j in range(self.backtrack_iters):
            step = self.backtrack_coeff ** j
            policy_new.set_params(self.policy.params(flatten=True, trainable_vars=True) - alpha * step_dir * step,
                                  is_flattened=True, is_trainable_vars=True)
            new_loss, policy_loss, entropy, kl = self.compute_loss(policy_new, sample_dict)

            # enforce hard KL constraint and loss improvement
            if kl <= self.delta and new_loss <= starting_loss:
                logger.debug(f'Accepting new params at step {j} of line search.')
                success = True
                break

        if not success:
            logger.debug('Line search failed! Keeping old params.')
            policy_new.set_params(self.policy.params())
            new_loss, policy_loss, entropy, kl = self.compute_loss(policy_new, sample_dict)

        return new_loss, policy_loss, entropy, kl

    @tf.function
    def hessian_vector_product(self, policy_new, v):
        """
        compute hessian vector product for H = grad**2 f, compute Hx
        :param v:
        :return:
        """
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                # policies still have the same parameters here, but TF cannot compute the gradient if this is not
                kl = tf.reduce_mean(self.policy.kl_divergence(policy_new, None))

            kl_grad = tape2.gradient(kl, policy_new.trainable_variables)
            kl_grad_flat = tf.concat([tf.reshape(x, (-1,)) for x in kl_grad], axis=0)

            kl_v_sum = tf.reduce_sum(kl_grad_flat * tf.stop_gradient(v))
        kl_v_grad = tape.gradient(kl_v_sum, policy_new.trainable_variables)
        hvp = tf.concat([tf.reshape(x, (-1,)) for x in kl_v_grad], axis=0)

        if self.damping_coeff > 0:
            hvp += self.damping_coeff * v

        return hvp

    @tf.function
    def exact_step_dir(self, policy_new, loss_grad):
        """
        find step direction by computing the inverse of the Hessian
        Only possible for smaller
        :param loss_grad:
        :return:
        """
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                # policies have same parameters
                kl = self.policy.kl_divergence(policy_new, None)

            grads = tape2.gradient(kl, policy_new.trainable_variables)
            grads_flat = tf.concat([tf.reshape(x, (-1,)) for x in grads], axis=0)

        # compute hessian
        hessian = tape.gradient(grads_flat, policy_new.trainable_variables)

        # compute flattened inverse hessian
        hessian_flat = tf.concat(
            [tf.reshape(x, (-1,)) for x in [tf.linalg.inv(g) if len(tf.squeeze(g).shape) > 1 else g for g in hessian]],
            axis=0)

        # return hessian vector product
        return hessian_flat * loss_grad

    def step(self, policy_new, sample_dict={}):

        # TODO
        sample_dict = self.sample(None, None, None)

        with tf.GradientTape() as tape:
            surrogate_loss, _, _, _ = self.compute_loss(policy_new, sample_dict)

        grads = tape.gradient(surrogate_loss, policy_new.trainable_variables)
        grads_flat = tf.concat([tf.reshape(x, (-1,)) for x in grads], axis=0)

        if self.cg_iterations == 0:
            # Analytical version
            step_dir = self.exact_step_dir(policy_new, grads_flat)
        else:
            # approximate CG version
            step_dir = conjugate_gradients(partial(self.hessian_vector_product, policy_new=policy_new), grads_flat,
                                           self.cg_iterations)

        alpha = tf.sqrt(
            2 * self.delta / (tf.tensordot(step_dir, self.hessian_vector_product(policy_new, step_dir), 1) + 1e-8))

        # Find step size for given direction
        new_loss, policy_loss, entropy, kl = self.line_search(policy_new, sample_dict, alpha, step_dir, surrogate_loss)

        # Update parameters of policy
        # TODO not required for general version
        self.policy.set_params(policy_new.params())

        rew = (self.objective(policy_new.mean) - self.objective.f_opt).numpy().item()

        return {
            'loss': surrogate_loss.numpy(),
            'loss_delta': new_loss.numpy() - surrogate_loss.numpy(),
            'adjusted_loss': new_loss.numpy(),
            'entropy': entropy.numpy(),
            'kl': kl.numpy(),
            'reward': rew,
            # 'reward_delta': np.asscalar((self.objective(policy.mean) - self.objective.f_opt).numpy()) - rew,
        }

    def learn(self, num_iter, **kwargs):

        policy_new = copy.deepcopy(self.policy)

        for i in range(1, num_iter + 1):
            # sample based on current policy
            # sample_dict = self.sample()

            # find update direction
            d = self.step(policy_new, None)

            if i % 10 == 0:
                logger.info("Iteration {}".format(i) + "-" * 100)
                logger.info(d)

            if d['reward'] <= 1e-15:
                logger.info("Iteration {}".format(i) + "-" * 100)
                logger.info("Reached minimum objective")
                logger.info(d)
                break


if __name__ == "__main__":
    np.random.seed(20)
    tf.random.set_seed(20)

    num_iter = 10000

    # specify dimensions
    input_dim = 10
    # target_dim = 1

    # generate objective
    objective = RosenbrockTensorflow(input_dim)
    # objective = Quad(input_dim)

    # init for Gaussian policy
    mean_init = np.random.randn(1, input_dim)
    cov_init = 1 * np.eye(input_dim)

    policy = GaussianPolicy(mean_init, cov_init[None, ...])

    # cg_iterations=0 == exact hessian inverse
    algo = TRPO(objective=objective, policy=policy, n_samples=500, ent_coef=0.0, kl_bound=0.5,
                max_backtrack_iter=10, backtrack_coeff=0.9, cg_iterations=0, damping_coeff=0.0)
    algo.learn(num_iter=num_iter)
