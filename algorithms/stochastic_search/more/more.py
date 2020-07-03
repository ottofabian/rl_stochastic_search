import copy
import logging
from collections import defaultdict, deque

import nlopt
# import autograd.numpy as np
import numpy as np
import tensorflow as tf
from autograd import grad
from matplotlib import pyplot as plt

from algorithms.abstract_policy_update import AbstractPolicyUpdate
from algorithms.stochastic_search.more.reward_model import QuadModelWhitening
from models.policy.abstract_policy import AbstractPolicy
from models.policy.gaussian_policy import GaussianPolicyNumpy
from objective_functions.f_rosenbrock import RosenbrockRaw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('more')


class MORE(AbstractPolicyUpdate):
    def __init__(self, objective: callable, policy: AbstractPolicy, n_samples: int,
                 epsilon: float, optim_runs: int = 10, beta: float = 0.1, max_samples: int = 150,
                 do_whiten: bool = False, reward_weighting: str = None, normalize_input: bool = False,
                 normalize_output: bool = False, enforce_equality: bool = False):

        super().__init__(policy)

        self.optim_runs = optim_runs
        self.objective = objective
        self.dim = objective.dim
        self.num_resample = n_samples
        self.max_samples = max_samples

        # self.reward_model = QuadModel(self.input_dim)
        self.reward_model = QuadModelWhitening(self.dim, max_samples)

        self.epsilon = epsilon
        self.beta = beta

        # saved variables
        # self.inv_Sigma_q = np.linalg.inv(self.q.cov)
        self._chol_Sigma_q = np.linalg.cholesky(self.policy.cov)
        self._chol_Q = np.linalg.inv(self._chol_Sigma_q)  # cholesky of precision of q
        self._Q = self._chol_Q.T @ self._chol_Q  # precision of q
        self._nat_q = self._Q @ self.policy.mean[:, None]  # canonical mean parameter of q
        self._detTerm_q = None
        self._std_Q = None

        # keep only max_samples most recent samples
        self._sample_q = deque(maxlen=max_samples)
        self._reward_q = deque(maxlen=max_samples)

        self.initial_rewards = False

        # options for quadratic model
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.do_whiten = do_whiten
        self.reward_weighting = reward_weighting

        self.opt = self._get_optimizer(enforce_equality)

        self.mb_loss_vals = defaultdict(list)
        self._global_iters = 1

    def sample(self, mean, std, n, context=None, **kwargs):
        # set entropy constraint
        if self.policy.entropy() < -300:
            self.beta = 0.01

        # initially sample max_samples thetas
        if self._sample_q:
            samples = self.policy.sample(self.num_resample)
            samples = np.atleast_2d(samples)
        else:
            samples = self.policy.sample(self.max_samples)
            self._sample_q.extend(samples)

        self._sample_q.extend(samples)
        self._reward_q.extend(-self.objective(samples))

    def learn(self, num_iter, **kwargs):
        opt = np.inf

        policy_new = copy.deepcopy(self.policy)

        while True or (opt > self.objective.f_opt + 1e-4 and self._global_iters <= num_iter):
            # print("----------iter {} -----------".format(i))

            # generate new samples
            # self.sample(None)

            d = self.step(policy_new, None)
            logger.info("Iteration {}".format(self._global_iters) + "-" * 100)
            logger.info(d)

            opt = self.objective(self.policy.mean)

            self._global_iters += 1

    def step(self, policy_new, sample_dict={}):

        self.sample(None, None, None)
        # sample, reward = sample_dict['x'], sample_dict['r']
        sample = np.stack(self._sample_q)
        reward = np.vstack(self._reward_q)

        # set model params for reuse
        self.cache_model_params()

        # fit quadratic surrogate model
        try:
            self.reward_model.fit_model(sample, reward, normalize_input=self.normalize_input,
                                        normalize_output=self.normalize_output, theta_mean=self.policy.mean[:, None].T,
                                        theta_inv_std=self._std_Q, reward_weighting=self.reward_weighting,
                                        do_whiten=self.do_whiten)
        except ValueError:
            logging.debug("Model could not be fitted.")
            # return False

        success_kl = False
        success_entropy = False

        eta_0 = 1.
        omega_0 = 1.

        # multiple runs to ensure appropriate lagrange multiplier are found
        for i in range(self.optim_runs):
            try:

                eta, omega = self.opt.optimize(np.hstack([eta_0, omega_0]))
                # kl, entropy = self.constraints(eta, omega)
                opt_val = self.opt.last_optimum_value()
                result_code = self.opt.last_optimize_result()

                if result_code in (1, 2, 3, 4) and ~np.isinf(opt_val):
                    # validation of KL and Entropy constraints
                    kl, entropy_diff, entropy = self.constraints(eta, omega)

                    success_kl = kl <= 1.1 * self.epsilon
                    success_entropy = entropy_diff <= 1.1 * self.beta

                if success_kl and success_entropy:
                    break

                eta_0 *= 2.

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.debug(e)
                eta_0 *= 2.

        if success_kl and success_entropy:
            self.update_policy(policy_new, eta=eta, omega=omega)
            kl = policy_new.kl_divergence(self.policy, None).item()
            # set search distribution to new parameters
            self.policy.set_params(policy_new.params())

        else:
            logger.debug(
                "KL or entropy bound not met. No suitable lagrange multipliers found. Not updating parameters.")
            kl = -1
            entropy_diff = -1
            eta = -1
            omega = -1

        out = {'epsilon': self.epsilon,
               'beta': self.beta,
               'eta': eta,
               'omega': omega,
               'kl': kl,
               'entropy_diff': entropy_diff,
               'entropy': policy_new.entropy(),
               'reward': (self.objective(policy_new.mean) - self.objective.f_opt).item(),
               }

        out = {k: (-1 if v == 'NaN' or np.isnan(v) else v) for k, v in out.items()}

        [self.mb_loss_vals[k].append(v) for k, v in out.items()]
        self.mb_loss_vals['mean'].append(policy_new.mean)
        self.mb_loss_vals['cov'].append(policy_new.cov)

        return out

    def update_policy(self, policy, eta, omega):

        # quadratic, linear and bias parameter
        R, r, r_0 = self.reward_model.get_model_params()

        p = (eta * self._nat_q + r) / (eta + omega)  # r parameter of p
        P = (eta * self._Q + R) / (eta + omega)  # precision of p

        chol_P = np.linalg.cholesky(P)
        chol_Sigma_p = np.linalg.inv(chol_P)
        Sigma_p = chol_Sigma_p.T @ chol_Sigma_p
        mu_p = np.linalg.solve(P, p).flatten()

        policy.set_params([mu_p, Sigma_p])

    def cache_model_params(self):
        mu_q = self.policy.mean[:, None]
        Sigma_q = self.policy.cov
        self._chol_Sigma_q = np.linalg.cholesky(Sigma_q)

        # cholesky of precision of q
        self._chol_Q = np.linalg.inv(self._chol_Sigma_q)

        # precision of q
        self._Q = self._chol_Q.T @ self._chol_Q

        self._detTerm_q = 2 * np.sum(np.log(np.diag(self._chol_Sigma_q)))

        lambdas, vecs = np.linalg.eigh(Sigma_q)
        lambdas = np.real(lambdas)
        vecs = np.real(vecs)
        sqrt_l = np.sqrt(lambdas)

        self._std_Q = vecs @ np.diag(sqrt_l ** -1) @ vecs.T
        # canonical mean parameter of q
        self._nat_q = self._Q @ mu_q

    def dual_function(self, x):
        """
        Analytic computation of the dual
        :param x: eta and omega to optimize dual
        :return:
        """
        # x = (eta, omega)

        if np.any(np.isnan(x)):
            return np.atleast_1d(np.inf)

        eta = x[0]
        omega = x[1]

        # TODO duplicate in constraints maybe change this
        R, r, r_0 = self.reward_model.get_model_params()
        mu_q = self.policy.mean[:, None]
        p = (eta * self._nat_q + r) / (eta + omega)  # canonical mean parameter of p
        P = (eta * self._Q + R) / (eta + omega)  # precision of p

        try:
            chol_P = np.linalg.cholesky(P)
            mu_p = np.linalg.solve(P, p)  # Sigma_p @ p

            chol_Sigma_p = np.linalg.inv(chol_P)
            Sigma_p = chol_Sigma_p.T @ chol_Sigma_p

            # compute log(det(Sigma_p))
            detTerm_p = 2 * np.sum(np.log(np.diag(chol_Sigma_p)))

            g = eta * self.epsilon + self.beta * omega \
                - 1 / 2 * mu_p.T @ R @ mu_p + mu_p.T @ r \
                - 1 / 2 * np.trace(R @ Sigma_p) \
                - 1 / 2 * eta * (mu_q - mu_p).T @ self._Q @ (mu_q - mu_p) \
                - 1 / 2 * eta * (np.trace(self._Q @ Sigma_p) - self.dim + self._detTerm_q - detTerm_p) \
                - 1 / 2 * omega * (self._detTerm_q - detTerm_p)

            return g.flatten()
        except np.linalg.LinAlgError:
            return np.atleast_1d(np.inf)

    def grad_dual(self, x):

        if np.any(np.isnan(x)):
            logger.debug("Nan input to gradient function")
            return np.zeros(2)

        eta = x[0]
        omega = x[1]

        R, r, r_0 = self.reward_model.get_model_params()
        mu_q = self.policy.mean[:, None]

        p = (eta * self._nat_q + r) / (eta + omega)  # canonical mean parameter of p
        P = (eta * self._Q + R) / (eta + omega)  # precision of p

        try:
            chol_P = np.linalg.cholesky(P)
            mu_p = np.linalg.solve(P, p)  # Sigma_p @ p

            # compute log(det(Sigma_p))
            detTerm_p = -2 * np.sum(np.log(np.diag(chol_P)))

            kl = 0.5 * ((mu_q - mu_p).T @ self._Q @ (mu_q - mu_p)
                        + self._detTerm_q
                        - detTerm_p
                        + np.trace(np.linalg.solve(P, self._Q))
                        - self.dim)

            d_g_d_eta = self.epsilon - kl[0]
            d_g_d_omega = self.beta - 0.5 * (self._detTerm_q - detTerm_p)

        except:
            d_g_d_eta = 0.
            d_g_d_omega = 0.

        return np.hstack((d_g_d_eta, d_g_d_omega))

    def constraints(self, eta, omega):
        """
        Compute
        :param eta:
        :param omega:
        :return:
        """

        # TODO duplicate in dual maybe change this
        R, r, r_0 = self.reward_model.get_model_params()
        mu_q = self.policy.mean[:, None]
        p = (eta * self._nat_q + r) / (eta + omega)  # canonical mean parameter of p
        P = (eta * self._Q + R) / (eta + omega)  # precision of p

        try:
            chol_P = np.linalg.cholesky(P)
            mu_p = np.linalg.solve(P, p)

            # compute log(det(Sigma_p))
            detTerm_p = -2. * np.sum(np.log(np.diag(chol_P)))  # + self.dim_theta * np.log(2 * np.pi)

            kl = 0.5 * ((mu_q - mu_p).T @ self._Q @ (mu_q - mu_p)
                        + self._detTerm_q
                        - detTerm_p
                        + np.trace(np.linalg.solve(P, self._Q))
                        - self.dim)

            entropy = 0.5 * (self.dim + detTerm_p)
            entropy_diff = 0.5 * (self._detTerm_q - detTerm_p)
        except np.linalg.LinAlgError:
            kl = 100.
            entropy_diff = 100.

        return kl, entropy_diff, entropy

    def _get_optimizer(self, enforce_equality):
        opt = nlopt.opt(nlopt.LD_LBFGS, 2)

        # enforce equality on the entropy loss constraint?
        if enforce_equality:
            opt.set_lower_bounds((1e-20, -1e6))
            opt.set_upper_bounds((1e20, 1e6))
        else:
            opt.set_lower_bounds((1e-20, 1e-20))
            opt.set_upper_bounds((1e20, 1e20))

        opt.set_ftol_abs(1e-12)
        opt.set_xtol_abs(1e-12)
        opt.set_maxeval(50000)
        opt.set_maxtime(5 * 60 * 60)

        def opt_func(x, g):
            val = self.dual_function(x)
            if g.size > 0:
                g[:] = self.grad_dual(x)
            return float(val.flatten())

        opt.set_min_objective(opt_func)
        return opt


if __name__ == "__main__":
    np.random.seed(20)

    num_iter = 5000
    dtype = tf.float64

    # specify dimensions
    input_dim = 2
    # target_dim = 1
    # mean_init = numpy.random.randn(1, input_dim)
    mean_init = np.array([[-1., 1.]])
    cov_init = 0.5 * np.eye(input_dim)

    # generate objective
    # objective = Rosenbrock(input_dim)
    objective = RosenbrockRaw(input_dim)
    # objective = Ellipsoid(input_dim)

    policy = GaussianPolicyNumpy(np.squeeze(mean_init), cov_init)

    algo = MORE(objective=objective, policy=policy, n_samples=25, epsilon=0.75, beta=1.55,
                max_samples=25, reward_weighting="exp", do_whiten=True, normalize_input=True, normalize_output=True,
                enforce_equality=False)

    algo.learn(num_iter)
