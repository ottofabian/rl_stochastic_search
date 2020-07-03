import logging
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal
from tensorflow import linalg as tfl

from models.policy.abstract_policy import AbstractPolicy

tfd = tfp.distributions

logger = logging.getLogger('gaussian_policy')


class GaussianPolicy(AbstractPolicy):

    def __init__(self, init_mean, init_cov, weight=None, dtype=tf.float32):
        super().__init__()

        self.dtype = dtype

        self._loc = tf.Variable(init_mean, dtype=dtype, name="loc")
        # self.scale_tril = tf.Variable(np.linalg.cholesky(init_cov), dtype=dtype, name="scale")
        self._weight = tf.Variable(weight, dtype=self.dtype) if weight is not None else None

        self._scale_tril = \
            tfp.util.TransformedVariable(init_cov, bijector=tfp.bijectors.FillScaleTriL(
                diag_bijector=tfp.bijectors.Exp(),
                diag_shift=tf.constant(1e-16, dtype=self.dtype),
            ), name="scale", dtype=self.dtype)

        self._model = tfp.distributions.MultivariateNormalTriL(loc=self._loc,
                                                               scale_tril=self._scale_tril,
                                                               name="gaussian_policy_dist")

    @property
    def distribution(self):
        if self.weight is not None:
            mean, chol = self._model.loc, self._model.scale_tril
            return tfd.MultivariateNormalTriL(mean, chol * self.weight)
        else:
            return self._model

    @distribution.setter
    def distribution(self, x):
        self._model = x

    @property
    def trainable_variables(self) -> list:
        # return [self.loc, self.weight]
        vars = self._model.trainable_variables

        if self.weight is not None:
            vars = vars + (self._weight,)

        return list(vars)

    def params(self, flatten: bool = False, trainable_vars: bool = False):

        cov = self.cov if not trainable_vars else tfp.math.fill_triangular_inverse(self.cov)

        if flatten:
            p = tf.concat([tf.reshape(self.mean, [-1]), tf.reshape(cov, [-1])], axis=0)
        else:
            p = [self.mean, cov]

        if self.weight is not None:
            p = p + [self.weight]

        return p

    def set_params(self, params: list, is_flattened: bool = False, is_trainable_vars: bool = False):

        if is_flattened:
            # TODO this does not support weighted covs now
            mean = tf.reshape(params[:self.mean.shape[1]], self.mean.shape)
            cov = params[self.mean.shape[1]:] if not is_trainable_vars else tfp.math.fill_triangular(
                params[self.mean.shape[1]:])
            cov = tf.reshape(cov, self.cov.shape)

            params = [mean, cov]

        self.mean = params[0]
        self.cov = tfp.math.fill_triangular(params[1]) if is_trainable_vars and not is_flattened else params[1]

        if self.weight is not None:
            self.weight = params[2]

    def predict_dist(self, context=None, training=True):
        return self.distribution

    def density(self, x, context=None):
        d = self.predict_dist(context)
        return d.prob(x)

    def log_prob(self, x, context=None):
        d = self.predict_dist(context)
        return d.log_prob(x)

    def sample(self, num_samples, context=None):
        d = self.predict_dist(context)
        sample = d.sample(num_samples)
        return sample

    @property
    def mean(self):
        d = self.predict_dist()
        return d.mean()

    @mean.setter
    def mean(self, x):
        self._loc.assign(tf.cast(x, self.dtype))

    @property
    def cov(self):
        c = self._scale_tril if self.weight is None else self._scale_tril * self.weight
        return tf.cast(c, self.dtype)

    @cov.setter
    def cov(self, x):
        if self.weight is not None:
            x /= self.weight

        self._scale_tril.assign(tf.cast(x, self.dtype))

    def output_shape(self):
        return self.distribution.event_shape

    def input_shape(self):
        return None

    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def weight(self):
        return tf.abs(self._weight) if self._weight is not None else None

    @weight.setter
    def weight(self, x):
        self._weight.assign(tf.cast(x, self.dtype))

    def entropy(self, context=None):
        d = self.predict_dist(context)
        try:
            return d.entropy()
        except Exception as e:
            logger.warning(e)
            return tf.constant(-1, dtype=self.dtype)

    def kl_divergence(self, other, context=None):

        d = self.predict_dist(context)
        other_dist = other.distribution if isinstance(other, AbstractPolicy) else other

        try:
            return tfp.distributions.kl_divergence(d, other_dist)
        except Exception as e:
            # Handle case when KL cannot be computed
            logger.warning(e)
            return tf.constant(-1, dtype=self.dtype)


class GaussianPolicySimplified(GaussianPolicy):

    def __init__(self, init_mean, init_cov, weight=None, dtype=tf.float32):
        super().__init__(init_mean, init_cov, weight, dtype)
        self.dtype = dtype

    def __call__(self, context=None, training=True, **kwargs):
        m, chol = self.distribution
        return tf.tile(m, [context.shape[0], 1]), tf.tile(chol, [context.shape[0], 1, 1])

    @property
    def distribution(self):
        chol = self._model.scale_tril if self.weight is None else self._model.scale_tril * self.weight
        return self._model.loc, chol

    def predict_dist(self, context=None, training=True):
        return self(context, training)

    @property
    def mean(self):
        return self._model.loc

    @mean.setter
    def mean(self, x):
        self._loc.assign(tf.cast(x, self.dtype))


class GaussianPolicyNumpy(AbstractPolicy):

    def __init__(self, init_mean, init_cov):
        super().__init__()

        # self.loc = init_mean
        # self.scale = init_cov

        self._model = multivariate_normal(mean=init_mean, cov=init_cov)

    @property
    def distribution(self):
        return self._model

    @distribution.setter
    def distribution(self, x):
        self._model = x

    def params(self):
        return [self.mean, self.cov]

    def set_params(self, params: list):
        self.mean = params[0]
        self.cov = params[1]

    @property
    def trainable_variables(self) -> list:
        return []

    def density(self, x, context=None):
        return self.distribution.pdf(x)

    def log_prob(self, x, context=None):
        return self.distribution.logpdf(x)

    def sample(self, num_samples, context=None):
        return self.distribution.rvs(num_samples)

    def entropy(self, x=None):
        return self.distribution.entropy()

    def predict_dist(self, context=None, training=True):
        return tfp.distributions.MultivariateNormalFullCovariance(self.mean, self.cov)

    @property
    def mean(self):
        return self.distribution.mean

    @mean.setter
    def mean(self, x):
        if x.shape == self.distribution.mean.shape:
            self.distribution.mean = x.astype(np.float64)
        else:
            raise ValueError(
                f"Shapes do not match current mean has shape {self.distribution.mean.shape} "
                f"and provided mean has shape {x.shape}")

    @property
    def cov(self):
        return self.distribution.cov

    @cov.setter
    def cov(self, x):
        if x.shape == self.distribution.cov.shape:
            self.distribution.cov = x.astype(np.float64)
        else:
            raise ValueError(
                f"Shapes do not match current covariance has shape {self.distribution.cov.shape} "
                f"and provided mean has shape {x.shape}")

    def kl_divergence(self, other, x=None):
        """
        Compute KL between two Gaussians.
        :param other:
        :return:

        Args:
            x:
        """
        chol_cov_self = np.linalg.cholesky(self.cov)
        det_term_self = 2 * np.sum(np.log(np.diag(chol_cov_self)))

        chol_cov_other = np.linalg.cholesky(other.cov)
        det_term_other = 2 * np.sum(np.log(np.diag(chol_cov_other)))
        chol_prec_other = np.linalg.inv(chol_cov_other)  # cholesky of precision of q
        prec_other = chol_prec_other.T @ chol_prec_other

        mean_div = .5 * (other.mean - self.mean).T @ prec_other @ (other.mean - self.mean)
        rot_div = .5 * (np.trace(prec_other @ self.cov) - self.cov.shape[1] + det_term_other - det_term_self)
        return mean_div + rot_div
