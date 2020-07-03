import numpy as np
import scipy.stats as scistats
import tensorflow as tf
from tensorflow import linalg as tflin
import tensorflow_probability as tfp

tfd = tfp.distributions

np.seterr(divide='ignore', invalid='ignore')


class BaseObjective(object):
    def __init__(self, dim, int_opt=None, val_opt=None, alpha=None, beta=None):
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        # check if optimal parameter is in interval...
        if int_opt is not None:
            self.x_opt = np.random.uniform(int_opt[0], int_opt[1], size=(1, dim))
        # ... or based on a single value
        elif val_opt is not None:
            self.one_pm = np.where(np.random.rand(1, dim) > 0.5, 1, -1)
            self.x_opt = val_opt * self.one_pm
        else:
            raise ValueError("Optimal value or interval has to be defined")
        self.f_opt = np.round(np.clip(scistats.cauchy.rvs(loc=0, scale=100, size=1)[0], -1000, 1000), decimals=2)
        self.i = np.arange(self.dim)
        self._lambda_alpha = None
        self._q = None
        self._r = None

    def __call__(self, x):
        return self.evaluate_full(x)

    def evaluate_full(self, x):
        raise NotImplementedError("Subclasses should implement this!")

    def gs(self):
        # Gram Schmidt ortho-normalization
        a = np.random.randn(self.dim, self.dim)
        b, _ = np.linalg.qr(a)
        return b

    # TODO: property probably unnecessary
    @property
    def q(self):
        if self._q is None:
            self._q = self.gs()
        return self._q

    @property
    def r(self):
        if self._r is None:
            self._r = self.gs()
        return self._r

    @property
    def lambda_alpha(self):
        if self._lambda_alpha is None:
            if isinstance(self.alpha, int):
                lambda_ii = np.power(self.alpha, 1 / 2 * self.i / (self.dim - 1))
                self._lambda_alpha = np.diag(lambda_ii)
            else:
                lambda_ii = np.power(self.alpha[:, None], 1 / 2 * self.i[None, :] / (self.dim - 1))
                self._lambda_alpha = np.stack([np.diag(l_ii) for l_ii in lambda_ii])
        return self._lambda_alpha

    @staticmethod
    def f_pen(x):
        return np.sum(np.maximum(0, np.abs(x) - 5), axis=1)

    def t_asy_beta(self, x):
        # exp = np.power(x, 1 + self.beta * self.i[:, None] / (self.input_dim - 1) * np.sqrt(x))
        # return np.where(x > 0, exp, x)
        return x

    def t_osz(self, x):
        x_hat = np.where(x != 0, np.log(np.abs(x)), 0)
        c_1 = np.where(x > 0, 10, 5.5)
        c_2 = np.where(x > 0, 7.9, 3.1)
        return np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c_1 * x_hat) + np.sin(c_2 * x_hat)))


class BaseObjectiveTensorflow(object):
    def __init__(self, dim, int_opt=None, val_opt=None, alpha=None, beta=None, dtype=tf.float32):
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        # check if optimal parameter is in interval...
        if int_opt is not None:
            self.x_opt = tf.random.uniform((1, dim), int_opt[0], int_opt[1], dtype=dtype)
        # ... or based on a single value
        elif val_opt is not None:
            self.one_pm = tf.where(tf.random.uniform(1, dim, dtype=dtype) > 0.5, 1., -1.)
            self.x_opt = val_opt * self.one_pm
        else:
            raise ValueError("Optimal value or interval has to be defined")

        # round by 2 decimal values
        multiplier = tf.cast(tf.constant(10. ** 2), dtype)
        self.f_opt = tf.round(tf.cast(tf.clip_by_value(tfd.Cauchy(loc=0., scale=100.).sample([1])[0], -1000, 1000),
                                      dtype=dtype) * multiplier) / multiplier

        self.i = tf.range(self.dim, dtype=dtype)
        self._lambda_alpha = None
        self._q = None
        self._r = None

        self.dtype = dtype

    def __call__(self, x, context=None):
        return self.evaluate_full(x)

    # @tf.function
    def evaluate_full(self, x, context=None):
        raise NotImplementedError("Subclasses should implement this!")

    @tf.function
    def gs(self):
        # Gram Schmidt ortho-normalization
        a = tf.random.normal((self.dim, self.dim), dtype=self.dtype)
        b, _ = tf.linalg.qr(a)
        return b

    # TODO: property probably unnecessary
    @property
    @tf.function
    def q(self):
        if self._q is None:
            self._q = self.gs()
        return self._q

    @property
    @tf.function
    def r(self):
        if self._r is None:
            self._r = self.gs()
        return self._r

    @property
    # TODO Fix tf stuff here
    @tf.function
    def lambda_alpha(self):
        if self._lambda_alpha is None:
            if isinstance(self.alpha, int):
                lambda_ii = self.alpha ** (1 / 2 * self.i / (self.dim - 1))
                self._lambda_alpha = tflin.diag(lambda_ii)
            else:
                lambda_ii = self.alpha[:, None] ** (1 / 2 * self.i[None, :] / (self.dim - 1))
                self._lambda_alpha = tf.stack([tflin.diag(l_ii) for l_ii in lambda_ii])
        return self._lambda_alpha

    @staticmethod
    def f_pen(x):
        return tf.reduce_sum(tf.maximum(0, tf.abs(x) - 5), axis=1)

    def t_asy_beta(self, x):
        x_safe = tf.where(x > 0, x, 1.0)
        return tf.where(x > 0, x_safe ** (1 + self.beta * (self.i[:, None] / (self.dim - 1)) * tf.sqrt(x_safe)), x)

    def t_osz(self, x):
        x_safe = tf.where(x != 0, x, 1.0)
        x_hat = tf.where(x != 0, tf.math.log(tf.abs(x_safe)), 0)
        c_1 = tf.cast(tf.where(x > 0, 10., 5.5), self.dtype)
        c_2 = tf.cast(tf.where(x > 0, 7.9, 3.1), self.dtype)
        return tf.sign(x) * tf.exp(x_hat + 0.049 * (tf.sin(c_1 * x_hat) + tf.sin(c_2 * x_hat)))
