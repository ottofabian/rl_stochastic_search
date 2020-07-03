import numpy as np
import tensorflow as tf

from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
import logging

from utils.tf_utils import tf_atleast_2d

logger = logging.getLogger('BentCigar')


class BentCigar(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.), beta=0.5, condition=1e6):
        super(BentCigar, self).__init__(dim, int_opt=int_opt, beta=beta)
        self.condition = condition

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = (self.r @ self.t_asy_beta(self.r @ (x - self.x_opt).T)).T

        return z[:, 0] ** 2 + self.condition * np.sum(z[:, 1:] ** 2, axis=1) + self.f_opt


class BentCigarTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-5., 5.), beta=0.5, condition=1e6, dtype=tf.float32):
        super(BentCigarTensorflow, self).__init__(dim, int_opt=int_opt, beta=beta, dtype=dtype)
        self.condition = condition

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        x = tf.transpose(x - self.x_opt)
        z = tf.transpose(self.r @ self.t_asy_beta(self.r @ x))

        out = z[:, 0] ** 2 + self.condition * tf.reduce_sum(z[:, 1:] ** 2, axis=1) + self.f_opt
        return out


class BentCigarRaw(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.), beta=0.5):
        super(BentCigarRaw, self).__init__(dim, int_opt=int_opt, beta=beta)
        self.f_opt = 0
        self.x_opt = np.zeros((1, dim))

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        return x[:, 0] ** 2 + 1e6 * np.sum(x[:, 1:] ** 2, axis=1)


class BentCigarRawTensorflow(BaseObjectiveTensorflow):
    def __init__(self, input_dim, int_opt=(-5., 5.), beta=0.5, dtype=tf.float32):
        super(BentCigarRawTensorflow, self).__init__(input_dim, int_opt=int_opt, beta=beta, dtype=dtype)
        self.f_opt = 0
        self.x_opt = tf.zeros((1, input_dim))

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        return x[:, 0] ** 2 + 1e6 * tf.reduce_sum(x[:, 1:] ** 2, axis=1)
