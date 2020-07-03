import numpy as np
import tensorflow as tf
from tensorflow import linalg as tflin
from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
from utils.tf_utils import tf_atleast_2d


class Rastrigin(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.), alpha=10, beta=0.2):
        super(Rastrigin, self).__init__(dim, int_opt, alpha=alpha, beta=beta)
        self.mat_fac = self.lambda_alpha

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        # TODO: maybe flip data dimensions?
        z = (self.mat_fac @ self.t_asy_beta(self.t_osz(x - self.x_opt).T)).T

        out = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z,
                                                                                       axis=1) ** 2 + self.f_opt

        return out


class RastriginTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-5., 5.), alpha=10, beta=0.2, dtype=tf.float32):
        super(RastriginTensorflow, self).__init__(dim, int_opt, alpha=alpha, beta=beta, dtype=dtype)
        self.mat_fac = self.lambda_alpha

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        # TODO: maybe flip data dimensions?
        z = tf.transpose(self.mat_fac @ self.t_asy_beta(tf.transpose(self.t_osz(x - self.x_opt))))

        out = 10 * (self.dim - tf.reduce_sum(tf.cos(2 * np.pi * z), axis=1)) + tflin.norm(z, axis=1) ** 2 + self.f_opt

        return out


class RastriginRotated(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.), alpha=10, beta=0.2):
        super(RastriginRotated, self).__init__(dim, int_opt, alpha=alpha, beta=beta)
        self.mat_fac = self.r @ self.lambda_alpha @ self.q

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        # TODO: maybe flip data dimensions?
        z = (self.mat_fac @ self.t_asy_beta(self.t_osz(self.r @ (x - self.x_opt).T))).T

        out = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z,
                                                                                       axis=1) ** 2 + self.f_opt

        return out
