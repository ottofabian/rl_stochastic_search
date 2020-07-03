import numpy as np
import tensorflow as tf

from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
from utils.tf_utils import tf_atleast_2d


class BuecheRastrigin(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(BuecheRastrigin, self).__init__(dim, int_opt)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        y = self.t_osz(x - self.x_opt)
        s = np.empty_like(y)

        for i in range(self.dim):
            if i % 2 == 0:
                s[:, i] = np.where(y[:, i] > 0,
                                   10 * np.power(np.sqrt(10), i / (self.dim - 1)),
                                   np.power(np.sqrt(10), i / (self.dim - 1)))

        z = s * y

        out = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1) ** 2 \
              + 100 * self.f_pen(x) + self.f_opt

        return out


class BuecheRastriginTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-5., 5.), dtype=tf.float32):
        super(BuecheRastriginTensorflow, self).__init__(dim, int_opt, dtype=dtype)

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        y = self.t_osz(x - self.x_opt)
        s = np.zeros_like(y)

        for i in range(self.dim):
            if i % 2 == 0:
                s[:, i] = np.where(y[:, i] > 0,
                                   10 * np.power(np.sqrt(10), i / (self.dim - 1)),
                                   np.power(np.sqrt(10), i / (self.dim - 1)))

        z = s * y

        out = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1) ** 2 \
              + 100 * self.f_pen(x) + self.f_opt

        return out
