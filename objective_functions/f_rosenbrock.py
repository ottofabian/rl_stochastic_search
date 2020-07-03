import numpy as np
from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
import tensorflow as tf

from utils.tf_utils import tf_atleast_2d


class Rosenbrock(BaseObjective):
    def __init__(self, dim, int_opt=(-3., 3.)):
        super(Rosenbrock, self).__init__(dim, int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.dim) / 8)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.c * (x - self.x_opt) + 1
        z_end = z[:, 1:]
        z_begin = z[:, :-1]

        a = z_begin ** 2 - z_end
        b = z_begin - 1

        return np.sum(100 * a ** 2 + b ** 2, axis=1) + self.f_opt


class RosenbrockTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-3., 3.), dtype=tf.float32):
        super(RosenbrockTensorflow, self).__init__(dim, int_opt=int_opt, dtype=dtype)
        self.c = tf.maximum(1, tf.sqrt(tf.cast(self.dim, dtype)) / 8)

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.c * (x - self.x_opt) + 1
        z_end = z[:, 1:]
        z_begin = z[:, :-1]

        a = z_begin ** 2 - z_end
        b = z_begin - 1

        return tf.reduce_sum(100 * a ** 2 + b ** 2, axis=1) + self.f_opt


class RosenbrockRotated(BaseObjective):
    def __init__(self, dim, int_opt=(-3., 3.)):
        super(RosenbrockRotated, self).__init__(dim, int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.dim) / 8)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = (self.c * self.r @ x.T + 1 / 2).T
        a = z[:, :-1] ** 2 - z[:, 1:]
        b = z[:, :-1] - 1

        return np.sum(100 * a ** 2 + b ** 2, axis=1) + self.f_opt


class RosenbrockRotatedTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-3., 3.), dtype=tf.float32):
        super(RosenbrockRotatedTensorflow, self).__init__(dim, int_opt=int_opt, dtype=dtype)
        self.c = tf.maximum(1, tf.sqrt(tf.cast(self.dim, dtype)) / 8)

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        z = tf.transpose(self.c * self.r @ tf.transpose(x) + 1 / 2)
        a = z[:, :-1] ** 2 - z[:, 1:]
        b = z[:, :-1] - 1

        return tf.reduce_sum(100 * a ** 2 + b ** 2, axis=1) + self.f_opt


class RosenbrockRaw(BaseObjective):
    def __init__(self, dim, int_opt=(-3., 3.)):
        super(RosenbrockRaw, self).__init__(dim, int_opt=int_opt)
        self.x_opt = np.ones((1, dim))
        self.f_opt = 0

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        a = x[:, :-1] ** 2 - x[:, 1:]
        b = x[:, :-1] - 1

        out = np.sum(100 * a ** 2 + b ** 2, axis=1)

        return out


class RosenbrockRawTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-3., 3.), dtype=tf.float32):
        super(RosenbrockRawTensorflow, self).__init__(dim, int_opt=int_opt, dtype=dtype)
        self.c = tf.maximum(1, tf.sqrt(tf.cast(self.dim, dtype)) / 8)
        self.f_opt = 0
        self.x_opt = tf.convert_to_tensor([[1, 1]])

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        a = x[:, :-1] ** 2 - x[:, 1:]
        b = x[:, :-1] - 1

        out = tf.reduce_sum(100 * a ** 2 + b ** 2, axis=1)

        return out
