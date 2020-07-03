import numpy as np
import tensorflow as tf
from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
from utils.tf_utils import tf_atleast_2d


class Ellipsoid(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(Ellipsoid, self).__init__(dim, int_opt)
        self.c = np.power(1e6, self.i / (self.dim - 1))

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.t_osz(x - self.x_opt)

        out = np.sum(self.c * z ** 2, axis=1) + self.f_opt
        return out


class EllipsoidTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-5., 5.), dtype=tf.float32):
        super(EllipsoidTensorflow, self).__init__(dim, int_opt=int_opt, dtype=dtype)
        self.c = 1e6 ** (self.i / (self.dim - 1))

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.t_osz(x - self.x_opt)

        out = tf.reduce_sum(self.c * z ** 2, axis=1) + self.f_opt
        return out


class EllipsoidRotated(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(EllipsoidRotated, self).__init__(dim, int_opt)
        self.c = np.power(1e6, self.i / (self.dim - 1))

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.t_osz(self.r @ (x - self.x_opt).T).T
        return np.sum(self.c * z ** 2, axis=1) + self.f_opt


class EllipsoidRaw(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(EllipsoidRaw, self).__init__(dim, int_opt)
        self.c = np.power(1e6, self.i / (self.dim - 1))
        self.x_opt = np.zeros(shape=(1, dim))
        self.f_opt = 0

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        out = np.sum(self.c * x ** 2, axis=1) + self.f_opt
        return out
