import numpy as np
import tensorflow as tf
from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
from utils.tf_utils import tf_atleast_2d


class Discus(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(Discus, self).__init__(dim, int_opt)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = self.t_osz(self.r @ (x - self.x_opt).T).T

        return 1.e6 * z[:, 0] ** 2 + np.sum(z[:, 1:] ** 2, axis=1) + self.f_opt


class DiscusTensorflow(BaseObjectiveTensorflow):
    def __init__(self, input_dim, int_opt=(-5., 5.), dtype=tf.float32):
        super(DiscusTensorflow, self).__init__(input_dim, int_opt, dtype=dtype)

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        z = tf.transpose(self.t_osz(self.r @ tf.transpose(x - self.x_opt)))

        return 1.e6 * z[:, 0] ** 2 + tf.reduce_sum(z[:, 1:] ** 2, axis=1) + self.f_opt
