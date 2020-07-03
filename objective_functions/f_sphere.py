import numpy as np
import tensorflow as tf
from objective_functions.f_base import BaseObjective, BaseObjectiveTensorflow
from utils.tf_utils import tf_atleast_2d


class Sphere(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(Sphere, self).__init__(dim, int_opt)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = x - self.x_opt
        return np.linalg.norm(z, axis=1) ** 2 + self.f_opt


class SphereTensorflow(BaseObjectiveTensorflow):
    def __init__(self, input_dim, int_opt=(-5., 5.), dtype=tf.float32):
        super(SphereTensorflow, self).__init__(input_dim, int_opt, dtype=dtype)

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        z = x - self.x_opt
        out = tf.linalg.norm(z, axis=1) ** 2 + self.f_opt
        return out


if __name__ == "__main__":
    test = Sphere(2)
    test(np.zeros(2))
