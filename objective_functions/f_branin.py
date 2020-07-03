import numpy as np
import tensorflow as tf
from objective_functions.f_base import BaseObjectiveTensorflow
from utils.tf_utils import tf_atleast_2d


class BraninTensorflow(BaseObjectiveTensorflow):
    def __init__(self, dim, int_opt=(-5., 5.), alpha=10, beta=0.2, dtype=tf.float32):
        # only defined for 2D
        assert dim == 2

        super(BraninTensorflow, self).__init__(dim, int_opt, alpha=alpha, beta=beta, dtype=dtype)
        self.mat_fac = self.lambda_alpha

    def evaluate_full(self, x, context=None):
        x = tf_atleast_2d(x)
        assert x.shape[1] == self.dim

        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5.0 / np.pi
        t = 1.0 / (8.0 * np.pi)
        u = x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - 6

        r = 10.0 * (1.0 - t) * tf.cos(x[:, 0]) + 10
        out = u ** 2 + r

        return out
