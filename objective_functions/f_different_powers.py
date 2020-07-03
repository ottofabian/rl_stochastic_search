import numpy as np
from objective_functions.f_base import BaseObjective


class DifferentPowers(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(DifferentPowers, self).__init__(dim, int_opt=int_opt)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = (self.r @ (x - self.x_opt).T).T

        out = np.sqrt(np.sum(np.power(np.abs(z), 2 + 4 * self.i / (self.dim - 1)), axis=1)) + self.f_opt

        return out


class DifferentPowersRaw(BaseObjective):
    def __init__(self, dim, int_opt=(-5., 5.)):
        super(DifferentPowersRaw, self).__init__(dim, int_opt=int_opt)
        self.x_opt = np.zeros((1, dim))
        self.f_opt = 0

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = x

        # out = np.sqrt(np.sum(np.power(np.abs(z), 2 + 4 * self.i / (self.d - 1)), axis=1))
        out = np.sum(np.power(np.abs(x), 2 + 10 * self.i / (self.dim - 1)), axis=1)

        return out
