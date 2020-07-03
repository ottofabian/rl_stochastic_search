import numpy as np
from objective_functions.f_base import BaseObjective


class LinearSlope(BaseObjective):
    def __init__(self, dim, val_opt=5):
        super(LinearSlope, self).__init__(dim, val_opt=val_opt)
        self.c = np.power(10, self.i / (self.dim - 1))

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.dim

        z = np.where(self.x_opt * x < 25, x, self.x_opt)
        s = np.sign(self.x_opt) * self.c
        # TODO: check implementation for boundary errors
        return np.sum(5 * np.abs(s) - s * z, axis=1) + self.f_opt
