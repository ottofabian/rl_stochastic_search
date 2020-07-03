import numpy as np


class Rosenbrock:
    def __init__(self, n):
        self.n = n

    def function_eval(self, x):
        assert x.shape[0] == self.n

        a = x[1:, :] - x[:-1, :]**2
        b = 1 - x[:-1, :]
        out = np.sum(100 * a**2 + b**2, axis=0)

        return out
