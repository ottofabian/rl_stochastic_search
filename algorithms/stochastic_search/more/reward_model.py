import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class QuadModelWhitening:
    def __init__(self, dim_theta, max_samples=None, ridge_factor=1e-10):

        # dimensionalities of data
        self.dim_theta = dim_theta
        dim_tri = int(self.dim_theta * (self.dim_theta + 1) / 2)
        self.dim_beta = 1 + self.dim_theta + dim_tri

        # bias term
        self.r_0 = 0
        # linear term
        self.r = np.zeros(dim_theta)
        # quadratic term
        self.R = np.zeros((dim_theta, dim_theta))

        # standardization parameters
        self.r_mean = None
        self.r_std = None

        # params for whitening inputs
        self.theta_org = None
        self.theta_mean = None
        self.theta_inv_std = None

        # FIXME: remove hard coding
        self.N = max_samples
        self.top = int(np.floor(self.N / 2))
        weights = np.log(self.N / 2 + 0.5) - np.log(np.arange(self.top, 0, -1))
        self.weights = np.reshape(weights, [self.top, 1])  # / np.sum(weights)

        self.lambda_ridge = ridge_factor
        self.tau = 5

    def get_model_params(self):
        return self.R, self.r, self.r_0

    def get_weighting(self, type, r):

        if type is None:
            weighting = np.ones(shape=(r.size, 1))
        elif type == "inverse":
            weighting = 1 / np.abs(r - r.max() - 1)
        elif type == "rank":
            temp = np.argsort(r, axis=0).flatten()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(r))
            weighting = self.weights[ranks]
            # theta = x_top
        elif type == "exp":
            weighting = np.exp(self.tau * (r - np.max(r)))
        else:
            raise NotImplementedError

        sumW = np.sum(weighting)
        return weighting / sumW

    def fit_model(self, theta, r, theta_mean=None, normalize_input=True, normalize_output=True, theta_inv_std=None,
                  reward_weighting=None, do_whiten=True):
        """

        :param theta:
        :param r:
        :param theta_mean:
        :param normalize_input:
        :param normalize_output:
        :param theta_inv_std:
        :param reward_weighting:
        :param do_whiten:
        :return:
        """

        if reward_weighting == "rank":
            # get unsorted indices of top half of individuals
            top_ind = np.argpartition(r.flatten(), -self.top)[-self.top:]
            theta = theta[top_ind, :]
            r = r[top_ind]

        # TODO make this as class and normalizer init
        if normalize_output:
            self.r_mean = np.mean(r)
            self.r_std = np.std(r)
            r = (r - self.r_mean) / self.r_std

        if do_whiten:
            self.theta_org = theta
            theta = theta - theta_mean
            theta = theta @ theta_inv_std
            self.theta_mean = theta_mean
            self.theta_inv_std = theta_inv_std
        else:
            self.theta_mean = np.zeros((1, self.dim_theta))
            self.theta_inv_std = np.eye(self.dim_theta)

        weighting = self.get_weighting(reward_weighting, r)

        poly = PolynomialFeatures(2)
        phi = poly.fit_transform(theta)

        if normalize_input:
            phi_mean = np.mean(phi[:, 1:], axis=0, keepdims=True)
            phi_std = np.std(phi[:, 1:], axis=0, keepdims=True, ddof=1)
            phi[:, 1:] = phi[:, 1:] - phi_mean
            phi[:, 1:] = phi[:, 1:] / phi_std

            # TODO: check for variance of theta to discard columns of phi?

        phi_weighted = phi * weighting

        phi_t_phi = phi_weighted.T @ phi
        phi_t_phi[:, 1:] += self.lambda_ridge * np.eye(self.dim_beta)[:, 1:]

        par = np.linalg.solve(phi_t_phi, phi_weighted.T @ r)

        if normalize_input:
            par[1:] = par[1:] / phi_std.T
            par[0] = par[0] - par[1:].T @ phi_mean.T

        square_feat_upper_tri_ind = np.triu_indices(self.dim_theta)
        M = np.zeros((self.dim_theta, self.dim_theta))
        M_tri = par[self.dim_theta + 1:].flatten()
        M[square_feat_upper_tri_ind] = M_tri
        M = 1 / 2 * (M + M.T)

        m_0 = par[0]
        m = par[1:self.dim_theta + 1]

        if do_whiten:
            D = theta_inv_std @ M @ theta_inv_std
            B = theta_inv_std @ m
            M = - 2 * D  # to achieve -1/2 xMx + xm form
            m = - 2 * (theta_mean @ D).T + B
            m_0 = m_0 + theta_mean @ (D @ theta_mean.T - B)
        else:
            M = - 2 * M
            m_0 = par[0]

        self.r_0 = m_0
        self.r = m
        self.R = M

    def test_error(self, theta, r):
        out = -.5 * np.diag(theta @ self.R @ theta.T)[:, None] + theta @ self.r + self.r_0
        err = np.abs(out.flatten() - (np.asarray(r) - self.r_mean) / self.r_std)
        return np.mean(err ** 2)

    def eval_model(self, x):
        return -1 / 2 * np.diag(x @ self.R @ x.T)[:, None] + x @ self.r + self.r_0
