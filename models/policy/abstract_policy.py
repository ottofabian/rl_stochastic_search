from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class AbstractPolicy(ABC):

    def __init__(self):
        pass

    def __call__(self, context=None, training=True, **kwargs):
        return self.predict_dist(context, training)

    @property
    @abstractmethod
    def distribution(self):
        pass

    @distribution.setter
    def distribution(self, x):
        pass

    def density(self, x, context=None):
        """

        :param x:
        :return:

        Args:
            context:
        """
        pass

    def log_prob(self, x, context=None):
        """
        compute log density of the policy
        """
        pass

    def sample(self, num_samples: int, context: Union[np.ndarray, tf.Tensor] = None):
        """

        :param num_samples:
        :param context:
        :return:
        """
        pass

    def predict_dist(self, context=None, training=True):
        """

        Args:
            training:
            context:

        Returns:

        """
        pass

    def entropy(self, context=None):
        """

        :return:

        Args:
            x:
        """
        pass

    @abstractmethod
    def params(self):
        """

        Returns:

        """
        pass

    @abstractmethod
    def set_params(self, params: list):
        """

        Args:
            params:

        Returns:

        """
        pass

    @property
    def mean(self):
        """

        :return:
        """
        pass

    @mean.setter
    def mean(self, x) -> None:
        """

        :param x:
        :return:
        """
        pass

    @property
    def cov(self):
        """

        :return:
        """
        pass

    @cov.setter
    def cov(self, x) -> None:
        """
        Setter for covariance matrix/matrices. Might be required for non gradient-based updates, such as MORE.
        :param x: new covariance matrix
        :return:
        """
        pass

    def kl_divergence(self, other, context=None):
        """

        :param other:
        :return:

        Args:
            x:
        """
        pass

    @property
    @abstractmethod
    def trainable_variables(self) -> list:
        """

        :return:
        """
        pass

    @property
    def is_root(self):
        """

        Returns:

        """
        return False

    @property
    def output_shape(self):
        return

    @property
    def input_shape(self):
        return

    @property
    def batch_shape(self):
        return

    def save(self, filepath, **kwargs):
        pass

    def load(self, filepath, **kwargs):
        pass
