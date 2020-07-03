import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DummyContext:
    def __init__(self, dim, dtype=tf.float32):
        self.dim = dim
        self.f_opt = 0

        self.dtype = dtype

    def __call__(self, action, context=None):
        g1 = tfd.MultivariateNormalDiag(tf.constant([-5, -1], dtype=self.dtype),
                                        tf.constant([1, 0.25], dtype=self.dtype) * 2)
        g2 = tfd.MultivariateNormalDiag(tf.ones(action.shape, dtype=self.dtype) * -2.,
                                        tf.ones(action.shape, dtype=self.dtype) * 3)
        if context is not None:
            return -tf.where(tf.squeeze(context) >= 0, g1.prob(action), g2.prob(action))
        else:
            return -g1.prob(action)
