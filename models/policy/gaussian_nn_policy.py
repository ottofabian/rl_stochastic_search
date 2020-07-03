# import logging
# from collections import OrderedDict
# from typing import Union
#
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as k
# import tensorflow_probability as tfp
# from tensorflow import linalg as tfl
# from tensorflow_probability.python.bijectors import FillScaleTriL
#
# from models.networks import get_mlp
# from models.policy.abstract_policy import AbstractPolicy
# from utils.tf_utils import get_layer_initializer
#
# logger = logging.getLogger('conditional_gaussian')
#
#
# @tf.custom_gradient
# def sign(x):
#     """
#     Sign function.
#
#     Clip and binarize tensor using the straight through estimator for the gradient,
#     e.g. used in `Binarized Neural Networks`: https://arxiv.org/abs/1602.02830.
#     Args:
#         x: input tensor
#
#     Returns: binary tensor with same shape as x.
#
#     """
#
#     def grad(dy):
#         return tf.clip_by_value(dy, -1, 1)
#
#     return tf.sign(x, name='sign'), grad
#
#
# class ConditionalGaussianPolicy(AbstractPolicy):
#     """
#     Conditional Gaussian policy network
#     """
#
#     def __init__(self,
#                  input_shape: Union[tuple, tf.TensorShape],
#                  output_shape: Union[tuple, tf.TensorShape],
#                  layers: Union[tuple, list],
#                  activation: str = "tanh",
#                  init_type: str = "orthogonal",
#                  max_action: float = 1,
#                  predict_root: bool = False,
#                  contextual_covariance: bool = False,
#                  l2_penalty: Union[None, float] = None,
#                  batch_norm: bool = False,
#                  dropout: Union[None, float] = None,
#                  use_bias: bool = True,
#                  dtype: tf.DType = tf.float32):
#
#         super().__init__()
#
#         self._input_shape = input_shape
#         self._output_shape = output_shape[0]
#         self.max_action = max_action
#
#         self._hparams = self.generate_h_params(layers, init_type, activation, l2_penalty or 0.0, batch_norm,
#                                                dropout or 0.0, contextual_covariance)
#
#         self._is_root = predict_root
#         self.contextual_covariance = contextual_covariance
#
#         self.bijector = FillScaleTriL(diag_bijector=tfp.bijectors.Exp(),
#                                       diag_shift=tf.constant(1e-5, dtype=dtype))
#
#         init = get_layer_initializer(init_type)
#
#         if self.contextual_covariance:
#             out_nodes = tfp.layers.MultivariateNormalTriL.params_size(self._output_shape)
#         else:
#             I = tf.eye(self._output_shape, batch_shape=(1,))
#             init_cov = I if not self.is_root else tf.sqrt(I)
#             self._scale_tril = tfp.util.TransformedVariable(init_cov, bijector=self.bijector, name="scale",
#                                                             dtype=dtype)
#             out_nodes = self._output_shape
#
#         l2 = k.regularizers.l2(l2_penalty) if l2_penalty else None
#
#         # model
#         inp = k.layers.Input(input_shape, dtype=dtype)
#         x = get_mlp(inp, layers, activation, init, l2, batch_norm, dropout, use_bias, dtype)
#         out = k.layers.Dense(out_nodes, activation="linear", kernel_initializer=init, kernel_regularizer=l2,
#                              use_bias=use_bias, dtype=dtype)(x)
#
#         self._model = k.Model(inputs=inp, outputs=out)
#
#     @tf.function
#     def __call__(self, context=None, training=True, **kwargs):
#
#         output = self._model(context, training=training)
#
#         if self.contextual_covariance:
#             std = self.bijector(output[..., self._output_shape:])
#             loc = output[..., :self._output_shape]
#         else:
#             std = tf.tile(tf.convert_to_tensor(self._scale_tril), [tf.shape(context)[0], 1, 1])
#             loc = output
#
#         # std += tf.eye(self._output_shape) * 3
#
#         if self.max_action is not None:
#             loc = k.activations.tanh(loc) * self.max_action
#             # loc = sign(loc) * self.max_action
#
#         if self.is_root:
#             # return sqrt instead of cholesky
#             # Model is predicting the cholesky of sqrt
#             std = std @ tfl.matrix_transpose(std)
#
#         return loc, std
#
#     @property
#     def distribution(self):
#         return self._model
#
#     @distribution.setter
#     def distribution(self, x):
#         if isinstance(x, type(self._model)):
#             self._model = x
#         else:
#             raise ValueError("Trying to set invalid model type.")
#
#     @property
#     def trainable_variables(self) -> list:
#
#         if self.contextual_covariance:
#             return self._model.trainable_variables
#         else:
#             return self._model.trainable_variables + list(self._scale_tril.trainable_variables)
#
#     def params(self, flatten: bool = False):
#         return self._model.get_weights()
#
#     def set_params(self, params: list, is_flattened: bool = False):
#         self._model.set_weights(params)
#
#     @tf.function
#     def predict_dist(self, context=None, training=True):
#         return self(context, training)
#
#     @property
#     def is_root(self):
#         return self._is_root
#
#     @property
#     def output_shape(self):
#         return self._output_shape
#
#     @property
#     def input_shape(self):
#         return self._input_shape
#
#     @property
#     def batch_shape(self):
#         return None
#
#     def save(self, filepath, **kwargs):
#         self._model.save(filepath, **kwargs)
#
#     def load(self, filepath, **kwargs):
#         self._model.load_weights(filepath, **kwargs)
#
#     @property
#     def hparams(self):
#         return self._hparams
#
#     def generate_h_params(self, layers, init_type, activation, l2_penalty, batch_norm, dropout, contextual_covariance):
#         from tensorboard.plugins.hparams import api as hp
#         HP_POLICY_N_HIDDEN = hp.HParam("policy_n_hidden", hp.Discrete((2 ** np.arange(3, 8)).tolist()))
#         HP_POLICY_N_LAYERS = hp.HParam("policy_n_layers", hp.IntInterval(0, 3))
#         HP_POLICY_INIT_TYPE = hp.HParam("policy_init_type", hp.Discrete(["xavier", "orthogonal"]))
#         HP_POLICY_ACTIVATION = hp.HParam("policy_activation", hp.Discrete(["tanh", "relu", "selu", "elu"]))
#         HP_POLICY_L2_PENALTY = hp.HParam("policy_l2_penalty", hp.RealInterval(0., 1.))
#         HP_POLICY_BATCH_NORM = hp.HParam("policy_batch_norm", hp.Discrete([True, False]))
#         HP_POLICY_DROPOUT = hp.HParam("policy_dropout", hp.RealInterval(0., 0.99))
#         HP_POLICY_CONT_COV = hp.HParam("policy_contextual_covariance", hp.Discrete([True, False]))
#
#         h_params = {
#             HP_POLICY_N_HIDDEN: layers[0],
#             HP_POLICY_N_LAYERS: len(layers),
#             HP_POLICY_INIT_TYPE: init_type,
#             HP_POLICY_ACTIVATION: activation,
#             HP_POLICY_L2_PENALTY: l2_penalty,
#             HP_POLICY_BATCH_NORM: batch_norm,
#             HP_POLICY_DROPOUT: dropout,
#             HP_POLICY_CONT_COV: contextual_covariance,
#         }
#
#         return h_params
