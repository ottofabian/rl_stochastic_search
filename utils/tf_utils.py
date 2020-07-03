import logging

import tensorflow as tf
import tensorflow.keras as k


def conditional_function(condition):
    return lambda func: tf.function(func) if condition else func


def tf_atleast_2d(x, reversed=False):
    """

    Args:
        x:
        reversed:

    Returns:

    """
    tensor = tf.convert_to_tensor(x)
    if len(tensor.shape) == 0:
        result = tf.reshape(tensor, [1, 1])
    elif len(tensor.shape) == 1:
        result = tensor[:, tf.newaxis] if reversed else tensor[tf.newaxis, :]
    else:
        result = tensor
    return result


def flatten_batch(tensor):
    """
        flatten axes 0 and 1
    Args:
        tensor: tensor to flatten

    Returns:

    """

    s = tensor.shape
    return tf.reshape(tensor, [s[0] * s[1], *s[2:]])


def get_optimizer(type, learning_rate, **kwargs):
    """

    Args:
        type:
        learning_rate:
        **kwargs:

    Returns:

    """
    if type == "sgd":
        # return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.25, nesterov=True)
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)
    elif type == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
    elif type == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate, **kwargs)
    else:
        ValueError(f"Optimizer {type} is not supported.")


def get_layer_initializer(type):
    """

    Args:
        type:

    Returns:

    """
    if type == "orthogonal":
        return k.initializers.Orthogonal(2 ** 0.5)
    elif type == "xavier":
        return k.initializers.GlorotNormal()
    else:
        logging.warning(f"Initalizer {type} not supported. Using Xavier Uniform")
        return k.initializers.GlorotUniform()
