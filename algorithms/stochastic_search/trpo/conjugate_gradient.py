import numpy as np
import tensorflow as tf


@tf.function
def conjugate_gradients(func, b, num_steps):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = tf.zeros_like(b)
    r = tf.identity(b)  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = tf.identity(r)
    r_dot_old = tf.tensordot(r, r, 1)
    for _ in range(num_steps):
        z = func(p)
        alpha = r_dot_old / (tf.tensordot(p, z, 1) + 1e-10)
        x += alpha * p
        r -= alpha * z
        r_dot_new = tf.tensordot(r, r, 1)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x


def conjugate_gradients_numpy(func, b, num_steps):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = np.zeros_like(b)
    r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for _ in range(num_steps):
        z = func(p)
        alpha = r_dot_old / (np.dot(p, z) + 1e-10)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x
