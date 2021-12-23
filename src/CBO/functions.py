import numpy as np
import tensorflow as tf


def rastrigin(v):
    return rastrigin_c()(v)


def rastrigin_c(c=10):
    return lambda v: tf.reduce_sum(v ** 2 - c * tf.math.cos(2 * np.pi * v)) + c * tf.cast(tf.size(v), tf.float32)


def square(v):
    return v ** 2
