import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf

tfd = tfp.distributions


def Normal(loc=0., scale=1.):
    return tfd.Normal(loc, scale)


class NumpyNormal:
    def sample(self, size):
        return tf.convert_to_tensor(np.random.normal(0, 1, size))
