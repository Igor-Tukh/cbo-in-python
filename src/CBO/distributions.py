import tensorflow_probability as tfp

tfd = tfp.distributions


def Normal(loc=0., scale=1.):
    return tfd.Normal(loc, scale)
