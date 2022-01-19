import tensorflow_probability as tfp

tfd = tfp.distributions

DEFAULT_OPTIMIZER_CONFIG = {
    'dt': 0.02,
    'lmbda': 1,
    'sigma': 1,
    'alpha': 100,
    'anisotropic': True,
}

DEFAULT_INITIAL_DISTRIBUTION = tfd.Normal(1, 4)
