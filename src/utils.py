import numpy as np
import random
import torch
import tensorflow as tf


def fix_seed(seed=239):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
