# weight initialisers

# uniform initialisation
import numpy as np


def uniform_init(weight_scale, a=-1, b=1):
    w = weight_scale * np.random.uniform(a, b)
    return w

# standard normal initialisation


def normal_init(weight_scale, loc=0, scale=1):
    w = weight_scale * np.random.normal(loc, scale)
    return w
