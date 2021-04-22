import numpy as np

import os


RNG = np.random.RandomState(2021)
"""
The defaults random value generator.
"""
eps = np.finfo(float).eps
"""
The smallest non-zero positive.
"""
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
"""
The root directory
"""
__data__ = os.path.join(__root__, 'data')


def set_rng(seed):
    """
    Sets the default random state.

    Parameters
    ----------
    seed: int
    """
    global RNG
    RNG = np.random.RandomState(seed)
