from invertsy.__helpers import *

import numpy as np


def uni_modal_reinforcement(nb_rotations, nb_reinforcements, dtype=float):
    us = np.zeros((nb_rotations, nb_reinforcements), dtype=dtype)
    us[0, 0] = 1.
    return us


def bimodal_reinforcement(nb_rotations, nb_reinforcements, dtype=float):
    us = np.zeros((nb_rotations, nb_reinforcements), dtype=dtype)
    reins = np.arange(nb_reinforcements)
    rotas = np.asarray(reins * nb_rotations / nb_reinforcements, dtype=int)
    us[rotas, reins] = 1.
    return us


def sinusoidal_reinforcement(nb_rotations, nb_reinforcements, dtype=float):
    us = np.zeros((nb_rotations, nb_reinforcements), dtype=dtype)
    for rein in range(nb_reinforcements):
        us[:, rein] = np.cos(2 * np.pi * rein / nb_reinforcements -
                             np.linspace(0, 2 * np.pi, nb_rotations, endpoint=False))
    return us


def exponential_reinforcement(nb_rotations, nb_reinforcements, m=5, dtype=float):
    us = np.zeros((nb_rotations, nb_reinforcements), dtype=dtype)
    for rein in range(nb_reinforcements):
        us[:, rein] = np.exp(m * np.absolute((2 * np.pi * rein / nb_reinforcements -
                                              np.linspace(0, 2 * np.pi, nb_rotations, endpoint=False) + np.pi)
                                             % (2 * np.pi) - np.pi))
    return us
