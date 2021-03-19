from ._helpers import eps
from .agent import Agent

from invertbrain.mushroombody import MushroomBody, WillshawNetwork
from invertsensing.vision import CompoundEye

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class VisualNavigationAgent(Agent):

    def __init__(self, eye: CompoundEye = None, memory: MushroomBody = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if eye is None:
            eye = CompoundEye(nb_input=100, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15), omm_res=5.,
                              c_sensitive=[0, 0., 1., 0., 0.])

        if memory is None:
            memory = WillshawNetwork(nb_cs=100, nb_kc=10000)

        self.add_sensor(eye)
        self.add_brain_component(memory)

        self._eye = eye
        self._mem = memory

        self._pref_angles = np.linspace(-60, 60, 7)
        self._familiarity = np.zeros_like(self._pref_angles)

    def _sense(self, sky=None, scene=None, **kwargs):

        front = self._familiarity.shape[0] // 2
        self._familiarity[:] = 0.

        if self.update:
            r = self.get_pn_responses(sky=sky, scene=scene)
            self._familiarity[front] = self._mem(cs=r, us=np.ones(1, dtype=self.dtype))
        else:
            ori = copy(self.ori)

            for i, angle in enumerate(self._pref_angles):
                self.ori = ori * R.from_euler('Z', angle, degrees=True)
                r = self.get_pn_responses(sky=sky, scene=scene)
                self._familiarity[i] = self._mem(cs=r)
            self.ori = ori

        return self._familiarity

    def _act(self):
        steer = get_steering(self.familiarity, self.pref_angles, degrees=True)
        self.rotate(R.from_euler('Z', steer, degrees=False))
        self.move_forward()

    def get_pn_responses(self, sky=None, scene=None):
        r = self._eye(sky=sky, scene=scene).sum(axis=1)
        return (r - r.min()) / (r.max() - r.min() + eps)

    @property
    def familiarity(self):
        return self._familiarity

    @property
    def pref_angles(self):
        return self._pref_angles

    @property
    def update(self):
        return self._mem.update

    @update.setter
    def update(self, v):
        self._mem.update = v


def get_steering(familiarity, pref_angles, degrees=True):
    if degrees:
        pref_angles = np.deg2rad(pref_angles)
    pref_angles_c = np.exp(1j * pref_angles)
    return np.angle(np.sum(-familiarity * pref_angles_c) / np.sum(familiarity))
