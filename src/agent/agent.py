from ._helpers import eps

from invertbrain.component import Component
from invertsensing.sensor import Sensor

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class Agent(object):
    def __init__(self, xyz=None, ori=None, speed=0.1, delta_time=0.1, dtype='float32', name='agent'):
        """

        Parameters
        ----------
        xyz: np.ndarray, list
        ori: R
        speed: float
        delta_time: float
        name: str
        """
        if xyz is None:
            xyz = [0, 0, 0]
        if ori is None:
            ori = R.from_euler('Z', 0)

        self._sensors = []
        self._brain = []

        self._xyz = np.array(xyz, dtype=dtype)
        self._ori = ori

        self._xyz_init = self._xyz.copy()
        self._ori_init = copy(self._ori)

        self._dt_default = delta_time  # seconds
        self._dx = speed  # meters / second

        self.name = name
        self.dtype = dtype

    def reset(self):
        self._xyz = self._xyz_init.copy()
        self._ori = copy(self._ori_init)

        for sensor in self.sensors:
            sensor.reset()

        for component in self.brain:
            component.reset()

    def _sense(self, *args, **kwargs):
        raise NotImplementedError()

    def _act(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        act = kwargs.pop('act', True)
        callback = kwargs.pop('callback', None)

        out = self._sense(*args, **kwargs)
        if act:
            self._act()

        if callback is not None:
            callback(self)

        return out

    def __repr__(self):
        return ("Agent(xyz=[%.2f, %.2f, %.2f] m, ori=[%.0f, %.0f, %.0f] degrees, speed=%.2f m/s, "
                "#sensors=%d, #brain_components=%d, name='%s')") % (
            self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self._dx,
            len(self.sensors), len(self.brain), self.name
        )

    def move_forward(self, dt=None):
        self.move_towards([1, 0, 0], dt)

    def move_backward(self, dt=None):
        self.move_towards([-1, 0, 0], dt)

    def move_right(self, dt=None):
        self.move_towards([0, 1, 0], dt)

    def move_left(self, dt=None):
        self.move_towards([0, -1, 0], dt)

    def move_towards(self, direction_xyz, dt=None):
        """

        Parameters
        ----------
        direction_xyz: np.ndarray, list
        dt: float
        """
        if dt is None:
            dt = self._dt_default

        # normalise the vector
        direction_xyz = np.array(direction_xyz) / np.maximum(np.linalg.norm(direction_xyz), eps)

        # compute the step size based on the new delta time
        dx = self._dx * dt

        self.translate(self._ori.apply(dx * direction_xyz))

    def rotate(self, d_ori: R):
        self._ori = self._ori * d_ori
        for sensor in self._sensors:
            sensor.rotate(d_ori, around_xyz=self._xyz)

    def translate(self, d_xyz):
        """

        Parameters
        ----------
        d_xyz: np.ndarray, list
        """
        self._xyz += np.array(d_xyz, dtype=self.dtype)
        for sensor in self._sensors:
            sensor.translate(d_xyz)

    def add_sensor(self, sensor, local=False):
        """

        Parameters
        ----------
        sensor: Sensor
            The sensor to add.
        local: bool
            If True, then the orientation and coordinates of the sensor are supposed to be local (with respect to the
            agent's orientation and coordinates, otherwise it is global. Default is False (global).

        """
        if local:
            sensor.rotate(self._ori)
            sensor.translate(self._xyz)
        self._sensors.append(sensor)

    def add_brain_component(self, component: Component):
        self._brain.append(component)

    @property
    def sensors(self):
        return self._sensors

    @property
    def brain(self):
        return self._brain

    @property
    def xyz(self):
        return self._xyz

    @xyz.setter
    def xyz(self, v):
        self.translate(np.array(v, dtype=self.dtype) - self._xyz)

    @property
    def x(self):
        return self._xyz[0]

    @property
    def y(self):
        return self._xyz[1]

    @property
    def z(self):
        return self._xyz[2]

    @property
    def ori(self):
        return self._ori

    @ori.setter
    def ori(self, v):
        self.rotate(d_ori=self._ori.inv() * v)

    @property
    def euler(self):
        return self._ori.as_euler('ZYX', degrees=False)

    @property
    def yaw(self):
        return self.euler[0]

    @property
    def pitch(self):
        return self.euler[1]

    @property
    def roll(self):
        return self.euler[2]

    @property
    def euler_deg(self):
        return self._ori.as_euler('ZYX', degrees=True)

    @property
    def yaw_deg(self):
        return self.euler_deg[0]

    @property
    def pitch_deg(self):
        return self.euler_deg[1]

    @property
    def roll_deg(self):
        return self.euler_deg[2]

    @property
    def position(self):
        return self._xyz

    @property
    def orientation(self):
        return self._ori

    @property
    def step_size(self):
        return self._dx

    @property
    def delta_time(self):
        return self._dt_default
