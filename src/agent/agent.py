__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

from ._helpers import eps, RNG

from invertsensing import PolarisationSensor, CompoundEye, Sensor
from invertbrain import MushroomBody, WillshawNetwork, CentralComplex, PolarisationCompass, Component
from invertbrain.compass import decode_sph
from invertbrain.synapses import whitening_synapses, whitening, dct_synapses
from invertbrain.activation import softmax

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class Agent(object):
    def __init__(self, xyz=None, ori=None, speed=0.1, delta_time=1., dtype='float32', name='agent', rng=RNG):
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

        self.rng = rng

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


class PathIntegrationAgent(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pol_sensor = PolarisationSensor(nb_input=60, field_of_view=56, degrees=True)
        pol_brain = PolarisationCompass(nb_pol=60, loc_ori=copy(pol_sensor.omm_ori), nb_sol=8, integrated=True)
        cx = CentralComplex(nb_tb1=8)

        self.add_sensor(pol_sensor, local=True)
        self.add_brain_component(pol_brain)
        self.add_brain_component(cx)

        self._pol_sensor = pol_sensor
        self._pol_brain = pol_brain
        self._cx = cx

        self._default_flow = self._dx * np.ones(2) / np.sqrt(2)

    def _sense(self, sky=None, scene=None, flow=None, **kwargs):
        if sky is None:
            r = 0.
        else:
            r = self._pol_sensor(sky=sky, scene=scene)

        if flow is None:
            flow = self._default_flow
            # if scene is None:
            #     flow = self._default_flow
            # else:
            #     flow = optic_flow(world, self._dx)

        r_tcl = self._pol_brain(r_pol=r, ori=self._pol_sensor.ori)
        _, phi = decode_sph(r_tcl)
        return self._cx(phi=phi, flow=flow)

    def _act(self):
        steer = self.get_steering(self._cx)
        self.rotate(R.from_euler('Z', steer, degrees=False))
        self.move_forward()

    @staticmethod
    def get_steering(cx) -> float:
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        cx

        Returns
        -------

        """

        cpu1a = cx.r_cpu1[1:-1]
        cpu1b = np.array([cx.r_cpu1[-1], cx.r_cpu1[0]])
        motor = cpu1a @ cx.w_cpu1a2motor + cpu1b @ cx.w_cpu1b2motor
        output = motor[0] - motor[1]  # * .25  # to kill the noise a bit!
        return output


class VisualNavigationAgent(Agent):

    def __init__(self, eye: CompoundEye = None, memory: MushroomBody = None, saturation=1.5, nb_scans=7,
                 freq_trans=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if eye is None:
            eye = CompoundEye(nb_input=5000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15), omm_res=saturation,
                              c_sensitive=[0, 0., 1., 0., 0.])

        if memory is None:
            memory = WillshawNetwork(nb_cs=eye.nb_ommatidia, nb_kc=eye.nb_ommatidia * 20, sparseness=0.01,
                                     eligibility_trace=.1)

        self.add_sensor(eye)
        self.add_brain_component(memory)

        self._eye = eye
        self._mem = memory

        self._pref_angles = np.linspace(-60, 60, nb_scans)
        self._familiarity = np.zeros_like(self._pref_angles)

        self._w_white = None
        self._m_white = None
        self._is_calibrated = False

        self._w_dct = None
        self._freq_trans = freq_trans

        self.reset()

    def reset(self):
        super().reset()

        self._familiarity = np.zeros_like(self._pref_angles)

        self._w_white = None
        self._m_white = None
        self._is_calibrated = False

        if self._freq_trans:
            self._w_dct = dct_synapses(self._eye.nb_ommatidia, dtype=self._eye.dtype)

    def _sense(self, sky=None, scene=None, **kwargs):

        front = self._familiarity.shape[0] // 2
        self._familiarity[:] = 0.

        if self.update:
            r = self.get_pn_responses(sky=sky, scene=scene)
            self._familiarity[front] = self._mem(cs=r, us=np.ones(1, dtype=self.dtype))
            self._familiarity[front] /= (np.sum(self._mem.r_kc[0] > 0) + eps)
        else:
            ori = copy(self.ori)

            for i, angle in enumerate(self._pref_angles):
                self.ori = ori * R.from_euler('Z', angle, degrees=True)
                r = self.get_pn_responses(sky=sky, scene=scene)
                self._familiarity[i] = self._mem(cs=r)
                self._familiarity[i] /= (np.sum(self._mem.r_kc[0] > 0) + eps)
            self.ori = ori

        return self._familiarity

    def _act(self):
        steer = self.get_steering(self.familiarity, self.pref_angles, max_steering=20, degrees=True)
        self.rotate(R.from_euler('Z', steer, degrees=True))
        self.move_forward()

    def calibrate(self, sky=None, scene=None, nb_samples=32, radius=2.):
        xyz = copy(self.xyz)
        ori = copy(self.ori)

        samples = np.zeros((nb_samples, self._mem.nb_cs), dtype=self.dtype)
        xyzs, oris = [], []
        for i in range(nb_samples):
            self.xyz = xyz + self.rng.uniform(-radius, radius, 3) * np.array([1., 1., 0])
            self.ori = R.from_euler("Z", self.rng.uniform(-180, 180), degrees=True)
            samples[i] = self.get_pn_responses(sky, scene)
            xyzs.append(copy(self.xyz))
            oris.append(copy(self.ori))
            print("Calibration: %d/%d - x: %.2f, y: %.2f, z: %.2f, yaw: %d" % (
                i + 1, nb_samples, self.x, self.y, self.z, self.yaw_deg))
        self._w_white, self._m_white = whitening_synapses(samples, dtype=self.dtype, bias=True)

        self.xyz = xyz
        self.ori = ori

        self._is_calibrated = True
        print("Calibration: DONE!")

        return xyzs, oris

    def get_pn_responses(self, sky=None, scene=None):
        r = np.clip(self._eye(sky=sky, scene=scene).mean(axis=1), 0, 1)
        if self._freq_trans:
            # transform the input to the coefficients using the Discrete Cosine Transform
            r = r @ self._w_dct
        if self._w_white is not None and self._m_white is not None:
            r_white = whitening(r, self._w_white, self._m_white)
            return softmax((r_white - r_white.min()) / (r_white.max() - r_white.min() + eps), tau=.2, axis=0)
        else:
            return r

    @property
    def familiarity(self):
        return self._familiarity

    @property
    def pref_angles(self):
        return self._pref_angles

    @property
    def nb_scans(self):
        return self._pref_angles.shape[0]

    @property
    def update(self):
        return self._mem.update

    @update.setter
    def update(self, v):
        self._mem.update = v

    @property
    def is_calibrated(self):
        return self._is_calibrated

    @staticmethod
    def get_steering(familiarity, pref_angles, max_steering=None, degrees=True):
        if max_steering is None:
            max_steering = np.deg2rad(30)
        elif degrees:
            max_steering = np.deg2rad(max_steering)
        if degrees:
            pref_angles = np.deg2rad(pref_angles)
        r = familiarity.max() - familiarity
        r = r / (r.sum() + eps)
        pref_angles_c = r * np.exp(1j * pref_angles)

        steer = np.clip(np.angle(np.sum(pref_angles_c) / (np.sum(familiarity) + eps)), -max_steering, max_steering)
        if np.isnan(steer):
            steer = 0.
        print("Steering: %d" % np.rad2deg(steer))
        if degrees:
            steer = np.rad2deg(steer)
        return steer
