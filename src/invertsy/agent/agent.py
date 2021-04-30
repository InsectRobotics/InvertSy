"""
Package the contains the default agents.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from ._helpers import eps, RNG
from invertsy.env import Sky, Seville2009

from invertpy.sense import PolarisationSensor, CompoundEye, Sensor
from invertpy.brain import MushroomBody, WillshawNetwork, CentralComplex, PolarisationCompass, Component
from invertpy.brain.mushroombody import IncentiveCircuit
from invertpy.brain.activation import winner_takes_all, relu
from invertpy.brain.compass import decode_sph
from invertpy.brain.preprocessing import Whitening, DiscreteCosineTransform

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class Agent(object):
    def __init__(self, xyz=None, ori=None, speed=0.1, delta_time=1., dtype='float32', name='agent', rng=RNG):
        """
        Abstract agent class that holds all the basic methods and attributes of an agent such as:

        - 3D position and initial position
        - 3D orientation and initial orientation
        - delta time: how fast its internal clock is ticking
        - delta x: how fast is it moving
        - name
        - sensors
        - brain components
        - translation and rotation methods

        Parameters
        ----------
        xyz: np.ndarray[float], optional
            the initial 3D position of the agent. Default is p=[0, 0, 0]
        ori: R, optional
            the initial 3D orientation of the agent. Default is q=[1, 0, 0, 0]
        speed: float, optional
            the agent's speed. Default is dx=0.1 meters/sec
        delta_time: float, optional
            the agent's internal clock speed. Default is 1 tick/second
        name: str, optional
            the name of the agent. Default is 'agent'
        dtype: np.dtype, optional
            the type of the agents parameters
        """
        if xyz is None:
            xyz = [0, 0, 0]
        if ori is None:
            ori = R.from_euler('Z', 0)

        self._sensors = []  # type: list[Sensor]
        self._brain = []  # type: list[Component]

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
        """
        Re-initialises the parameters, sensors and brain components of the agent.
        """
        self._xyz = self._xyz_init.copy()
        self._ori = copy(self._ori_init)

        for sensor in self.sensors:
            sensor.reset()

        for component in self.brain:
            component.reset()

    def _sense(self, *args, **kwargs):
        """
        Senses the environment. This method needs to be implemented by the sub-class.

        Returns
        -------
        out
            the output of the sensors

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def _act(self):
        """
        Acts in the environment. This method needs to be implemented by the sub-class.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """
        Senses the environment and then acts in it given the parameters.

        Returns
        -------
        out
            the output of the sensors
        """
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

    def move_forward(self, dx=None, dt=None):
        """
        Move towards the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([1, 0, 0], dx, dt)

    def move_backward(self, dx=None, dt=None):
        """
        Move towards the opposite of the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([-1, 0, 0], dx, dt)

    def move_right(self, dx=None, dt=None):
        """
        Move sideways to the right of the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([0, 1, 0], dx, dt)

    def move_left(self, dx=None, dt=None):
        """
        Move sideways to the left of the facing direction.

        Parameters
        ----------
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        self.move_towards([0, -1, 0], dx, dt)

    def move_towards(self, direction_xyz, dx=None, dt=None):
        """
        Moves the agent towards a 3D direction (locally - relative to the current direction) using for a dx/dt distance.

        Parameters
        ----------
        direction_xyz: np.ndarray[float], list[float]
            3D vector showing the direction of motion
        dx: float, optional
            the length of the motion per seconds. Default is the internal one
        dt: float, optional
            the seconds passing. Default is the internal one
        """
        if dt is None:
            dt = self._dt_default
        if dx is None:
            dx = self._dx

        # compute the step size based on the new delta time
        dx = dx * dt

        # normalise the vector
        direction_xyz = np.array(direction_xyz) / np.maximum(np.linalg.norm(direction_xyz), eps)

        self.translate(self._ori.apply(dx * direction_xyz))

    def rotate(self, d_ori: R):
        """
        Rotate the agent and its sensor on the spot.

        Parameters
        ----------
        d_ori: Rotation
            the rotation to apply on the current direction of the agent
        """
        self._ori = self._ori * d_ori
        for sensor in self._sensors:
            sensor.rotate(d_ori, around_xyz=self._xyz)

    def translate(self, d_xyz):
        """
        Translates the agent and its sensors by adding the given vector in global coordinates.

        Parameters
        ----------
        d_xyz: np.ndarray[float], list[float]
            the vector to add in global coordinates
        """
        self._xyz += np.array(d_xyz, dtype=self.dtype)
        for sensor in self._sensors:
            sensor.translate(d_xyz)

    def add_sensor(self, sensor, local=False):
        """
        Adds a sensor to the agent. By default, the sensor is assumed to have its orientation and position in global
        coordinates, but this can be changed through the 'local' option.

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
        """
        The sensors of the agent.

        Returns
        -------
        sensors: list[Sensor]
        """
        return self._sensors

    @property
    def brain(self):
        """
        The brain components of the agent.

        Returns
        -------
        brain: list[Component]
        """
        return self._brain

    @property
    def xyz(self):
        """
        The position of the agent.

        Returns
        -------
        xyz: np.ndarray[float]

        See Also
        --------
        Agent.position
        """
        return self._xyz

    @xyz.setter
    def xyz(self, v):
        """
        The position of the agent.

        Parameters
        ----------
        v: np.ndarray[float]

        See Also
        --------
        Agent.position
        """
        self.translate(np.array(v, dtype=self.dtype) - self._xyz)

    @property
    def x(self):
        """
        The x component of the position of the agent.

        Returns
        -------
        x: float
        """
        return self._xyz[0]

    @property
    def y(self):
        """
        The y component of the position of the agent.

        Returns
        -------
        y: float
        """
        return self._xyz[1]

    @property
    def z(self):
        """
        The z component of the position of the agent.

        Returns
        -------
        z: float
        """
        return self._xyz[2]

    @property
    def ori(self):
        """
        The orientation of the agent

        Returns
        -------
        ori: R

        See Also
        --------
        Agent.orientation
        """
        return self._ori

    @ori.setter
    def ori(self, v):
        """
        Parameters
        ----------
        v: R

        See Also
        --------
        Agent.orientation
        """
        self.rotate(d_ori=self._ori.inv() * v)

    @property
    def euler(self):
        """
        The orientation of the agent as euler angles (yaw, pitch, roll) in radians.

        Returns
        -------
        euler: np.ndarray[float]
        """
        return self._ori.as_euler('ZYX', degrees=False)

    @property
    def yaw(self):
        """
        The yaw of the agent in radians.

        Returns
        -------
        yaw: float
        """
        return self.euler[0]

    @property
    def pitch(self):
        """
        The pitch of the agent in radians.

        Returns
        -------
        pitch: float
        """
        return self.euler[1]

    @property
    def roll(self):
        """
        The roll of the agent in radians.

        Returns
        -------
        roll: float
        """
        return self.euler[2]

    @property
    def euler_deg(self):
        """
        The orientation of the agent as euler angles (yaw, pitch, roll) in degrees.

        Returns
        -------
        euler_deg: np.ndarray[float]
        """
        return self._ori.as_euler('ZYX', degrees=True)

    @property
    def yaw_deg(self):
        """
        The yaw of the agent in degrees.

        Returns
        -------
        yaw_deg: float
        """
        return self.euler_deg[0]

    @property
    def pitch_deg(self):
        """
        The pitch of the agent in degrees.

        Returns
        -------
        pitch_deg: float
        """
        return self.euler_deg[1]

    @property
    def roll_deg(self):
        """
        The roll of the agent in degrees.

        Returns
        -------
        roll_deg: float
        """
        return self.euler_deg[2]

    @property
    def position(self):
        """
        The position of the agent.

        Returns
        -------
        position: np.ndarray[float]

        See Also
        --------
        Agent.xyz
        """
        return self._xyz

    @property
    def orientation(self):
        """
        The orientation of the agent.

        Returns
        -------
        orientation: np.ndarray[float]

        See Also
        --------
        Agent.ori
        """
        return self._ori

    @property
    def step_size(self):
        """
        The step size (dx) per delta time (dt).

        Returns
        -------
        dx: float
        """
        return self._dx

    @property
    def delta_time(self):
        """
        The delta time (dt) among time-steps.

        Returns
        -------
        dt: float
        """
        return self._dt_default


class PathIntegrationAgent(Agent):

    def __init__(self, *args, **kwargs):
        """
        Agent specialised in the path integration task. It contains the Dorsal Rim Area as a sensor, the polarised
        light compass and the central complex as brain components.
        """
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
        """
        Using its only sensor (the dorsal rim area) it senses the radiation from the sky which is interrupted by the
        given scene, and the optic flow for self motion calculation.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        flow: np.ndarray[float], optional
            the optic flow. Default is the preset optic flow

        Returns
        -------
        out: np.ndarray[float]
            the output of the central complex
        """
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
        """
        Uses the output of the central complex to compute the next movement and moves the agent to its new position.
        """
        steer = self.get_steering(self._cx)
        self.rotate(R.from_euler('Z', steer, degrees=False))
        self.move_forward()

    @staticmethod
    def get_steering(cx) -> float:
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        cx: CentralComplex
            the central complex instance of the agent

        Returns
        -------
        output: float
            the angle of steering in radians
        """

        cpu1a = cx.r_cpu1[1:-1]
        cpu1b = np.array([cx.r_cpu1[-1], cx.r_cpu1[0]])
        motor = cpu1a @ cx.w_cpu1a2motor + cpu1b @ cx.w_cpu1b2motor
        output = motor[0] - motor[1]  # * .25  # to kill the noise a bit!
        return output


class VisualNavigationAgent(Agent):

    def __init__(self, eye=None, memory=None, saturation=1.5, nb_scans=7, freq_trans=True, *args, **kwargs):
        """
        Agent specialised in the visual navigation task. It contains the CompoundEye as a sensor and the mushroom body
        as the brain component.

        Parameters
        ----------
        eye: CompoundEye, optional
            instance of the compound eye of the agent. Default is a compound eye with 5000 ommatidia, with 15 deg
            acceptance angle each, sensitive to green only and not sensitive to polarised light.
        memory: MushroomBody, optional
            instance of a mushroom body model as a processing component. Default is the WillshawNetwork with #PN equal
            to the number of ommatidia, #KC equal to 40 x #PN, sparseness is 1%, and eligibility trace (lambda) is 0.1
        saturation: float, optional
            the maximum radiation level that the eye can handle, anything above this threshold will be saturated.
            Default is 1.5
        nb_scans: int, optional
            the number of scans during the route following task. Default is 7
        freq_trans: bool, optional
            whether to transform the visual input into the frequency domain by using the DCT method. Default is False
        """
        super().__init__(*args, **kwargs)

        if eye is None:
            eye = CompoundEye(nb_input=5000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15), omm_res=saturation,
                              c_sensitive=[0, 0., 1., 0., 0.])

        if memory is None:
            # #KC = 40 * #PN
            memory = WillshawNetwork(nb_cs=eye.nb_ommatidia, nb_kc=eye.nb_ommatidia * 40, sparseness=0.01,
                                     eligibility_trace=.1)
        if isinstance(memory, IncentiveCircuit):
            memory.f_cs = lambda x: np.asarray(x > np.sort(x)[int(memory.nb_cs * .7)], dtype=self.dtype)
            memory.f_kc = lambda x: np.asarray(winner_takes_all(x, percentage=memory.sparseness), dtype=self.dtype)
            memory.f_mbon = lambda x: relu(x * memory.sparseness, cmax=2)
            memory.b_m = np.array([0, 0, 0, 0, 0, 0])
            memory.b_d = np.array([-.0, -.0, -.0, -.0, -.0, -.0])

        self.add_sensor(eye)
        self.add_brain_component(memory)

        self._eye = eye  # type: CompoundEye
        self._mem = memory  # type: MushroomBody

        self._pref_angles = np.linspace(-60, 60, nb_scans)
        """
        The preferred angles for scanning
        """
        self._familiarity = np.zeros_like(self._pref_angles)
        """
        The familiarity of each preferred angle
        """

        self._preprocessing = [Whitening(nb_input=eye.nb_ommatidia, dtype=eye.dtype)]
        """
        List of the preprocessing components
        """
        if freq_trans:
            self._preprocessing.insert(0, DiscreteCosineTransform(nb_input=eye.nb_ommatidia, dtype=eye.dtype))

        self.reset()

    def reset(self):
        super().reset()

        self._familiarity = np.zeros_like(self._pref_angles)

        for process in self._preprocessing:
            process.reset()

    def _sense(self, sky=None, scene=None, **kwargs):
        """
        Using its only sensor (the compound eye) it senses the radiation from the sky and the given scene to calculate
        the familiarity. In the case of route following (when there is no update), it scans in all the preferred angles
        and calculates the familiarity in all of them.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None

        Returns
        -------
        familiarity: np.ndarray[float]
            how familiar does the agent is with every scan made
        """

        front = self._familiarity.shape[0] // 2
        self._familiarity[:] = 0.

        if self.update:
            r = self.get_pn_responses(sky=sky, scene=scene)
            r_mbon = self._mem(cs=r, us=np.ones(1, dtype=self.dtype))

            self._familiarity[front] = self.get_familiarity(r_mbon)
            if self._mem.nb_kc > 0:
                self._familiarity[front] /= (np.sum(self._mem.r_kc[0] > 0) + eps)
        else:
            ori = copy(self.ori)

            for i, angle in enumerate(self._pref_angles):
                self.ori = ori * R.from_euler('Z', angle, degrees=True)
                r = self.get_pn_responses(sky=sky, scene=scene)
                self._familiarity[i] = self.get_familiarity(self._mem(cs=r))
                if self._mem.nb_kc > 0:
                    self._familiarity[i] /= (np.sum(self._mem.r_kc[0] > 0) + eps)
            self.ori = ori

        return self._familiarity

    def _act(self):
        """
        Uses the familiarity vector to compute the next movement and moves the agent to its new position.
        """
        steer = self.get_steering(self.familiarity, self.pref_angles, max_steering=20, degrees=True)
        self.rotate(R.from_euler('Z', steer, degrees=True))
        self.move_forward()

    def calibrate(self, sky=None, scene=None, nb_samples=32, radius=2.):
        """
        Approximates the calibration of the optic lobes of the agent.
        In this case, it randomly collects a number of samples (in different positions and direction) in a radius
        around the nest. These samples are used in order to build a PCA whitening map, that transforms the visual
        input from the ommatidia to a white signal thying to maximise its information.

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None
        nb_samples: int, optional
            the number of samples to use
        radius: float, optional
            the radius around the nest from where the samples will be taken

        Returns
        -------
        xyz: list[np.ndarray[float]]
            the positions of the agent for every sample
        ori: list[R]
            the orientations of the agent for every sample
        """
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
        self._preprocessing[-1].reset(samples)

        self.xyz = xyz
        self.ori = ori

        print("Calibration: DONE!")

        return xyzs, oris

    def get_pn_responses(self, sky=None, scene=None):
        """
        Transforms the current snapshot of the environment into the PN responses.

        - Apply DCT (if applicable)
        - Apply PCA whitening (if applicable)

        Parameters
        ----------
        sky: Sky, optional
            the sky instance. Default is None
        scene: Seville2009, optional
            the world instance. Default is None

        Returns
        -------
        r: np.ndarray[float]
            the responses of the PNs
        """
        r = np.clip(self._eye(sky=sky, scene=scene).mean(axis=1), 0, 1)
        for process in self._preprocessing:
            r = process(r)
        return r

    @property
    def familiarity(self):
        """
        The familiarity of the latest snapshot per preferred angle
        """
        return self._familiarity

    @property
    def pref_angles(self):
        """
        The preferred angles of the agent, where it will look at when scanning
        """
        return self._pref_angles

    @property
    def nb_scans(self):
        """
        The number of scans to be applied
        """
        return self._pref_angles.shape[0]

    @property
    def update(self):
        """
        Whether the memory will be updated or not
        """
        return self._mem.update

    @update.setter
    def update(self, v):
        """
        Enables (True) or disables (False) the memory updates.

        Parameters
        ----------
        v: bool
            memory updates
        """
        self._mem.update = v

    @property
    def is_calibrated(self):
        """
        Indicates if calibration has been completed
        """
        return self._preprocessing[-1].calibrated

    @staticmethod
    def get_steering(familiarity, pref_angles, max_steering=None, degrees=True):
        """
        Outputs a scalar where sign determines left or right turn.

        Parameters
        ----------
        familiarity: np.ndarray[float]
            the familiarity vector computed by scanning the environment
        pref_angles: np.ndarray[float]
            the preference angle associated to the values of the familiarity vector
        max_steering: float, optional
            the maximum steering allowed for the agent. Default is 30 degrees
        degrees: bool, optional
            whether the max_steering is in degrees or radians. Default is degrees

        Returns
        -------
        output: float
            the angle of steering in radians
        """
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

    def get_familiarity(self, r_mbon):
        """
        Computes the familiarity using the MBON responses.

        Parameters
        ----------
        r_mbon: np.ndarray[float]
            MBON responses

        Returns
        -------
        float
            the familiarity
        """
        if isinstance(self._mem, IncentiveCircuit):
            return .5 + np.mean(r_mbon[[0, 2, 4]] - r_mbon[[1, 3, 5]])
        else:
            return r_mbon
