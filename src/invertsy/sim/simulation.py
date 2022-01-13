"""
Package that contains a number of different simulations.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from abc import ABC

from ._helpers import col2x, row2y, yaw2ori, x2col, y2row, ori2yaw

from invertsy.__helpers import __data__, RNG
from invertsy.env import UniformSky, Sky, Seville2009, WorldBase, StaticOdour
from invertsy.agent import VisualNavigationAgent, PathIntegrationAgent, NavigatingAgent, Agent

from invertpy.sense import CompoundEye
from invertpy.sense.polarisation import PolarisationSensor
from invertpy.brain.preprocessing import Preprocessing
from invertpy.brain.compass import PolarisationCompass
from invertpy.brain.memory import MemoryComponent
from invertpy.brain.centralcomplex import CentralComplexBase

from scipy.spatial.transform import Rotation as R

import numpy as np

from time import time
from copy import copy

import os

__stat_dir__ = os.path.abspath(os.path.join(__data__, "animation", "stats"))
if not os.path.isdir(__stat_dir__):
    os.makedirs(__stat_dir__)


class Simulation(object):

    def __init__(self, nb_iterations, noise=0., rng=RNG, name="simulation"):
        """
        Abstract class that runs a simulation for a fixed number of iterations and logs statistics.

        Parameters
        ----------
        nb_iterations: int
            the number of iterations to run the simulation
        noise : float
            the noise amplitude in the system
        rng : np.random.RandomState
            the random number generator
        name: str, optional
            a unique name for the simulation. Default is 'simulation'
        """
        self._nb_iterations = nb_iterations
        self._iteration = 0
        self._message_intervals = 1
        self._stats = {}
        self._noise = noise
        self.rng = rng
        self._name = name

    def reset(self):
        """
        Resets the parameters and the logger of the simulation.

        Raises
        ------
        NotImplementedError
            Classes inheriting this interface must implement this method.
        """
        raise NotImplementedError()

    def _step(self, i):
        """
        Runs one step of the simulation for the given iteration.

        Parameters
        ----------
        i: int
            the iteration to run

        Raises
        ------
        NotImplementedError
            Classes inheriting this interface must implement this method.
        """
        raise NotImplementedError()

    def step(self, i):
        """
        Runs one step of the simulation for the given iteration and return the time needed to run the step.
        There is no need to override this step as it runs the code implemented in the _step method.

        Parameters
        ----------
        i: the iteration to run

        Returns
        -------
        dt: float
            the time needed to run the iteration (in sec)

        Raises
        ------
        NotImplementedError
            if the _step method has not been implemented
        """
        self._iteration = i
        t0 = time()
        self._step(i)
        t1 = time()
        return t1 - t0

    def save(self, filename=None):
        """
        Saves the logged statistics in a file.

        Parameters
        ----------
        filename: str, optional
            the name of the file without the ending or the path. The path is assumed to be the default path for the
            data in data/animation/stats. Default is the name of the simulation
        """
        if filename is None:
            filename = self._name
        else:
            filename = filename.replace('.npz', '')
        save_path = os.path.join(__stat_dir__, "%s.npz" % filename)
        np.savez_compressed(save_path, **self._stats)
        print("\nSaved stats in: '%s'" % save_path)

    def __call__(self, save=False):
        """
        Resets the simulation and runs all its iterations. At the end it saves its logged statistics if required.

        Parameters
        ----------
        save: bool
            if True the logged statistics are saved in the default file (name of the simulation) when the simulation
            ends or when it is interrupted by the keyboard. Default is False
        """
        try:
            self.reset()

            for self._iteration in range(self._iteration, self.nb_frames):
                dt = self.step(self._iteration)
                if self._iteration % self.message_intervals == 0:
                    print(self.message() + " - time: %.2f sec" % dt)
        except KeyboardInterrupt:
            print("Simulation interrupted by keyboard!")
        finally:
            if save:
                self.save()

    def message(self):
        """
        The message that shows the current progress of the simulation.
        This is printed after every iteration when the simulation is called.

        Returns
        -------
        str
        """
        str_len = len(f"{self._nb_iterations}")
        return f"Simulation {self._iteration + 1:{str_len}d}/{self._nb_iterations}"

    def set_name(self, name):
        """
        Changes the name of the simulation.

        Parameters
        ----------
        name: str
        """
        self._name = name

    @property
    def stats(self):
        """
        The logged statistics of the simulation as a dictionary.

        Returns
        -------
        dict
        """
        return self._stats

    @property
    def nb_frames(self):
        """
        The number of iterations (frames).

        Returns
        -------
        int
        """
        return self._nb_iterations

    @property
    def frame(self):
        """
        The current iteration ID.

        Returns
        -------
        int
        """
        return self._iteration

    @property
    def message_intervals(self):
        """
        The number of steps between printed messages.

        Returns
        -------
        int
        """
        return self._message_intervals

    @message_intervals.setter
    def message_intervals(self, v):
        self._message_intervals = v

    @property
    def name(self):
        """
        The name of the simulation.

        Returns
        -------
        str
        """
        return self._name


class RouteSimulation(Simulation):
    def __init__(self, route, eye=None, preprocessing=None, sky=None, world=None, *args, **kwargs):
        """
        Simulation that runs a predefined route in a world, given a sky and an eye model, and logs the input from the
        eye.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that holds the 3D position and 1D orientation (yaw) of the eye in every iteration.
        eye: CompoundEye, optional
            the compound eye model. Default is a green sensitive eye with 5000 ommatidia of 10 deg acceptance angle each
        preprocessing: list[Preprocessing]
            a list of preprocessors for the input eye. Default is None
        sky: Sky, optional
            the sky model. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world where the route was captured. Default is the Seville ant world
        """
        kwargs.setdefault('nb_iterations', route.shape[0])
        name = kwargs.get('name', None)
        super().__init__(*args, **kwargs)

        self._route = route

        if eye is None:
            eye = CompoundEye(nb_input=2000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10), omm_res=5.,
                              c_sensitive=[0., 0., 1., 0., 0.])
        self._eye = eye

        if preprocessing is None:
            preprocessing = []
        self._preprocessing = preprocessing

        if sky is None:
            sky = UniformSky(luminance=1.)
        self._sky = sky

        if world is None:
            world = Seville2009()
        self._world = world

        if name is None:
            name = world.name
        self._name = name

        self._r = eye(sky=sky, scene=world).mean(axis=1)

    def reset(self):
        """
        Runs the first iteration.
        """
        self._step(0)

    def _step(self, i: int):
        """
        Sets the position and orientation of the eye to the one indicated by the route for the given iteration, and
        captures the responses of its photo-receptors.

        Parameters
        ----------
        i: int
            the current iteration
        """
        self._iteration = i
        self._eye._xyz = self._route[i, :3]
        self._eye._ori = R.from_euler('Z', self._route[i, 3], degrees=True)

        r = self._eye(sky=self._sky, scene=self._world).mean(axis=1)

        for preprocess in self._preprocessing:
            r = preprocess(r)

        self._r = r

    def message(self):
        return super().message() + " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f" % tuple(self._route[self._iteration])

    @property
    def world(self):
        """
        The world where the simulation takes place.

        Returns
        -------
        Seville2009
        """
        return self._world

    @property
    def sky(self):
        """
        The sky of the world.

        Returns
        -------
        Sky
        """
        return self._sky

    @property
    def route(self):
        """
        The route that the eye follows.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route

    @property
    def eye(self):
        """
        The compound eye that captures the world.

        Returns
        -------
        CompoundEye
        """
        return self._eye

    @property
    def responses(self):
        """
        The responses of the eye's photo-receptors.

        Returns
        -------
        np.ndarray[float]
        """
        return self._r.T.flatten()


class NavigationSimulationBase(Simulation, ABC):

    def __init__(self, agent=None, sky=None, world=None, *args, **kwargs):
        """
        Runs the path integration task.
        An agent equipped with a compass and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        agent: Agent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world that creates the scenery disturbing the celestial compass or using the visual surroundings as a
            a compass. Default is None

        Other Parameters
        ----------------
        nb_iterations: int, optional
            the number of iterations to run the simulation. Default is 2.5 times the number of time-steps of the route
        name: str, optional
            the name of the simulation. Default is the name of the world or 'simulation'
        """

        name = kwargs.get('name', None)
        super().__init__(*args, **kwargs)

        self._agent = agent

        if sky is None:
            sky = Sky(30, 180, degrees=True)
        self._sky = sky
        self._world = world

        if name is None and world is not None:
            self._name = world.name

    def reset(self):
        """
        Resets the agent anf the logged statistics.
        """
        self._iteration = 0
        self.init_stats()
        self.agent.reset()

    def init_stats(self):
        self._stats = {"xyz": []}

    def update_stats(self, a):
        """
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: Agent
        """

        self._assert_agent(a)
        self._stats["xyz"].append([a.x, a.y, a.z, a.yaw])

    def approach_point(self, xyz, acceleration=0.0055):
        """
        Forces the agent to move towards the given point.

        Parameters
        ----------
        xyz : np.ndarray[float]
            the attraction point.
        acceleration : float, optional
            the acceleration of the agent. Default is 0.0055
        """
        # the attractive force
        f = xyz - self.agent.xyz

        # z-axis has 0 force
        f[2] = 0.

        # the magnitude of the attractive force
        f = f / np.linalg.norm(f[:2])

        if len(self.stats["xyz"]) > 1:
            v0 = np.array(self.stats["xyz"][-1])[:3] - np.array(self.stats["xyz"][-2])[:3]

            # the updated acceleration
            a = acceleration * f

            # updated velocity direction
            v = v0 + a
            v = v / np.linalg.norm(v[:2])
        else:
            v = f

        # update the velocity magnitude
        v = self.agent.step_size * v

        # move the agent to the new position
        self._agent.translate(v)

        # rotate the agent accordingly
        yaw = np.arctan2(v[1] / self.agent.step_size, v[0] / self.agent.step_size)
        self._agent.ori = R.from_euler('Z', yaw, degrees=False)

    def distance_from(self, xyz):
        return np.linalg.norm(self.agent.xyz - xyz)

    def _assert_agent(self, a):
        """
        Asserts an error message if the given agent is not the same as the internal one.

        Parameters
        ----------
        a: Agent
        """
        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

    def message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        return f"{super().message()} - x: {x:.2f}, y:{y:.2f}, z: {z:.2f}, Φ: {phi:.0f}"

    @property
    def agent(self):
        """
        The agent of the simulation.

        Returns
        -------
        Agent
        """
        return self._agent

    @property
    def sky(self):
        """
        The sky model of the environment.

        Returns
        -------
        Sky
        """
        return self._sky

    @property
    def world(self):
        """
        The vegetation of the environment.

        Returns
        -------
        Seville2009
        """
        return self._world


class CentralPointNavigationSimulationBase(NavigationSimulationBase, ABC):

    def __init__(self, xyz, *args, **kwargs):
        """
        Runs the path integration task.
        An agent equipped with a compass and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that holds the 3D position (x, y, z) and 1D orientation (yaw) for each iteration of the route
        agent: PathIntegrationAgent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world that creates the scenery disturbing the celestial compass or using the visual surroundings as a
            a compass. Default is None

        Other Parameters
        ----------------
        nb_iterations: int, optional
            the number of iterations to run the simulation. Default is 2.5 times the number of time-steps of the route
        name: str, optional
            the name of the simulation. Default is the name of the world or 'simulation'
        """
        super().__init__(*args, **kwargs)
        self._central_point = xyz

    def init_stats(self):
        super().init_stats()
        self._stats["L"] = []  # straight distance from the central point
        self._stats["C"] = []  # distance that the agent has covered

    def update_stats(self, a):
        super().update_stats(a)

        self._stats["L"].append(self.d_central_point)
        c = self._stats["C"][-1] if len(self._stats["C"]) > 0 else 0.
        if len(self._stats["xyz"]) > 1:
            step = np.linalg.norm(np.array(self._stats["xyz"][-1])[:3] - np.array(self._stats["xyz"][-2])[:3])
        else:
            step = 0.
        self._stats["C"].append(c + step)

    def init_inbound(self):
        """
        Sets up the inbound phase.
        Changes the labels of the logged stats to their outbound equivalent and resets them for the new phase to come.
        """
        self.init_inbound_stats()

    def init_inbound_stats(self):
        # create a separate line
        if "xyz_out" not in self._stats:
            self._stats["xyz_out"] = []
            self._stats["L_out"] = []
            self._stats["C_out"] = []

        self._stats["xyz_out"].extend(copy(self._stats["xyz"]))
        self._stats["L_out"].extend(copy(self._stats["L"]))
        self._stats["C_out"].extend(copy(self._stats["C"]))
        self._stats["xyz"] = []
        self._stats["L"] = []
        self._stats["C"] = []

    def message(self):
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))

        message = super().message()
        return f"{message} - L: {self.d_central_point:.2f}m, C: {d_trav:.2f}m"

    @property
    def central_point(self):
        return self._central_point

    @property
    def d_central_point(self):
        """
        The distance between the agent and the central point.

        Returns
        -------
        float
        """
        return np.linalg.norm(self.agent.xyz - self._central_point)


class PathIntegrationSimulation(CentralPointNavigationSimulationBase):

    def __init__(self, route, *args, **kwargs):
        """
        Runs the path integration task.
        An agent equipped with a compass and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that holds the 3D position (x, y, z) and 1D orientation (yaw) for each iteration of the route
        agent: PathIntegrationAgent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world that creates the scenery disturbing the celestial compass or using the visual surroundings as a
            a compass. Default is None

        Other Parameters
        ----------------
        nb_iterations: int, optional
            the number of iterations to run the simulation. Default is 2.5 times the number of time-steps of the route
        name: str, optional
            the name of the simulation. Default is the name of the world or 'simulation'
        """
        if len(args) == 0:
            kwargs.setdefault("xyz", route[0, :3])
        kwargs.setdefault('nb_iterations', int(3.5 * route.shape[0]))
        super().__init__(*args, **kwargs)
        self._route = route

        if self.agent is None:
            self._agent = PathIntegrationAgent(nb_feeders=1, speed=.01, rng=self.rng, noise=self._noise)

        self._compass_sensor = self.agent.sensors[0]
        self._compass_model, self._cx = self.agent.brain[:2]

        self._foraging = True
        self._distant_point = route[-1, :3]

    def reset(self):
        super().reset()

        self._foraging = True
        self.agent.ori = R.from_euler("Z", self.route[0, 3], degrees=True)

    def init_stats(self):
        super().init_stats()
        self._stats["POL"] = []
        self._stats["SOL"] = []
        self._stats["TB1"] = []
        self._stats["CL1"] = []
        self._stats["CPU1"] = []
        self._stats["CPU4"] = []
        self._stats["CPU4mem"] = []

    def _step(self, i):
        """
        Runs one iteration of the simulation. If the iteration is less than the maximum number of iterations in the
        route it forces the agent to follow the route, otherwise it lets the agent decide its actions.

        Parameters
        ----------
        i: int
            the iteration ID
        """
        act = True
        if i < self._route.shape[0]:  # outbound
            x, y, z, yaw = self._route[i]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
            self._foraging = True
        elif i == self._route.shape[0]:
            self.init_inbound()
            self._foraging = False
        elif self._foraging and self.distance_from(self.distant_point) < 0.5:
            self.approach_point(self.distant_point)
        elif not self._foraging and self.d_nest < 0.5:
            self.approach_point(self.central_point)
        elif self._foraging and self.distance_from(self.distant_point) < 0.2:
            self._foraging = False
            print("START PI FROM FEEDER")
        elif not self._foraging and self.d_nest < 0.2:
            self._foraging = True
            print("START FORAGING!")

        if self._foraging:
            motivation = np.array([0, 1])
        else:
            motivation = np.array([1, 0])

        self._agent(sky=self._sky, act=act, motivation=motivation, callback=self.update_stats)

    def update_stats(self, a):
        """
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: PathIntegrationAgent, NavigatingAgent
        """

        super().update_stats(a)

        compass, cx = a.brain[:2]
        self._stats["POL"].append(compass.r_pol.copy())
        self._stats["SOL"].append(compass.r_sol.copy())
        self._stats["CL1"].append(cx.r_cl1.copy())
        self._stats["TB1"].append(cx.r_tb1.copy())
        self._stats["CPU4"].append(cx.r_cpu4.copy())
        self._stats["CPU1"].append(cx.r_cpu1.copy())
        self._stats["CPU4mem"].append(cx.cpu4_mem.copy())

    @property
    def agent(self):
        """
        The agent of the simulation.

        Returns
        -------
        PathIntegrationAgent
        """
        return self._agent

    @property
    def route(self):
        """
        N x 4 array representing the route that the agent follows before returning to its initial position.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route

    @property
    def distant_point(self):
        return self._distant_point

    @property
    def compass_sensor(self):
        """
        The polarisation compass sensor.

        Returns
        -------
        PolarisationSensor
        """
        return self._compass_sensor

    @property
    def compass_model(self):
        """
        The Compass model.

        Returns
        -------
        PolarisationCompass
        """
        return self._compass_model

    @property
    def central_complex(self):
        """
        The Central Complex model.

        Returns
        -------
        CentralComplexBase
        """
        return self._cx

    @property
    def d_nest(self):
        """
        The distance between the agent and the nest.

        Returns
        -------
        float
        """
        return self.d_central_point

    @property
    def r_pol(self):
        """
        The POL responses of the compass model of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._compass_model.r_pol.T.flatten()

    @property
    def r_tb1(self):
        """
        The TB1 responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_tb1.T.flatten()

    @property
    def r_cl1(self):
        """
        The CL1 responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_cl1.T.flatten()

    @property
    def r_cpu1(self):
        """
        The CPU1 responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_cpu1.T.flatten()

    @property
    def r_cpu4(self):
        """
        The CPU4 responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_cpu4.T.flatten()

    @property
    def cpu4_mem(self):
        """
        The CPU4 memory of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.cpu4_mem.T.flatten()


class TwoSourcePathIntegrationSimulation(PathIntegrationSimulation):

    def __init__(self, route_a, route_b=None, *args, **kwargs):
        """
        Runs the path integration task.
        An agent equipped with a compass and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        route_a: np.ndarray[float]
            N x 4 matrix that holds the 3D position (x, y, z) and 1D orientation (yaw) for each iteration of the route
        agent: PathIntegrationAgent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world that creates the scenery disturbing the celestial compass or using the visual surroundings as a
            a compass. Default is None

        Other Parameters
        ----------------
        nb_iterations: int, optional
            the number of iterations to run the simulation. Default is 2.5 times the number of time-steps of the route
        name: str, optional
            the name of the simulation. Default is the name of the world or 'simulation'
        """
        if route_b is None:
            route_b = route_a.copy()   # mirror route A and create route B
            route_b[:, [1, 3]] = -route_b[:, [1, 3]]
            route_b[:, :3] += route_a[0, :3] - route_b[0, :3]
            route_b[:, 3] = (route_b[:, 3] + 180) % 360 - 180

        agent = kwargs.get('agent', None)
        kwargs.setdefault('nb_iterations', int(6. * route_a.shape[0] + 6. * route_b.shape[0]))
        super().__init__(route_a, *args, **kwargs)

        self._route_b = route_b
        self._distant_point_b = route_b[-1, :3]

        if agent is None:
            self._agent = PathIntegrationAgent(nb_feeders=2, speed=.01, rng=self.rng, noise=self._noise)

        self._forage_id = 0
        self._b_iter_offset = None
        self._state = []

    def reset(self):
        """
        Resets the agent anf the logged statistics.
        """

        super().reset()

        self._b_iter_offset = None
        self._foraging = True
        self._forage_id = 0
        self._state = []

    def init_stats(self):
        super().init_stats()

        if hasattr(self.agent.central_complex, "r_vec"):
            self._stats["vec"] = []

    def init_inbound(self, route_name='a'):
        """
        Sets up the inbound phase from source A.
        Changes the labels of the logged stats to their outbound equivalent and resets them for the new phase to come.

        Parameters
        ----------
        route_name : str, optional
            the route for which to initialise the inbound route. Default is 'a'
        """
        super().init_inbound()

        if len(self._state) == 0 or route_name != self._state[-1]:
            self._state.append(route_name)
            print(f"STATE: {self._state}")

    def _step(self, i):
        """
        Runs one iteration of the simulation. If the iteration is less than the maximum number of iterations in the
        route it forces the agent to follow the route, otherwise it lets the agent decide its actions.

        Parameters
        ----------
        i: int
            the iteration ID
        """
        if self._foraging:
            act = self.forage()
        else:
            act = self.home()

        motivation = self.get_motivation()

        self.agent(sky=self._sky, act=act, mbon=motivation, callback=self.update_stats)

    def forage(self):
        i = self._iteration
        act = True

        run_a = i < self.route_a.shape[0]
        run_b = self._b_iter_offset is None or (
                self._b_iter_offset is not None and i - self._b_iter_offset < self._route_b.shape[0])
        if run_a:  # outbound
            x, y, z, yaw = self.route_a[i]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
            self._forage_id = 1
        elif run_b:
            if self._b_iter_offset is None:
                self._b_iter_offset = i
            x, y, z, yaw = self._route_b[i - self._b_iter_offset]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
            self._forage_id = 2

        if self.distance_from(self.route_a[-1, :3]) < .1:
            self.init_inbound('a')
            if np.sum('b' == np.array(self._state)) > 0:
                self._foraging = True
                self._forage_id = 2
                print("GO TO B FROM A")
            else:
                self._foraging = False
                self._forage_id = 0
                print("START PI FROM A")
        elif self.distance_from(self.route_b[-1, :3]) < .1:
            self.init_inbound('b')
            if np.sum('b' == np.array(self._state)) > 1:
                self._foraging = True
                self._forage_id = 1
                print("GO TO A FROM B")
            else:
                self._foraging = False
                self._forage_id = 0
                print("START PI FROM B")
        elif act and self._state[-1] != 'a' and self.distance_from(self.route_a[-1, :3]) < .5:
            self.approach_point(self.route_a[-1, :3])
            act = False
        elif act and self._state[-1] != 'b' and self.distance_from(self.route_b[-1, :3]) < .5:
            self.approach_point(self.route_b[-1, :3])
            act = False

        return act

    def home(self):
        act = True

        if len(self.stats["L"]) > 1 and self.stats["L"][-2] < self.stats["L"][-1] < 0.1:
            self._foraging = True
            self._forage_id += 1
            self._forage_id = 1 + self._forage_id % 2
            if len(self._state) == 0 or 'n' != self._state[-1]:
                self._state.append('n')
            self.agent.central_complex.reset_integrator()

            print("START FORAGING!")

        elif len(self.stats["L"]) > 1 and self.stats["L"][-2] < self.stats["L"][-1] < 0.5:
            self.approach_point(self.route_a[0, :3])
            act = False

        return act

    def get_motivation(self):
        motivation = np.zeros(3, dtype=self.agent.dtype)
        if self._foraging:
            motivation[self._forage_id] = 1
        else:
            motivation[0] = 1
        return  motivation

    def update_stats(self, a):
        """
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: PathIntegrationAgent
        """
        super().update_stats(a)

        if hasattr(a.central_complex, "r_vec"):
            self._stats["vec"].append(a.central_complex.r_vec.copy())

    def message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        d_nest = self.d_nest
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
        return (super().message() + " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f - L: %.2fm, C: %.2fm") % (
            x, y, z, phi, d_nest, d_trav)

    @property
    def route_a(self):
        """
        N x 4 array representing the route that the agent follows to food source A before returning to its initial
        position.

        Returns
        -------
        np.ndarray[float]
        """
        return self.route

    @property
    def route_b(self):
        """
        N x 4 array representing the route that the agent follows to food source B before returning to its initial
        position.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route_b

    @property
    def feeder_a(self):
        return self._distant_point

    @property
    def feeder_b(self):
        return self._distant_point_b

    @property
    def r_vec(self):
        if hasattr(self.agent.central_complex, 'r_vec'):
            return self.agent.central_complex.r_vec
        else:
            return None


class NavigationSimulation(PathIntegrationSimulation):

    def __init__(self, routes, odours=None, *args, **kwargs):
        """
        Runs the path integration task.
        An agent equipped with a compass and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        route_a: np.ndarray[float]
            N x 4 matrix that holds the 3D position (x, y, z) and 1D orientation (yaw) for each iteration of the route
        agent: NavigatingAgent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world that creates the scenery disturbing the celestial compass or using the visual surroundings as a
            a compass. Default is None

        Other Parameters
        ----------------
        nb_iterations: int, optional
            the number of iterations to run the simulation. Default is 2.5 times the number of time-steps of the route
        name: str, optional
            the name of the simulation. Default is the name of the world or 'simulation'
        """

        agent = kwargs.get('agent', None)
        kwargs.setdefault('route', routes[0])
        kwargs.setdefault('nb_iterations', int(np.sum([5 * route.shape[0] for route in routes])))
        super().__init__(*args, **kwargs)

        self._route = routes
        self._distant_point = [route[-1, :3] for route in routes]

        if agent is None:
            self._agent = agent = NavigatingAgent(nb_feeders=len(routes), speed=.01, rng=self.rng, noise=self._noise)

        if odours is None:
            # add odour around the nest
            odours = [StaticOdour(centre=routes[0][0, :3], spread=1.)]
            for route in routes:
                # add odour around the food sources
                odours.append(StaticOdour(centre=route[-1, :3], spread=1.))
        self._odours = odours

        self._antennas = agent.sensors[1]
        self._mb = agent.brain[2]

        self._iter_offset = np.zeros(len(routes), dtype=int)
        self._iter_offset[1:] = -1
        self._current_route_id = 0
        self._food = np.zeros(len(odours), dtype=self.agent.dtype)
        self._food_supply = np.ones(len(routes), dtype=int)
        self._learning_phase = True

    def reset(self):
        """
        Resets the agent anf the logged statistics.
        """
        super().reset()

        self._iter_offset = np.zeros(len(self._route), dtype=int)
        self._iter_offset[1:] = -1
        self._current_route_id = 0
        self._food = np.zeros(len(self._odours), dtype=self.agent.dtype)
        self._food_supply = np.ones(len(self._route), dtype=int)
        self._food_supply[0] = 3
        self._learning_phase = True

    def init_stats(self):
        super().init_stats()

        for i in range(len(self._route)):
            self._stats[f"L_{i}"] = []

        self._stats["MBON"] = []
        self._stats["DAN"] = []
        self._stats["KC"] = []
        self._stats["PN"] = []
        self._stats["US"] = []

    def update_stats(self, a):
        """
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: NavigatingAgent
        """
        super().update_stats(a)

        self._stats[f"L_{self._current_route_id}"].append(self.d_distant_point)

        self._stats["MBON"].append(a.mushroom_body.r_mbon[0].copy())
        self._stats["DAN"].append(a.mushroom_body.r_dan[0].copy())
        self._stats["KC"].append(a.mushroom_body.r_kc[0].copy())
        self._stats["PN"].append(a.mushroom_body.r_cs[0].copy())
        self._stats["US"].append(a.mushroom_body.r_us[0].copy())

    def init_outbound(self):
        self.init_outbound_stats()

    def init_inbound_stats(self):
        self.__init_bound_stats("out")

    def init_outbound_stats(self):
        self.__init_bound_stats("in")

    def __init_bound_stats(self, inout="out"):
        i = 0
        while f"xyz_{inout}_{i}" in self._stats:
            i += 1

        if f"xyz_{inout}_{i}" not in self._stats:
            self._stats[f"xyz_{inout}_{i}"] = []
            self._stats[f"L_{inout}_{i}"] = []
            self._stats[f"C_{inout}_{i}"] = []

        self._stats[f"xyz_{inout}_{i}"].extend(copy(self._stats["xyz"]))
        self._stats[f"L_{inout}_{i}"].extend(copy(self._stats["L"]))
        self._stats[f"C_{inout}_{i}"].extend(copy(self._stats["C"]))
        self._stats["xyz"] = []
        self._stats["L"] = []
        self._stats["C"] = []

    def _step(self, i):
        """
        Runs one iteration of the simulation. If the iteration is less than the maximum number of iterations in the
        route it forces the agent to follow the route, otherwise it lets the agent decide its actions.

        Parameters
        ----------
        i: int
            the iteration ID
        """
        reinforcement = np.zeros(self.agent.mushroom_body.nb_us, dtype=self._mb.dtype)

        if self._learning_phase:
            vectors = np.eye(self.agent.mushroom_body.nb_us)
            if self._foraging and i - self.i_offset < 20:
                reinforcement[:] = vectors[(self._current_route_id + 1) * 2]
            elif not self._foraging and i - self.i_offset < self.route.shape[0] + 20:
                reinforcement[:] = vectors[0]
        # elif self.agent.central_complex.v_change:
        #     reinforcement[::2] = self.agent.central_complex.r_vec

        if np.all(np.isclose(self._food, 0)):
            self._food[:] = np.eye(len(self._odours))[0]

        if self._foraging:
            act = self.forage()
        else:
            act = self.home()

        # if (len(self.stats["US"]) >= repeats and
        #     np.any(self.stats["US"][-1] > 0) and
        #     np.any(self.stats["US"][-1] != self.stats["US"][-repeats])):
        #     reinforcement[:] = self.stats["US"][-1]

        # self._food[:] = 0.
        self.agent(sky=self._sky, odours=self._odours, food=self._food, reinforcement=reinforcement,
                   act=act, callback=self.update_stats)

    def forage(self):
        i = self._iteration
        act = True

        # search for the first unprocessed route that meets the criteria
        route = self.route
        self._food[:] = np.eye(self.agent.nb_odours)[self._current_route_id % len(self.routes) + 1]

        if self.i_offset < 0:
            # if the route has not been processed yet, initialise the counting offset
            self.i_offset = i
        i_off = i - self.i_offset  # the local (route[i]) iteration

        in_route = i_off < route.shape[0]

        # if i_off == 0:  # give reinforcement at the start of a new route
        #     reinforcement[self._current_route_id + 2] = 1.

        # if the iteration falls in the range of this route load its position and orientation
        if in_route:
            x, y, z, yaw = route[i_off]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False

        elif self.is_approaching_distant(tol=0.1):
            self.init_inbound()
            self._foraging = False

            # pick up the food if it's there and go home
            stock = self._food_supply[self._current_route_id]
            if stock > 0 and self._food[0] < 1:
                # change the food state
                self._food[:] = np.eye(self.agent.nb_odours)[0]

                # deduct the food from the feeder
                self._food_supply[self._current_route_id] = self._food_supply[self._current_route_id] - 1
            elif stock == 0:
                # if there is no food in the feeder move to the next feeder
                self._current_route_id = (self._current_route_id + 1) % len(self.routes)
                self._food[:] = np.eye(self.agent.nb_odours)[self._current_route_id + 1]

            if np.isclose(self._food.sum(), 0):
                self._food[1] = 1.  # continue searching towards the first source
                self._foraging = True
            else:
                print(f"START PI FROM ROUTE {self._current_route_id + 1}")
        elif self.is_approaching_distant(tol=0.5):
            self.approach_point(self.distant_point)
            act = False

        return act

    def home(self):
        act = True

        if self.is_approaching_central(tol=0.1):
            # if the agent has moved for more than 1 meter and is less than 5 cm away from the nest
            # approach the nest

            # self._agent.xyz = self._route_a[0, :3]
            self.init_outbound()
            self._foraging = True

            self._current_route_id += 1
            if self._current_route_id >= len(self.routes):
                self._learning_phase = False
                self._current_route_id = self._current_route_id % len(self.routes)
            self._food[:] = np.eye(self.agent.nb_odours)[self._current_route_id % len(self.routes) + 1]
            self.agent.central_complex.reset_integrator()

            print("START FORAGING!")

        elif self.is_approaching_central(tol=0.5):
            # if the agent has moved for more than 1 meter and is less than 50 cm away from the nest
            # approach the nest
            self.approach_point(self.central_point)
            act = False

        return act

    def is_approaching_central(self, tol=0.):
        return len(self.stats["L"]) > 1 and self.stats["L"][-2] < self.stats["L"][-1] < tol

    def is_approaching_distant(self, tol=0.):
        return (len(self.stats[f"L_{self._current_route_id}"]) > 1 and
                self.stats[f"L_{self._current_route_id}"][-2] < self.stats[f"L_{self._current_route_id}"][-1] < tol)

    def message(self):
        mbon = np.argmax([self.stats['MBON'][-1][0::2].mean(), self.stats['MBON'][-1][1::2].mean()]) + 1
        mbon = 0 if np.all(np.isclose(np.diff(self._stats['MBON'][-1]), 0)) else mbon
        pn = 0 if np.all(np.isclose(np.diff(self._stats['PN'][-1]), 0)) else (np.argmax(self.stats['PN'][-1]) + 1)
        us = 0 if np.all(np.isclose(np.diff(self._stats['US'][-1]), 0)) else (np.argmax(self.stats['US'][-1]) + 1)

        message = super().message().replace("- L", f"- mot: {mbon:d}, CS: {pn:d}, US: {us:d} - L")
        message += f" - route: {self._current_route_id + 1:d}{' - foraging' if self._foraging else ''}"
        return message

    @property
    def agent(self):
        """
        The agent of the simulation.

        Returns
        -------
        NavigatingAgent
        """
        return self._agent

    @property
    def routes(self):
        """
        N x 4 array representing the route that the agent follows to food source A before returning to its initial
        position.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route

    @property
    def route(self):
        return self._route[self._current_route_id]

    @property
    def distant_point(self):
        return self._distant_point[self._current_route_id]

    @property
    def d_distant_point(self):
        return self.distance_from(self.distant_point)

    @property
    def odours(self):
        """
        A list with all the odours in the simulation.

        Returns
        -------
        list[StaticOdour]
        """
        return self._odours

    @property
    def i_offset(self):
        return self._iter_offset[self._current_route_id]

    @i_offset.setter
    def i_offset(self, v):
        self._iter_offset[self._current_route_id] = v

    @property
    def food_supply(self):
        """
        The number of crumbs left in each food source.

        Returns
        -------
        np.ndarray[int]
        """
        return self._food_supply

    @property
    def r_mbon(self):
        return self.agent.mushroom_body.r_mbon[0].T.flatten()

    @property
    def r_dan(self):
        return self.agent.mushroom_body.r_dan[0].T.flatten()

    @property
    def r_kc(self):
        return self.agent.mushroom_body.r_kc[0].T.flatten()

    @property
    def r_pn(self):
        return self.agent.mushroom_body.r_cs[0].T.flatten()

    @property
    def r_us(self):
        return self.agent.mushroom_body.r_us[0].T.flatten()


class VisualNavigationSimulation(NavigationSimulationBase):

    def __init__(self, route, agent=None, nb_ommatidia=None, nb_scans=121, saturation=5.,
                 calibrate=False, frequency=False, free_motion=True, **kwargs):
        """
        Runs the route following task for an autonomous agent, by using entirely its vision. First it forces the agent
        to run through a predefined route. Then it places the agent back at the beginning and lets it autonomously reach
        the goal destination by following the exact same route.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that contains the 3D positions and 1D orientation (yaw) of the agent for the route it has to
            follow
        agent: VisualNavigationAgent, optional
            the agent that contains the compound eye and the memory component. Default is the an agent with an eye of
            nb_ommatidia ommatidia, sensitive to green and 15 degrees acceptance angle
        sky: Sky, optional
            the sky model. Default is a sky with the sun in the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world where the vegetation will be captured from. Default is the Seville ant world
        nb_ommatidia: int, optional
            the number of ommatidia for the default agent. If the agent is explicitly set, this attribute is not used.
            Default is None, which results in the default eye for the agent
        nb_scans: int, optional
            the number of scans the default agent will do trying to find the most familiar scene. Default is 7
        calibrate: bool, optional
            if True, the agent calibrate its eye using PCA whitening, by collecting 32 samples in a radius of 2 meters
            around the nest, and uses this as an input to its memory component. If False, the raw responses of the
            photo-receptors are used instead. Default is False
        frequency: bool, optional
            if True, the frequency domain is used an input to the memory of the agent. The raw photo-receptor responses
            are decomposed using the DCT algorithm. Default is False
        free_motion: bool, optional
            if True, the agent is let free to find its way to the goal after the training. If False, it is automatically
            brought back on the route when it deviated for more than 10 cm from it and this even is logged. Default is
            True

        Other Parameters
        ----------------
        nb_iterations: int, optional
            number of iterations that the simulation will run. Default is 2.1 time the iterations needed to complete
            the route
        name: str, optional
            the name of the simulation. Default is `vn-simulation`
        """
        kwargs.setdefault('nb_iterations', int(2.1 * route.shape[0]))
        kwargs.setdefault('name', 'vn-simulation')
        kwargs.setdefault('sky', UniformSky(luminance=10.))

        self._route = route

        if agent is None:
            eye = None
            if nb_ommatidia is not None:
                eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                                  omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])
            agent = VisualNavigationAgent(eye=eye, saturation=saturation, zernike=frequency, nb_scans=nb_scans,
                                          speed=0.01)

        super().__init__(agent, **kwargs)

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]
        self._stats += {
            "L": [],  # straight distance from the nest
            "C": [],  # distance towards the nest that the agent has covered
        }

        self._calibrate = calibrate
        self._free_motion = free_motion
        self._inbound = True
        self._outbound = True

    def reset(self):
        """
        Initialises the logged statistics and iteration count, calibrates the eye of agent if applicable and places it
        to the beginning of the route.

        Returns
        -------
        np.ndarray[float]
            array of the 3D positions of the samples used for the calibration
        """
        super().reset()
        xyzs = self.calibration()

        return xyzs

    def calibration(self):
        xyzs = None
        # the number of samples must be at least the same number as the dimensions of the input
        nb_samples = self.eye.nb_ommatidia

        if self._calibrate and not self.agent.is_calibrated:
            self.agent.xyz = self._route[-1, :3]
            self.agent.ori = R.from_euler('Z', self._route[-1, 3], degrees=True)
            self.agent.update = False
            xyzs, _ = self.agent.calibrate(self._sky, self._world, nb_samples=nb_samples, radius=2.)

        self.agent.xyz = self._route[0, :3]
        self.agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self.agent.update = True

        return xyzs

    def init_stats(self):
        super().init_stats()

        self._stats["ommatidia"] = []
        self._stats["input_layer"] = []
        self._stats["hidden_layer"] = []
        self._stats["output_layer"] = []
        self._stats["L"] = []  # straight distance from the nest
        self._stats["C"] = []  # distance that the agent has covered
        self._stats["capacity"] = []
        self._stats["familiarity"] = []

    def init_inbound(self):
        """
        Prepares the simulation for the second phase (inbound) where the agent will try to follow the learnt route.
        Sets new labels to the logged statistics and erases the current labels, which will be used to store the
        produced values.
        """
        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = False

        self.init_stats_inbound()

    def init_stats_inbound(self):
        # create a separate line
        self._stats["xyz_out"] = self._stats["xyz"]
        self._stats["L_out"] = self._stats["L"]
        self._stats["C_out"] = self._stats["C"]
        self._stats["capacity_out"] = self._stats["capacity"]
        self._stats["familiarity_out"] = self._stats["familiarity"]
        self._stats["xyz"] = []
        self._stats["L"] = []
        self._stats["C"] = []
        self._stats["capacity"] = []
        self._stats["familiarity"] = []
        self._stats["replace"] = []

    def _step(self, i):
        """
        Runs the iterations of the simulation. If the iteration ID exists in the route, it runs steps for the outbound
        path. If it is the end of the outbound path, it initialises the inbound and then runs the inbound steps. In case
        of the restrained motion, it prints '~REPLACE~' every time that the agent is brought back to the route.

        Parameters
        ----------
        i: int
            the iteration ID to run
        """
        if i == self._route.shape[0]:  # initialise route following
            self.init_inbound()

        if self.has_outbound and i < self._route.shape[0]:  # outbound path
            x, y, z, yaw = self._route[i]
            self._agent(sky=self._sky, scene=self._world, act=False, callback=self.update_stats)
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)

        elif self.has_inbound:  # inbound path
            act = not (len(self._stats["L"]) > 0 and self._stats["L"][-1] <= 0.01)
            self._agent(sky=self._sky, scene=self._world, act=act, callback=self.update_stats)
            if not act:
                self._agent.rotate(R.from_euler('Z', 1, degrees=True))
            if not self._free_motion and "replace" in self._stats:
                d_route = np.linalg.norm(self._route[:, :3] - self._agent.xyz, axis=1)
                point = np.argmin(d_route)
                if d_route[point] > 0.1:  # move for more than 10cm away from the route
                    self._agent.xyz = self._route[point, :3]
                    self._agent.ori = R.from_euler('Z', self._route[point, 3], degrees=True)
                    self._stats["replace"].append(True)
                    print(" ~ REPLACE ~")
                else:
                    self._stats["replace"].append(False)

    def update_stats(self, a):
        """
        Logs the current internal values of the agent.

        Parameters
        ----------
        a: VisualNavigationAgent
            the internal agent
        """

        super().update_stats(a)

        self._stats["ommatidia"].append(self.eye.responses.copy())
        self._stats["input_layer"].append(self.mem.r_inp[0].copy())
        self._stats["hidden_layer"].append(self.mem.r_hid[0].copy())
        self._stats["output_layer"].append(self.mem.r_out[0].copy())
        self._stats["familiarity"].append(self.familiarity)
        self._stats["capacity"].append(self.mem.free_space)
        c = self._stats["C"][-1] if len(self._stats["C"]) > 0 else 0.
        if len(self._stats["xyz"]) > 1:
            step = np.linalg.norm(np.array(self._stats["xyz"][-1])[:3] - np.array(self._stats["xyz"][-2])[:3])
        else:
            step = 0.
        self._stats["C"].append(c + step)

    def message(self):
        fam = self.familiarity
        if self.frame > 1:
            pn_diff = np.absolute(self._stats["input_layer"][-1] - self._stats["input_layer"][-2]).mean()
            kc_diff = np.absolute(self._stats["hidden_layer"][-1] - self._stats["hidden_layer"][-2]).mean()
        else:
            pn_diff = np.absolute(self.mem.r_inp[0]).mean()
            kc_diff = np.absolute(self.mem.r_hid[0]).mean()
        capacity = self.capacity
        d_nest = self.d_nest
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
        return (f"{super().message()}"
                f" - inp (change): {pn_diff * 100:.2f}%, hid (change): {kc_diff * 100:.2f}%,"
                f" familiarity: {fam * 100:.2f}%, capacity: {capacity * 100:.2f}%,"
                f" L: {d_nest:.2f}m, C: {d_trav:.2f}m")

    @property
    def agent(self):
        """
        The agent that runs in the simulation.

        Returns
        -------
        VisualNavigationAgent
        """
        return self._agent

    @property
    def route(self):
        """
        The route that the agent tries to follow.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route

    @property
    def eye(self):
        """
        The compound eye of the agent.

        Returns
        -------
        CompoundEye
        """
        return self._eye

    @property
    def mem(self):
        """
        The memory component of the agent.

        Returns
        -------
        MushroomBody
        """
        return self._mem

    @property
    def familiarity(self):
        """
        The maximum familiarity observed.

        Returns
        -------
        float
        """
        v = np.exp(-1j * np.deg2rad(self.agent.pref_angles))
        w = np.power(np.cos(self.agent.pref_angles) / 2 + .5, 4)
        return np.clip(np.sum(w * v * self.agent.familiarity / np.sum(w)).real, 0, 1)

    @property
    def capacity(self):
        """
        The percentage of unused memory left.

        Returns
        -------
        float
        """
        return self.mem.free_space

    @property
    def d_nest(self):
        """
        The distance (in meters) between the agent and the goal position (nest).

        Returns
        -------
        float
        """
        return (self._stats["L"][-1] if len(self._stats["L"]) > 0
                else np.linalg.norm(self._route[-1, :3] - self._route[0, :3]))

    @property
    def calibrate(self):
        """
        If calibration is set.

        Returns
        -------
        bool
        """
        return self._calibrate

    @property
    def free_motion(self):
        """
        If free motion is set.

        Returns
        -------
        bool
        """
        return self._free_motion

    @property
    def has_inbound(self):
        """
        Whether the agent will have a route-following phase.

        Returns
        -------
        bool
        """
        return self._inbound

    @has_inbound.setter
    def has_inbound(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._inbound = v

    @property
    def has_outbound(self):
        """
        Whether the agent will have a learning phase.

        Returns
        -------
        bool
        """
        return self._outbound

    @has_outbound.setter
    def has_outbound(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._outbound = v


class VisualFamiliarityDataCollectionSimulation(Simulation):

    def __init__(self, route, eye=None, sky=None, world=None, nb_ommatidia=None, saturation=5.,
                 nb_orientations=16, nb_rows=100, nb_cols=100, nb_parallel=21, disposition_step=0.02,
                 method="grid", **kwargs):
        """
        Simulation that collects data input (visual) and output (position and orientation) from a world.

        It stores the ommatidia responses and agent positions during a route and over a fixed amount of positions and
        orientations of the agent, uniformly distributed in the world (grid).

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that contains the 3D positions and 1D orientation (yaw) of the agent for the route it has to
            follow
        eye: CompoundEye, optional
            the compound eye that renders the visual input. Default is an eye of nb_ommatidia ommatidia, sensitive to
            green and 15 degrees acceptance angle
        sky: Sky, optional
            the sky model. Default is a sky with the sun in the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world where the vegetation will be captured from. Default is the Seville ant world
        nb_ommatidia: int, optional
            the number of ommatidia for the default agent. If the agent is explicitly set, this attribute is not used.
            Default is None, which results in the default eye for the agent
        saturation: float, optional
            the sensitivity in light. High saturation denotes high sensitivity in light ('burns' the visual input).
            Default is 5
        nb_orientations: int, optional
            the number of fixed orientation for the world's grid. Default is 16
        nb_rows: int, optional
            the number of rows of the world's grid. Default is 100
        nb_cols: int, optional
            the number of columns for the world's grid. Default is 100

        Other Parameters
        ----------------
        name: str, optional
            the name of the simulation. Default is `vn-simulation`
        """
        if method == "grid":
            kwargs.setdefault('nb_iterations', int(route.shape[0]) + nb_orientations * nb_rows * nb_cols)
        elif method == "parallel":
            kwargs.setdefault('nb_iterations', int(route.shape[0]) * (nb_orientations * nb_parallel + 1))
        kwargs.setdefault('name', 'vn-simulation')
        super().__init__(**kwargs)

        self._route = route

        if eye is None:
            if nb_ommatidia is None:
                nb_ommatidia = 1000
            eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                              omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])

        if sky is None:
            sky = UniformSky(luminance=10.)
        self._sky = sky
        self._world = world

        self._eye = eye

        self._has_grid = False
        self._has_parallel_routes = False
        if method == "grid":
            self._has_grid = True
        elif method == "parallel":
            self._has_parallel_routes = True
        self._outbound = True
        self._dump_map = np.empty((nb_rows, nb_cols, nb_orientations))
        self.__nb_cols = nb_cols
        self.__nb_rows = nb_rows
        self.__nb_oris = nb_orientations
        self.__ndindex = [index for index in np.ndindex(self._dump_map.shape[:3])]

        route_length = self._route[-1, :2] - self._route[0, :2]
        norm_length = route_length / np.linalg.norm(route_length)
        self._disposition = np.array([-norm_length[1], norm_length[0]])
        self._disposition_step = disposition_step

        self._stats = {
            "ommatidia": [],
            "xyz": []
        }

    def reset(self):
        """
        Initialises the logged statistics and iteration count, and places the eye at the beginning of the route.
        """
        self._stats["ommatidia"] = []
        self._stats["xyz"] = []

        self._iteration = 0

        self._eye._xyz = self._route[0, :3]
        self._eye._ori = R.from_euler('Z', self._route[0, 3], degrees=True)

    def init_inbound(self):
        """
        Prepares the simulation for the second phase (building the grid) where the eye will render visual input from
        predefined positions and orientations.
        Sets new labels to the logged statistics and erases the current labels, which will be used to store the
        produced values.
        """
        self._eye.xyz = self._route[0, :3]
        self._eye.ori = R.from_euler('Z', self._route[0, 3], degrees=True)

        # create a separate line
        self._stats["xyz_out"] = copy(self._stats["xyz"])
        self._stats["ommatidia_out"] = copy(self._stats["ommatidia"])
        self._stats["xyz"] = []
        self._stats["ommatidia"] = []

    def _step(self, i):
        """
        Runs the iterations of the simulation. If the iteration ID exists in the route, it runs steps for the outbound
        path. If it is the end of the outbound path, it initialises and runs the grid phase.

        Parameters
        ----------
        i: int
            the iteration ID to run
        """
        if i == self._route.shape[0]:  # initialise route following
            self.init_inbound()

        if self.has_outbound and i < self._route.shape[0]:  # outbound path
            x, y, z, yaw = self._route[i]
        elif self.has_grid:  # build the map
            j = i - self._route.shape[0] * int(self.has_outbound)
            row, col, ori = self.__ndindex[j]
            x = col2x(col, nb_cols=self.nb_cols, max_meters=10.)
            y = row2y(row, nb_rows=self.nb_rows, max_meters=10.)
            z = self._eye.z
            yaw = ori2yaw(ori, nb_oris=self.nb_orientations, degrees=True)
        elif self.has_parallel_routes:  # build parallel routes
            j = i - self._route.shape[0] * int(self.has_outbound)  # overall iteration of map
            m = (j // self.nb_orientations) % self._route.shape[0]  # iteration of position on the route
            k = j // (self._route.shape[0] * self.nb_orientations)  # disposition iteration
            ori = j % self.nb_orientations  # rotation iteration

            # calculate the disposition vector
            r = self._disposition_step * float((k % 2 - (k + 1) % 2) * ((k + 1) // 2))
            shift = r * self._disposition

            # calculate the orientation different
            d_yaw = ori2yaw(ori, nb_oris=self.nb_orientations, degrees=True)

            # apply the disposition to the route position
            x = self._route[m, 0] + shift[0]
            y = self._route[m, 1] + shift[1]
            z = self._route[m, 2]  # but not on the z axis

            # apply the rotation of the route orientation
            yaw = (self._route[m, 3] + d_yaw + 180) % 360 - 180
        else:
            return

        self._eye.xyz = [x, y, z]
        self._eye.ori = R.from_euler('Z', yaw, degrees=True)
        self._eye(sky=self._sky, scene=self._world, callback=self.update_stats)

    def update_stats(self, eye):
        """
        Logs the current position orientation and responses of the eye.

        Parameters
        ----------
        eye: CompoundEye
            the internal agent
        """

        assert eye == self._eye, "The input agent should be the same as the one used in the simulation!"

        self._stats["ommatidia"].append(self.eye.responses.copy())
        self._stats["xyz"].append([self._eye.x, self._eye.y, self._eye.z, self._eye.yaw_deg])

    def message(self):
        x, y, z = self._eye.xyz
        yaw = self._eye.yaw_deg

        if len(self._stats["ommatidia"]) > 1:
            omm_2, omm_1 = self._stats["ommatidia"][-2:]
            omm_diff = np.sqrt(np.square(omm_1 - omm_2).mean())
        else:
            omm_diff = 0.

        x_ext, y_ext, phi_ext = "", "", ""
        if self.has_grid:
            col = x2col(x, nb_cols=self.nb_cols, max_meters=10.)
            row = y2row(y, nb_rows=self.nb_rows, max_meters=10.)
            ori = yaw2ori(yaw, nb_oris=self.nb_orientations, degrees=True)

            x_ext = " (col: % 4d)" % col
            y_ext = " (row: % 4d)" % row
            phi_ext = " (ori: % 4d)" % ori
        elif self.has_parallel_routes:
            ori = yaw2ori(yaw, nb_oris=self.nb_orientations, degrees=True)
            i = (self._iteration - self._route.shape[0]) // self.nb_orientations
            if i >= 0:
                x_ext = " (x': %.2f)" % self._route[i % self._route.shape[0], 0]
                y_ext = " (y': %.2f)" % self._route[i % self._route.shape[0], 1]

            phi_ext = " (ori: % 4d)" % ori

        return (super().message() +
                " - x: %.2f%s, y: %.2f%s, z: %.2f, Φ: % 4d%s, omm (change): %.2f%%"
                ) % (x, x_ext, y, y_ext, z, yaw, phi_ext, omm_diff * 100)

    @property
    def world(self):
        """
        The world used for the simulation.

        Returns
        -------
        Seville2009
        """
        return self._world

    @property
    def route(self):
        """
        The route that the eye follows during the outbound (training).

        Returns
        -------
        np.ndarray[float]
        """
        return self._route

    @property
    def eye(self):
        """
        The compound eye.

        Returns
        -------
        CompoundEye
        """
        return self._eye

    @property
    def nb_cols(self):
        """
        The number of columns of the grid.

        Returns
        -------
        int
        """
        return self._dump_map.shape[0]

    @property
    def nb_rows(self):
        """
        The number of rows of the grid.

        Returns
        -------
        int
        """
        return self._dump_map.shape[1]

    @property
    def nb_orientations(self):
        """
        The number of orientation to render per cell in the grid.

        Returns
        -------
        int
        """
        return self._dump_map.shape[2]

    @property
    def disposition_step(self):
        """
        The step size of the parallel disposition in meters.

        Returns
        -------
        float
        """
        return self._disposition_step

    @property
    def has_grid(self):
        """
        Whether the simulation will run a grid (test) phase.

        Returns
        -------
        bool
        """
        return self._has_grid

    @has_grid.setter
    def has_grid(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._has_grid = v

    @property
    def has_parallel_routes(self):
        """
        Whether the simulation will run a pallelel-route (test) phase.

        Returns
        -------
        bool
        """
        return self._has_parallel_routes

    @has_parallel_routes.setter
    def has_parallel_routes(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._has_parallel_routes = v

    @property
    def has_outbound(self):
        """
        Whether the simulation will render visual input for an outbound (training) phase.

        Returns
        -------
        bool
        """
        return self._outbound

    @has_outbound.setter
    def has_outbound(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._outbound = v


class VisualFamiliarityParallelExplorationSimulation(Simulation):

    def __init__(self, data, nb_par, nb_oris, agent=None, calibrate=False, saturation=5., pre_training=False, **kwargs):
        """
        Runs the route following task for an autonomous agent, by using entirely its vision. First it forces the agent
        to run through a predefined route. Then it places the agent back at the beginning and lets it autonomously reach
        the goal destination by following the exact same route.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that contains the 3D positions and 1D orientation (yaw) of the agent for the route it has to
            follow
        agent: VisualNavigationAgent, optional
            the agent that contains the compound eye and the memory component. Default is the an agent with an eye of
            nb_ommatidia ommatidia, sensitive to green and 15 degrees acceptance angle
        calibrate: bool, optional
            if True, the agent calibrate its eye using PCA whitening, by collecting 32 samples in a radius of 2 meters
            around the nest, and uses this as an input to its memory component. If False, the raw responses of the
            photo-receptors are used instead. Default is False

        Other Parameters
        ----------------
        nb_iterations: int, optional
            number of iterations that the simulation will run. Default is 2.1 time the iterations needed to complete
            the route
        name: str, optional
            the name of the simulation. Default is `vn-simulation`
        """

        if isinstance(data, str):
            data = np.load(os.path.join(__stat_dir__, data))

        views_route = data["ommatidia_out"]
        if "xyz" in data:
            xyz = "xyz"
        elif "path" in data:
            xyz = "path"
        elif "position" in data:
            xyz = "position"
        elif "positions" in data:
            xyz = "positions"
        else:
            print([key for key in data.keys()])
            raise KeyError("'xyz' key could not be found in the data.")

        route = data[f"{xyz}_out"]
        views_par = data["ommatidia"]
        route_par = data[f"{xyz}"]

        kwargs.setdefault('nb_iterations', int(route.shape[0]) * (1 + int(pre_training)) + int(route_par.shape[0]))
        kwargs.setdefault('name', 'vfpe-simulation')
        super().__init__(**kwargs)

        nb_ommatidia = views_route.shape[1]

        if agent is None:
            eye = None
            if nb_ommatidia is not None:
                eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                                  omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])
            agent = VisualNavigationAgent(eye=eye, saturation=saturation, speed=0.01)
        self._agent = agent

        self._views = np.concatenate([views_route, views_par], axis=0)
        self._route = np.concatenate([route, route_par], axis=0)
        self._route_length = route.shape[0]

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]

        self._calibrate = calibrate
        self._outbound = True
        self._has_grid = True
        self._familiarity_par = np.zeros((route.shape[0], nb_par, nb_oris), dtype=agent.dtype)
        self.__nb_par = nb_par
        self.__nb_oris = nb_oris
        self.__ndindex = [index for index in np.ndindex(self._familiarity_par.shape[:3])]
        self.__pre_training = pre_training

        self._stats = {
            "familiarity_par": self._familiarity_par,
            "xyz": []
        }

    def reset(self):
        """
        Initialises the logged statistics and iteration count, calibrates the eye of agent if applicable and places it
        to the beginning of the route.

        Returns
        -------
        np.ndarray[float]
            array of the 3D positions of the samples used for the calibration
        """
        self._iteration = 0
        route_xyzs = self._route[:self._route_length, :3]
        d_nest = np.linalg.norm(self._route[:, :3] - self._route[self._route_length, :3], axis=1)
        xyzs = self._route[d_nest < 2., :3]
        i = self.agent.rng.permutation(np.arange(xyzs.shape[0]))[:self._views.shape[1]]
        xyzs = xyzs[i]
        if self._calibrate and not self._agent.is_calibrated:
            self._agent.xyz = self._route[self._route_length-1, :3]
            self._agent.ori = R.from_euler('Z', self._route[self._route_length-1, 3], degrees=True)
            self._agent.update = False
            self._agent.calibrate(omm_responses=self._views[i])

        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = True

        self._familiarity_par[:] = 0.

        self._stats["ommatidia"] = []
        self._stats["input_layer"] = []
        self._stats["hidden_layer"] = []
        self._stats["output_layer"] = []
        self._stats["capacity"] = []
        self._stats["familiarity"] = []
        self._stats["familiarity_par"] = self._familiarity_par
        self._stats["xyz"] = []

        return xyzs

    def init_grid(self):
        """
        Prepares the simulation for the second phase (inbound) where the agent will try to follow the learnt route.
        Sets new labels to the logged statistics and erases the current labels, which will be used to store the
        produced values.
        """
        self._agent.update = False

        # create a separate line
        self._stats["xyz_out"] = self._stats["xyz"]
        self._stats["capacity_out"] = self._stats["capacity"]
        self._stats["familiarity_out"] = self._stats["familiarity"]
        self._stats["capacity"] = []
        self._stats["familiarity"] = []
        self._stats["xyz"] = []

    def _step(self, i):
        """
        Runs the iterations of the simulation. If the iteration ID exists in the route, it runs steps for the outbound
        path. If it is the end of the outbound path, it initialises the inbound and then runs the inbound steps. In case
        of the restrained motion, it prints '~REPLACE~' every time that the agent is brought back to the route.

        Parameters
        ----------
        i: int
            the iteration ID to run
        """
        if self.__pre_training and i == self._route_length:
            self.reset()
        elif i == self._route_length:  # initialise route following
            self.init_grid()
        elif self.__pre_training and i == 2 * self._route_length:
            self.init_grid()

        x, y, z, yaw = self._route[i]

        self._agent.xyz = [x, y, z]
        self._agent.ori = R.from_euler('Z', yaw, degrees=True)
        self._agent(omm_responses=self._views[i], act=False, callback=self.update_stats)

        if self.has_grid and i >= self._route_length:
            j = i - self._route_length
            m = (j // self.nb_orientations) % self._route_length
            k = j // (self._route_length * self.nb_orientations)
            ori = j % self.nb_orientations

            self._familiarity_par[m, k, ori] = self.familiarity

    def update_stats(self, a):
        """
        Logs the current internal values of the agent.

        Parameters
        ----------
        a: VisualNavigationAgent
            the internal agent
        """

        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

        self._stats["ommatidia"].append(np.asarray(self._views[self._iteration], dtype=self.mem.dtype))
        self._stats["xyz"].append(np.asarray(self._route[self._iteration], dtype=self.mem.dtype))
        self._stats["input_layer"].append(self.mem.r_inp[0].copy())
        self._stats["hidden_layer"].append(self.mem.r_hid[0].copy())
        # if len(self._stats["hidden_layer"]) > 2:
        #     self._stats["hidden_layer"] = self._stats["hidden_layer"][-2:]
        self._stats["output_layer"].append(self.mem.r_out[0].copy())
        self._stats["capacity"].append(self.mem.free_space)
        self._stats["familiarity"].append(self.familiarity)

    def message(self):
        x, y, z = self._agent.xyz
        yaw = self._agent.yaw_deg
        fam = self.familiarity
        if self.frame > 1:
            pn_diff = np.absolute(self._stats["input_layer"][-1] - self._stats["input_layer"][-2]).mean()
            kc_diff = np.absolute(self._stats["hidden_layer"][-1] - self._stats["hidden_layer"][-2]).mean()
        else:
            pn_diff = np.absolute(self.mem.r_inp).mean()
            kc_diff = np.absolute(self.mem.r_hid).mean()
        capacity = self.capacity

        j = self._iteration - self._route_length
        m = (j // self.nb_orientations) % self._route_length
        x_, y_, _, yaw_ = self.route[m]

        # i = self._iteration - self._route_length * int(self.has_outbound)
        # if i < 0:
        #     row, col, ori = -1, -1, -1
        # else:
        #     row, col, ori = self.__ndindex[i]
        return (super().message() +
                " - x: %.2f (x': %.2f), y: %.2f (y': %.2f), z: %.2f, Φ: % 4d (Φ': % 4d)"
                " - input (change): %.2f%%, hidden (change): %.2f%%, familiarity: %.2f%%,"
                " capacity: %.2f%%") % (
            x, x_, y, y_, z, yaw, yaw_, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100.)

    @property
    def agent(self):
        """
        The agent that runs in the simulation.

        Returns
        -------
        VisualNavigationAgent
        """
        return self._agent

    @property
    def route(self):
        """
        The route that the agent tries to follow.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route[:self._route_length]

    @property
    def world(self):
        """
        The world used for the simulation.

        Returns
        -------
        None
        """
        return None

    @property
    def eye(self):
        """
        The compound eye of the agent.

        Returns
        -------
        CompoundEye
        """
        return self._eye

    @property
    def mem(self):
        """
        The memory component of the agent.

        Returns
        -------
        MemoryComponent
        """
        return self._mem

    @property
    def familiarity(self):
        """
        The maximum familiarity observed.

        Returns
        -------
        float
        """
        v = np.exp(-1j * np.deg2rad(self._agent.pref_angles))
        w = np.power(np.cos(self._agent.pref_angles) / 2 + .5, 4)
        return np.clip(np.absolute(np.sum(w * v * self._agent.familiarity / np.sum(w))), 0, 1)

    @property
    def capacity(self):
        """
        The percentage of unused memory left.

        Returns
        -------
        float
        """
        return self.mem.free_space

    @property
    def d_nest(self):
        """
        The distance (in meters) between the agent and the goal position (nest).

        Returns
        -------
        float
        """
        return (self._stats["L"][-1] if len(self._stats["L"]) > 0
                else np.linalg.norm(self._route[self._route_length-1, :3] - self._route[0, :3]))

    @property
    def calibrate(self):
        """
        If calibration is set.

        Returns
        -------
        bool
        """
        return self._calibrate

    @property
    def familiarity_par(self):
        return self._familiarity_par

    @property
    def nb_par(self):
        return self._familiarity_par.shape[1]

    @property
    def nb_orientations(self):
        return self._familiarity_par.shape[2]

    @property
    def has_grid(self):
        """
        Whether the agent will have a route-following phase.

        Returns
        -------
        bool
        """
        return self._has_grid

    @has_grid.setter
    def has_grid(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._has_grid = v

    @property
    def has_outbound(self):
        """
        Whether the agent will have a learning phase.

        Returns
        -------
        bool
        """
        return self._outbound

    @has_outbound.setter
    def has_outbound(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._outbound = v
