"""
Package that contains a number of different simulations.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertpy.brain.mushroombody import IncentiveCircuit
from invertsy.__helpers import __data__
from invertsy.env import UniformSky, Sky, Seville2009, WorldBase
from invertsy.agent import VisualNavigationAgent, PathIntegrationAgent

from invertpy.sense import CompoundEye
from invertpy.brain.memory import MemoryComponent

from scipy.spatial.transform import Rotation as R

import numpy as np

from time import time
from copy import copy

import os

__stat_dir__ = os.path.abspath(os.path.join(__data__, "animation", "stats"))
if not os.path.isdir(__stat_dir__):
    os.makedirs(__stat_dir__)


class Simulation(object):

    def __init__(self, nb_iterations, name="simulation"):
        """
        Abstract class that runs a simulation for a fixed number of iterations and logs statistics.

        Parameters
        ----------
        nb_iterations: int
            the number of iterations to run the simulation
        name: str, optional
            a unique name for the simulation. Default is 'simulation'
        """
        self._nb_iterations = nb_iterations
        self._iteration = 0
        self._stats = {}
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
        return "Simulation %d/%d" % (self._iteration + 1, self._nb_iterations)

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
    def name(self):
        """
        The name of the simulation.

        Returns
        -------
        str
        """
        return self._name


class RouteSimulation(Simulation):
    def __init__(self, route, eye=None, sky=None, world=None, *args, **kwargs):
        """
        Simulation that runs a predefined route in a world, given a sky and an eye model, and logs the input from the
        eye.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that holds the 3D position and 1D orientation (yaw) of the eye in every iteration.
        eye: CompoundEye, optional
            the compound eye model. Default is a green sensitive eye with 5000 ommatidia of 10 deg acceptance angle each
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

        self._r = self._eye(sky=self._sky, scene=self._world)

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


class VisualNavigationSimulation(Simulation):

    def __init__(self, route, agent=None, sky=None, world=None, nb_ommatidia=None, nb_scans=121,
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
        super().__init__(**kwargs)

        self._route = route

        if agent is None:
            saturation = 5.
            eye = None
            if nb_ommatidia is not None:
                eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                                  omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])
            agent = VisualNavigationAgent(eye=eye, saturation=saturation, freq_trans=frequency, nb_scans=nb_scans,
                                          speed=0.01)
        self._agent = agent

        if sky is None:
            sky = UniformSky(luminance=10.)
        self._sky = sky
        self._world = world

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]
        self._stats = {
            "path": [],
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
        self._stats["ommatidia"] = []
        self._stats["PN"] = []
        self._stats["KC"] = []
        self._stats["MBON"] = []
        self._stats["DAN"] = []
        self._stats["path"] = []
        self._stats["L"] = []  # straight distance from the nest
        self._stats["C"] = []  # distance that the agent has covered
        self._stats["capacity"] = []
        self._stats["familiarity"] = []

        self._iteration = 0
        xyzs = None

        if self._calibrate and not self._agent.is_calibrated:
            self._agent.xyz = self._route[-1, :3]
            self._agent.ori = R.from_euler('Z', self._route[-1, 3], degrees=True)
            self._agent.update = False
            xyzs, _ = self._agent.calibrate(self._sky, self._world, nb_samples=32, radius=2.)

        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = True

        return xyzs

    def init_inbound(self):
        """
        Prepares the simulation for the second phase (inbound) where the agent will try to follow the learnt route.
        Sets new labels to the logged statistics and erases the current labels, which will be used to store the
        produced values.
        """
        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = False

        # create a separate line
        self._stats["outbound"] = self._stats["path"]
        self._stats["L_out"] = self._stats["L"]
        self._stats["C_out"] = self._stats["C"]
        self._stats["capacity_out"] = self._stats["capacity"]
        self._stats["familiarity_out"] = self._stats["familiarity"]
        self._stats["path"] = []
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

        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

        self._stats["ommatidia"].append(self.eye.responses.copy())
        self._stats["PN"].append(self.mem.r_cs.copy())
        self._stats["KC"].append(self.mem.r_kc.copy())
        self._stats["MBON"].append(self.mem.r_mbon.copy())
        self._stats["DAN"].append(self.mem.r_dan.copy())
        self._stats["path"].append([self.agent.x, self.agent.y, self.agent.z, self.agent.yaw])
        self._stats["L"].append(np.linalg.norm(self.agent.xyz - self._route[-1, :3]))
        self._stats["capacity"].append(self.mem.free_space)
        self._stats["familiarity"].append(self.familiarity)
        c = self._stats["C"][-1] if len(self._stats["C"]) > 0 else 0.
        if len(self._stats["path"]) > 1:
            step = np.linalg.norm(np.array(self._stats["path"][-1])[:3] - np.array(self._stats["path"][-2])[:3])
        else:
            step = 0.
        self._stats["C"].append(c + step)

    def message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        fam = self.familiarity
        if self.frame > 1:
            pn_diff = np.absolute(self._stats["PN"][-1] - self._stats["PN"][-2]).mean()
            kc_diff = np.absolute(self._stats["KC"][-1] - self._stats["KC"][-2]).mean()
        else:
            pn_diff = np.absolute(self.mem.r_inp[0]).mean()
            kc_diff = np.absolute(self.mem.r_hid[0]).mean()
        capacity = self.capacity
        d_nest = self.d_nest
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
        return (super().message() +
                " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f"
                " - PN (change): %.2f%%, KC (change): %.2f%%, familiarity: %.2f%%,"
                " capacity: %.2f%%, L: %.2fm, C: %.2fm") % (
            x, y, z, phi, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100., d_nest, d_trav)

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
        fam_array = self._agent.familiarity
        return fam_array[len(fam_array) // 2] if self._iteration < self._route.shape[0] else fam_array.max()

    @property
    def capacity(self):
        """
        The percentage of unused memory left.

        Returns
        -------
        float
        """
        return np.clip(self.mem.w_k2m, 0, 1).mean()

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


class VisualFamiliaritySimulation(Simulation):

    def __init__(self, route, agent=None, sky=None, world=None, nb_ommatidia=None,
                 nb_orientations=8, nb_rows=100, nb_cols=100, calibrate=False, frequency=False, **kwargs):
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
        kwargs.setdefault('nb_iterations', int(route.shape[0]) + nb_orientations * nb_rows * nb_cols)
        kwargs.setdefault('name', 'vn-simulation')
        super().__init__(**kwargs)

        self._route = route

        if agent is None:
            saturation = 5.
            eye = None
            if nb_ommatidia is not None:
                eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                                  omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])
            agent = VisualNavigationAgent(eye=eye, saturation=saturation, freq_trans=frequency, speed=0.01)
        self._agent = agent

        if sky is None:
            sky = UniformSky(luminance=10.)
        self._sky = sky
        self._world = world

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]

        self._calibrate = calibrate
        self._has_map = True
        self._outbound = True
        self._familiarity_map = np.zeros((nb_rows, nb_cols, nb_orientations), dtype=agent.dtype)
        self.__nb_cols = nb_cols
        self.__nb_rows = nb_rows
        self.__nb_oris = nb_orientations

        self._stats = {
            "familiarity_map": self._familiarity_map,
            "positions": []
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
        self._stats["ommatidia"] = []
        self._stats["PN"] = []
        self._stats["KC"] = []
        self._stats["MBON"] = []
        self._stats["DAN"] = []
        self._stats["position"] = []
        self._stats["capacity"] = []
        self._stats["familiarity"] = []

        self._iteration = 0
        xyzs = None

        if self._calibrate and not self._agent.is_calibrated:
            self._agent.xyz = self._route[-1, :3]
            self._agent.ori = R.from_euler('Z', self._route[-1, 3], degrees=True)
            self._agent.update = False
            xyzs, _ = self._agent.calibrate(self._sky, self._world, nb_samples=32, radius=2.)

        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = True

        self._familiarity_map[:] = 0.

        return xyzs

    def init_inbound(self):
        """
        Prepares the simulation for the second phase (inbound) where the agent will try to follow the learnt route.
        Sets new labels to the logged statistics and erases the current labels, which will be used to store the
        produced values.
        """
        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = False

        # create a separate line
        self._stats["outbound"] = self._stats["position"]
        self._stats["capacity_out"] = self._stats["capacity"]
        self._stats["familiarity_out"] = self._stats["familiarity"]
        self._stats["position"] = []
        self._stats["capacity"] = []
        self._stats["familiarity"] = []

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

        elif self.has_map:  # build the map
            j = i - self._route.shape[0] * int(self.has_outbound)
            row, col, ori = [index for index in np.ndindex(self._familiarity_map.shape[:3])][j]
            x = col / self.nb_cols * 10.
            y = row / self.nb_rows * 10.
            yaw = ori / self.nb_orientations * 360.
            self._agent.xyz = [x, y, self.agent.z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            self._agent(sky=self._sky, scene=self._world, act=False, callback=self.update_stats)
            self._familiarity_map[row, col, ori] = self._stats["familiarity"][-1]

    def update_stats(self, a):
        """
        Logs the current internal values of the agent.

        Parameters
        ----------
        a: VisualNavigationAgent
            the internal agent
        """

        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

        self._stats["ommatidia"].append(self.eye.responses.copy())
        self._stats["PN"].append(self.mem.r_cs.copy())
        self._stats["KC"].append(self.mem.r_kc.copy())
        self._stats["MBON"].append(self.mem.r_mbon.copy())
        self._stats["DAN"].append(self.mem.r_dan.copy())
        self._stats["position"].append([self.agent.x, self.agent.y, self.agent.z, self.agent.yaw])
        self._stats["capacity"].append(np.clip(self.mem.w_k2m, 0, 1).mean())
        self._stats["familiarity"].append(self.familiarity)

    def message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        fam = self.familiarity
        if self.frame > 1:
            pn_diff = np.absolute(self._stats["PN"][-1] - self._stats["PN"][-2]).mean()
            kc_diff = np.absolute(self._stats["KC"][-1] - self._stats["KC"][-2]).mean()
        else:
            pn_diff = np.absolute(self.mem.r_cs[0]).mean()
            kc_diff = np.absolute(self.mem.r_kc[0]).mean()
        capacity = self.capacity
        i = self._iteration - self._route.shape[0] * int(self.has_outbound)
        if i < 0:
            col, row, ori = -1, -1, -1
        else:
            col, row, ori = [index for index in np.ndindex(self._familiarity_map.shape[:3])][i]
        return (super().message() +
                " - x: %.2f (row: % 4d), y: %.2f (col: % 4d), z: %.2f, Φ: % 4d (scan: % 4d)"
                " - PN (change): %.2f%%, KC (change): %.2f%%, familiarity: %.2f%%,"
                " capacity: %.2f%%") % (
            x, row, y, col, z, phi, ori, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100.)

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
        fam_array = self._agent.familiarity
        return fam_array[len(fam_array) // 2] if self._iteration < self._route.shape[0] else fam_array.max()

    @property
    def capacity(self):
        """
        The percentage of unused memory left.

        Returns
        -------
        float
        """
        return np.clip(self.mem.w_k2m, 0, 1).mean()

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
    def familiarity_map(self):
        return self._familiarity_map

    @property
    def nb_cols(self):
        return self._familiarity_map.shape[0]

    @property
    def nb_rows(self):
        return self._familiarity_map.shape[1]

    @property
    def nb_orientations(self):
        return self._familiarity_map.shape[2]

    @property
    def has_map(self):
        """
        Whether the agent will have a route-following phase.

        Returns
        -------
        bool
        """
        return self._has_map

    @has_map.setter
    def has_map(self, v):
        """
        Parameters
        ----------
        v: bool
        """
        self._has_map = v

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
                 nb_orientations=16, nb_rows=100, nb_cols=100, **kwargs):
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
        kwargs.setdefault('nb_iterations', int(route.shape[0]) + nb_orientations * nb_rows * nb_cols)
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

        self._has_grid = True
        self._outbound = True
        self._dump_map = np.empty((nb_rows, nb_cols, nb_orientations))
        self.__nb_cols = nb_cols
        self.__nb_rows = nb_rows
        self.__nb_oris = nb_orientations
        self.__ndindex = [index for index in np.ndindex(self._dump_map.shape[:3])]

        self._stats = {
            "ommatidia": [],
            "positions": []
        }

    def reset(self):
        """
        Initialises the logged statistics and iteration count, and places the eye at the beginning of the route.
        """
        self._stats["ommatidia"] = []
        self._stats["positions"] = []

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
        self._stats["positions_out"] = copy(self._stats["positions"])
        self._stats["ommatidia_out"] = copy(self._stats["ommatidia"])
        self._stats["positions"] = []
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
            x = self.col2x(col)
            y = self.row2y(row)
            z = self._eye.z
            yaw = self.ori2yaw(ori)
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
        self._stats["positions"].append([self._eye.x, self._eye.y, self._eye.z, self._eye.yaw])

    def message(self):
        x, y, z = self._eye.xyz
        phi = self._eye.yaw_deg
        i = self._iteration - self._route.shape[0] * int(self.has_outbound)
        if i < 0:
            col, row, ori = -1, -1, -1
        else:
            col, row, ori = self.__ndindex[i]

        if len(self._stats["ommatidia"]) > 1:
            omm_2, omm_1 = self._stats["ommatidia"][-2:]
            omm_diff = np.sqrt(np.square(omm_1 - omm_2).mean())
        else:
            omm_diff = 0.
        return (super().message() +
                " - x: %.2f (row: % 4d), y: %.2f (col: % 4d), z: %.2f, Φ: % 4d (scan: % 4d), omm (change): %.2f%%"
                ) % (x, row, y, col, z, phi, ori, omm_diff * 100)

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

    def col2x(self, col):
        """
        Transforms column number to the 'x' coordinate.

        Parameters
        ----------
        col : int
            the column number

        Returns
        -------
        float
            the 'x' coordinate
        """
        return col / self.nb_cols * 10

    def row2y(self, row):
        """
        Transforms row number to the 'y' coordinate.

        Parameters
        ----------
        row : int
            the row number

        Returns
        -------
        float
            the 'y' coordinate
        """
        return row / self.nb_rows * 10

    def ori2yaw(self, ori, degrees=True):
        """
        Transforms the orientation identity to the yaw direction.

        Parameters
        ----------
        ori : int
            the orientation identity
        degrees : bool
            whether we want the output to be in degrees

        Returns
        -------
        float
            the yaw direction
        """

        two_pi = 360 if degrees else (2 * np.pi)
        return ori / self.nb_orientations * two_pi


class VisualFamiliarityGridExplorationSimulation(Simulation):

    def __init__(self, data, nb_rows, nb_cols, nb_oris, agent=None, calibrate=False, saturation=5., **kwargs):
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
        route = data["positions_out"]
        views_grid = data["ommatidia"]
        grid = data["positions"]

        kwargs.setdefault('nb_iterations', int(route.shape[0]) + int(grid.shape[0]))
        kwargs.setdefault('name', 'vn-simulation')
        super().__init__(**kwargs)

        self._route = route
        nb_ommatidia = views_route.shape[1]

        if agent is None:
            eye = None
            if nb_ommatidia is not None:
                eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                                  omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])
            agent = VisualNavigationAgent(eye=eye, saturation=saturation, speed=0.01)
        self._agent = agent

        self._views = np.concatenate([views_route, views_grid], axis=0)
        self._route = np.concatenate([route, grid], axis=0)
        self._route_length = route.shape[0]

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]

        self._calibrate = calibrate
        self._outbound = True
        self._has_grid = True
        self._familiarity_map = np.zeros((nb_rows, nb_cols, nb_oris), dtype=agent.dtype)
        self.__nb_cols = nb_cols
        self.__nb_rows = nb_rows
        self.__nb_oris = nb_oris
        self.__ndindex = [index for index in np.ndindex(self._familiarity_map.shape[:3])]

        self._stats = {
            "familiarity_map": self._familiarity_map,
            "position": []
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
        d_nest = np.linalg.norm(route_xyzs - route_xyzs[-1], axis=1)
        xyzs = route_xyzs[d_nest < 2.]
        i = self.agent.rng.permutation(np.arange(xyzs.shape[0]))[:32]
        xyzs = xyzs[i]
        if self._calibrate and not self._agent.is_calibrated:
            self._agent.xyz = self._route[self._route_length-1, :3]
            self._agent.ori = R.from_euler('Z', self._route[self._route_length-1, 3], degrees=True)
            self._agent.update = False
            self._agent.calibrate(omm_responses=self._views[i])

        self._agent.xyz = self._route[0, :3]
        self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
        self._agent.update = True

        self._familiarity_map[:] = 0.

        self._stats["ommatidia"] = []
        self._stats["input_layer"] = []
        self._stats["hidden_layer"] = []
        self._stats["output_layer"] = []
        self._stats["capacity"] = []
        self._stats["familiarity"] = []
        self._stats["familiarity_map"] = self._familiarity_map
        self._stats["position"] = []

        return xyzs

    def init_grid(self):
        """
        Prepares the simulation for the second phase (inbound) where the agent will try to follow the learnt route.
        Sets new labels to the logged statistics and erases the current labels, which will be used to store the
        produced values.
        """
        self._agent.update = False

        # create a separate line
        self._stats["position_out"] = self._stats["position"]
        self._stats["capacity_out"] = self._stats["capacity"]
        self._stats["familiarity_out"] = self._stats["familiarity"]
        self._stats["capacity"] = []
        self._stats["familiarity"] = []
        self._stats["position"] = []

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
        row, col, ori = 0, 0, 0

        if i == self._route_length:  # initialise route following
            self.init_grid()

        if self.has_outbound and i < self._route_length:  # outbound path
            x, y, z, yaw = self._route[i]

        elif self.has_grid:  # build the map
            j = i - self._route_length * int(self.has_outbound)
            row, col, ori = self.__ndindex[j]
            x = self.col2x(col)
            y = self.row2y(row)
            z = self.agent.z
            yaw = self.ori2yaw(ori)
        else:
            return

        self._agent.xyz = [x, y, z]
        self._agent.ori = R.from_euler('Z', yaw, degrees=True)
        self._agent(omm_responses=self._views[i], act=False, callback=self.update_stats)

        if self.has_grid and i >= self._route_length:
            self._familiarity_map[row, col, ori] = self.familiarity

    def update_stats(self, a):
        """
        Logs the current internal values of the agent.

        Parameters
        ----------
        a: VisualNavigationAgent
            the internal agent
        """

        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

        self._stats["ommatidia"].append(self._views[self._iteration])
        self._stats["position"].append(self._route[self._iteration])
        self._stats["input_layer"].append(self.mem.r_inp.copy())
        self._stats["hidden_layer"].append(self.mem.r_hid.copy())
        self._stats["output_layer"].append(self.mem.r_out.copy())
        self._stats["capacity"].append(self.mem.free_space)
        self._stats["familiarity"].append(self.familiarity)

    def message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        fam = self.familiarity
        if self.frame > 1:
            pn_diff = np.absolute(self._stats["input_layer"][-1] - self._stats["input_layer"][-2]).mean()
            kc_diff = np.absolute(self._stats["hidden_layer"][-1] - self._stats["hidden_layer"][-2]).mean()
        else:
            pn_diff = np.absolute(self.mem.r_inp).mean()
            kc_diff = np.absolute(self.mem.r_hid).mean()
        capacity = self.capacity
        i = self._iteration - self._route_length * int(self.has_outbound)
        if i < 0:
            col, row, ori = -1, -1, -1
        else:
            col, row, ori = self.__ndindex[i]
        return (super().message() +
                " - x: %.2f (row: % 4d), y: %.2f (col: % 4d), z: %.2f, Φ: % 4d (scan: % 4d)"
                " - input (change): %.2f%%, hidden (change): %.2f%%, familiarity: %.2f%%,"
                " capacity: %.2f%%") % (
            x, row, y, col, z, phi, ori, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100.)

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
        fam_array = self._agent.familiarity
        return fam_array[len(fam_array) // 2] if self._iteration < self._route.shape[0] else fam_array.max()

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
    def familiarity_map(self):
        return self._familiarity_map

    @property
    def nb_cols(self):
        return self._familiarity_map.shape[0]

    @property
    def nb_rows(self):
        return self._familiarity_map.shape[1]

    @property
    def nb_orientations(self):
        return self._familiarity_map.shape[2]

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

    def col2x(self, col):
        """
        Transforms column number to the 'x' coordinate.

        Parameters
        ----------
        col : int
            the column number

        Returns
        -------
        float
            the 'x' coordinate
        """
        return col / self.nb_cols * 10

    def row2y(self, row):
        """
        Transforms row number to the 'y' coordinate.

        Parameters
        ----------
        row : int
            the row number

        Returns
        -------
        float
            the 'y' coordinate
        """
        return row / self.nb_rows * 10

    def ori2yaw(self, ori, degrees=True):
        """
        Transforms the orientation identity to the yaw direction.

        Parameters
        ----------
        ori : int
            the orientation identity
        degrees : bool
            whether we want the output to be in degrees

        Returns
        -------
        float
            the yaw direction
        """

        two_pi = 360 if degrees else (2 * np.pi)
        return ori / self.nb_orientations * two_pi

    def x2col(self, x):
        """
        Transforms the 'x' coordinate to the column number.

        Parameters
        ----------
        x : the 'x' coordinate

        Returns
        -------
        int :
            the column number
        """
        return int(x / 10 * self.nb_cols)

    def y2row(self, y):
        """
        Transforms the 'y' coordinate to the row number.

        Parameters
        ----------
        y : the 'y' coordinate

        Returns
        -------
        int :
            the row number
        """
        return int(y / 10 * self.nb_rows)

    def yaw2ori(self, yaw, degrees=True):
        """
        Transforms the 'yaw' direction to the orientation identity.

        Parameters
        ----------
        yaw : float
            the 'yaw' direction
        degrees : bool
            whether the input is in degrees

        Returns
        -------
        int :
            the orientation identity
        """
        two_pi = 360 if degrees else (2 * np.pi)
        return int(yaw / two_pi * self.nb_orientations)


class PathIntegrationSimulation(Simulation):

    def __init__(self, route, agent=None, sky=None, world=None, *args, **kwargs):
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
        name = kwargs.pop('name', None)
        kwargs.setdefault('nb_iterations', int(2.5 * route.shape[0]))
        super().__init__(*args, **kwargs)
        self._route = route

        if agent is None:
            agent = PathIntegrationAgent(speed=.01)
        self._agent = agent

        if sky is None:
            sky = Sky(30, 180, degrees=True)
        self._sky = sky
        self._world = world

        if name is None and world is not None:
            self._name = world.name

        self._compass_sensor = agent.sensors[0]
        self._compass_model, self._cx = agent.brain

    def reset(self):
        """
        Resets the agent anf the logged statistics.
        """
        self._stats["POL"] = []
        self._stats["SOL"] = []
        self._stats["TB1"] = []
        self._stats["CL1"] = []
        self._stats["CPU1"] = []
        self._stats["CPU4"] = []
        self._stats["CPU4mem"] = []
        self._stats["path"] = []
        self._stats["L"] = []
        self._stats["C"] = []

        self.agent.reset()

    def init_inbound(self):
        """
        Sets up the inbound phase.
        Changes the labels of the logged stats to their outbound equivalent and resets them for the new phase to come.
        """
        # create a separate line
        self._stats["outbound"] = self._stats["path"]
        self._stats["L_out"] = self._stats["L"]
        self._stats["C_out"] = self._stats["C"]
        self._stats["path"] = []
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
        self._iteration = i

        act = True
        if i < self._route.shape[0]:  # outbound
            x, y, z, yaw = self._route[i]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
        elif i == self.route.shape[0]:
            self.init_inbound()
        self._agent(sky=self._sky, act=act, callback=self.update_stats)

    def update_stats(self, a):
        """
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: PathIntegrationAgent
        """

        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

        compass, cx = a.brain
        self._stats["POL"].append(compass.r_pol.copy())
        self._stats["SOL"].append(compass.r_sol.copy())
        self._stats["TB1"].append(cx.r_tb1.copy())
        self._stats["CL1"].append(cx.r_cl1.copy())
        self._stats["CPU1"].append(cx.r_cpu1.copy())
        self._stats["CPU4"].append(cx.r_cpu4.copy())
        self._stats["CPU4mem"].append(cx.cpu4_mem.copy())
        self._stats["path"].append([a.x, a.y, a.z, a.yaw])
        self._stats["L"].append(np.linalg.norm(a.xyz - self._route[-1, :3]))
        c = self._stats["C"][-1] if len(self._stats["C"]) > 0 else 0.
        if len(self._stats["path"]) > 1:
            step = np.linalg.norm(np.array(self._stats["path"][-1])[:3] - np.array(self._stats["path"][-2])[:3])
        else:
            step = 0.
        self._stats["C"].append(c + step)

    def message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        d_nest = self.d_nest
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
        return (super().message() + " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f - L: %.2fm, C: %.2fm") % (
            x, y, z, phi, d_nest, d_trav)

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

    @property
    def d_nest(self):
        """
        The distance between the agent and the nest.

        Returns
        -------
        float
        """
        return (self._stats["L"][-1] if len(self._stats["L"]) > 0
                else np.linalg.norm(self._route[-1, :3] - self._route[0, :3]))

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
