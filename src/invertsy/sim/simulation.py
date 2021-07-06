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
from invertsy.__helpers import __data__, eps
from invertsy.env import Sky, Seville2009
from invertsy.agent import VisualNavigationAgent, PathIntegrationAgent
from invertsy.agent.agent import LandmarkIntegrationAgent

from invertpy.sense import CompoundEye
from invertpy.brain import MushroomBody
from invertpy.brain.compass import photoreceptor2pooling

from scipy.spatial.transform import Rotation as R

import numpy as np

from time import time

import os

__stat_dir__ = os.path.abspath(os.path.join(__data__, "animation", "stats"))


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
        world: Seville2009, optional
            the world where the route was captured. Default is the Seville ant world
        """
        kwargs.setdefault('nb_iterations', route.shape[0])
        name = kwargs.get('name', None)
        super().__init__(*args, **kwargs)

        self._route = route

        if eye is None:
            eye = CompoundEye(nb_input=5000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10), omm_res=5.,
                              c_sensitive=[0., 0., 1., 0., 0.])
        self._eye = eye

        if sky is None:
            sky = Sky(30, 180, degrees=True)
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
        world: Seville2009, optional
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
            sky = Sky(30, 180, degrees=True)
        self._sky = sky
        self._world = world

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]
        self._stats = {
            "path": [],
            "L": [],  # straight distance from the nest
            "C": [],  # distance towards the nest that the agent has covered
        }

        self._steering_tol = 2e-01
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
            steering = yaw - self._agent.ori.as_euler("ZYX", degrees=True)[0]
            if np.absolute(steering) < self._steering_tol:
                steering = 0.
            self._agent(sky=self._sky, scene=self._world, steering=steering, act=False, callback=self.update_stats)
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)

        elif self.has_inbound:  # inbound path
            act = not (len(self._stats["L"]) > 0 and self._stats["L"][-1] <= 0.01)
            self._agent(sky=self._sky, scene=self._world, steering=0., act=act, callback=self.update_stats)
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
        # the number of KCs not associated with any valence
        cap_per_kc = 1 - np.max(np.absolute(self.mem.w_k2m - self.mem.w_rest), axis=1)
        self._stats["capacity"].append(np.clip(cap_per_kc, 0, 1).mean())
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
            pn_diff = np.absolute(self.mem.r_cs[0]).mean()
            kc_diff = np.absolute(self.mem.r_kc[0]).mean()
        capacity = self.capacity
        d_nest = self.d_nest
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
        replaces = np.sum(self._stats["replace"]) if "replace" in self._stats else 0
        return (super().message() +
                " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f"
                " - PN (change): %.2f%%, KC (change): %.2f%%, familiarity: %.2f%%,"
                " capacity: %.2f%%, L: %.2fm, C: %.2fm, #replaces: %d") % (
            x, y, z, phi, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100., d_nest, d_trav, replaces)

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
        return fam_array[0] if self._iteration < self._route.shape[0] else fam_array.max()

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
        world: Seville2009, optional
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


class LandmarkIntegrationSimulation(Simulation):

    def __init__(self, route, agent=None, sky=None, world=None, calibrate=True, is_replacing=True,
                 path_integration=False, visual_navigation=True, *args, **kwargs):
        """
        Runs the landmark integration task.
        An agent equipped with a compound eye and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that holds the 3D position (x, y, z) and 1D orientation (yaw) for each iteration of the route
        agent: LandmarkIntegrationAgent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: Seville2009, optional
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
        kwargs.setdefault('nb_iterations', int(2.1 * route.shape[0]))
        super().__init__(*args, **kwargs)
        self._route = route

        if agent is None:
            agent = LandmarkIntegrationAgent(speed=.01)
        self._agent = agent

        if sky is None:
            sky = Sky(30, 180, degrees=True)
        self._sky = sky
        self._world = world

        if name is None and world is not None:
            self.set_name(world.name)

        self._eye = agent.sensors[0]
        self._compass, self._mb, self._cx = agent.brain

        self._calibrate = calibrate
        self._is_replacing = is_replacing
        self._inbound = True
        self._outbound = True

        self._path_integration = path_integration
        self._visual_navigation = visual_navigation

    def reset(self):
        """
        Resets the agent anf the logged statistics.
        """
        self._stats["ommatidia"] = []
        self._stats["POL"] = []
        self._stats["SOL"] = []
        self._stats["CMP"] = []
        self._stats["E-PG"] = []
        self._stats["P-EG"] = []
        self._stats["P-EN"] = []
        self._stats["PFL3"] = []
        self._stats["FsBN"] = []
        self._stats["Noduli"] = []
        self._stats["DNa2"] = []
        self._stats["KC"] = []
        self._stats["MBON"] = []
        self._stats["PN"] = []
        self._stats["path"] = []
        self._stats["L"] = []
        self._stats["C"] = []
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
        Sets up the inbound phase.
        Changes the labels of the logged stats to their outbound equivalent and resets them for the new phase to come.
        """

        if not self.does_path_integration:
            self._agent.xyz = self._route[0, :3]
            self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
            self._agent.update = False

        # create a separate line
        self._stats["outbound"] = self._stats["path"]
        self._stats["L_out"] = self._stats["L"]
        self._stats["C_out"] = self._stats["C"]
        self._stats["path"] = []
        self._stats["L"] = []
        self._stats["C"] = []

        # mushroom body
        self._stats["capacity_out"] = self._stats["capacity"]
        self._stats["familiarity_out"] = self._stats["familiarity"]
        self._stats["capacity"] = []
        self._stats["familiarity"] = []
        if not self.does_path_integration:
            self._stats["replace"] = []

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

        if self.has_inbound and i == self.route.shape[0]:
            self.init_inbound()

        act = True
        if self.has_outbound and i < self._route.shape[0]:  # outbound
            x, y, z, yaw = self._route[i]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
        elif self.has_inbound and not self.does_path_integration:
            act = not (len(self._stats["L"]) > 0 and self._stats["L"][-1] <= 0.01)

        tol = np.deg2rad(1e-01)
        nod = np.array([0., 0.], dtype=self.agent.dtype)
        if len(self.stats["path"]) > 1:
            curr = self.agent.yaw
            prev = self.stats["path"][-2][-1]
            if (curr - prev + np.pi) % (2 * np.pi) - np.pi < -tol:
                nod = np.array([1., 0.], dtype=self.agent.dtype)
            elif (curr - prev + np.pi) % (2 * np.pi) - np.pi > tol:
                nod = np.array([0., 1.], dtype=self.agent.dtype)
            print(len(self.stats["path"]), "Noduli", nod, (curr - prev + np.pi) % (2 * np.pi) - np.pi, act)
        self._agent(sky=self._sky, scene=self._world, flow=nod, act=act, callback=self.update_stats)

        if not self.does_path_integration and not act:
            self._agent.rotate(R.from_euler('Z', 1, degrees=True))
        elif self._is_replacing and "replace" in self._stats:
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
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: LandmarkIntegrationAgent
        """

        assert a == self.agent, "The input agent should be the same as the one used in the simulation!"

        eye = a.sensors[0]  # type: CompoundEye
        compass, mb, cx = a.brain
        self._stats["ommatidia"].append(eye.responses.copy())
        self._stats["POL"].append(compass.r_pol.copy())
        self._stats["SOL"].append(compass.r_sol.copy())
        self._stats["CMP"].append(cx.r_cmp.copy())
        self._stats["PN"].append(mb.r_cs[0].flatten().copy())
        self._stats["KC"].append(mb.r_kc[0].flatten().copy())
        self._stats["MBON"].append(mb.r_mbon[0].flatten().copy())
        self._stats["E-PG"].append(cx.r_epg.copy())
        self._stats["P-EG"].append(cx.r_peg.copy())
        self._stats["P-EN"].append(cx.r_pen.copy())
        self._stats["PFL3"].append(cx.r_pfl3.copy())
        self._stats["FsBN"].append(cx.r_fbn.copy())
        self._stats["Noduli"].append(cx.r_nod.copy())
        self._stats["DNa2"].append(cx.r_dna2.copy())

        # the number of KCs not associated with any valence
        cap_per_kc = 1 - np.max(np.absolute(self._mb.w_k2m - self._mb.w_rest), axis=1)
        self._stats["capacity"].append(np.clip(cap_per_kc, 0, 1).mean())
        self._stats["familiarity"].append(self.familiarity)

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
        fam = self.familiarity
        if self.frame > 1:
            z_pn = np.maximum((self._stats["PN"][-1] + self._stats["PN"][-2] > 0).sum(), eps)
            z_kc = np.maximum((self._stats["KC"][-1] + self._stats["KC"][-2] > 0).sum(), eps)
            pn_diff = np.absolute(self._stats["PN"][-1] - self._stats["PN"][-2]).sum() / z_pn
            kc_diff = np.absolute(self._stats["KC"][-1] - self._stats["KC"][-2]).sum() / z_kc
        else:
            pn_diff = np.absolute(self.mb.r_cs[0]).mean()
            kc_diff = np.absolute(self.mb.r_kc[0]).mean()
        capacity = self.capacity
        d_nest = self.d_nest
        d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
                  else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
        replaces = np.sum(self._stats["replace"]) if "replace" in self._stats else 0
        return (super().message() +
                " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f"
                " - PN (change): %.2f%%, KC (change): %.2f%%, familiarity: %.2f%%,"
                " capacity: %.2f%%, L: %.2fm, C: %.2fm, #replaces: %d") % (
            x, y, z, phi, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100., d_nest, d_trav, replaces)

    @property
    def agent(self):
        """
        The agent of the simulation.

        Returns
        -------
        LandmarkIntegrationAgent
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
    def eye(self):
        """
        The compound eye of the agent.

        Returns
        -------
        CompoundEye
        """
        return self._eye

    @property
    def mb(self):
        """
        The memory component of the agent.

        Returns
        -------
        MushroomBody
        """
        return self._mb

    @property
    def cx(self):
        """
        The path integration component of the agent.

        Returns
        -------
        FlyCentralComplex
        """
        return self._cx

    @property
    def familiarity(self):
        """
        The maximum familiarity observed.

        Returns
        -------
        float
        """
        fam_array = self._agent.familiarity
        return fam_array[0] if self._iteration < self._route.shape[0] else fam_array.max()

    @property
    def capacity(self):
        """
        The percentage of unused memory left.

        Returns
        -------
        float
        """
        return 1. - np.clip(np.absolute(1. - self.mb.w_k2m), 0, 1).sum() / self.mb.nb_kc

    @property
    def is_replacing(self):
        """
        If the agents position is replaced when it moves far away from the original route.

        Returns
        -------
        bool
        """
        return self._replacing

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

    @property
    def does_path_integration(self):
        return self._path_integration

    @property
    def does_visual_navigation(self):
        return self._visual_navigation

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
    def r_omm(self):
        """
        The photoreceptor responses of the compound eye of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return photoreceptor2pooling(self._eye.responses).T.flatten()

    @property
    def r_pol(self):
        """
        The POL responses of the compass model of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._compass.r_pol.T.flatten()

    @property
    def r_tcl(self):
        """
        The TCL responses of the compass model of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._compass.r_tcl.T.flatten()

    @property
    def r_epg(self):
        """
        The E-PG responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_epg.T.flatten()

    @property
    def r_peg(self):
        """
        The P-EG responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_peg.T.flatten()

    @property
    def r_pen(self):
        """
        The P-EN responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_pen.T.flatten()

    @property
    def r_pfl3(self):
        """
        The PFL3 responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_pfl3.T.flatten()

    @property
    def r_fbn(self):
        """
        The FsBN responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_fbn.T.flatten()

    @property
    def r_nod(self):
        """
        The Noduli responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_nod.T.flatten()

    @property
    def r_dna2(self):
        """
        The DNa2 responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_dna2.T.flatten()

    @property
    def r_pn(self):
        """
        The PN responses of the mushroom body of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._mb.r_pn.T.flatten()

    @property
    def r_kc(self):
        """
        The KC responses of the mushroom body of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._mb.r_kc[0].T.flatten()

    @property
    def r_mbon(self):
        """
        The MBON responses of the mushroom body of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._mb.r_mbon.T.flatten()
