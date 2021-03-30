from env import Sky
from env.seville2009 import __root__
from agent import VisualNavigationAgent

from invertsensing import CompoundEye

from scipy.spatial.transform import Rotation as R

import numpy as np

from time import time

import os

__stat_dir__ = os.path.abspath(os.path.join(__root__, "data", "animation", "stats"))


class Simulation(object):

    def __init__(self, nb_iterations, name="simulation"):
        self._nb_iterations = nb_iterations
        self._iteration = 0
        self._stats = {}
        self._name = name

    def reset(self):
        raise NotImplementedError()

    def step(self, i: int):
        self._iteration = i
        t0 = time()
        self._step(i)
        t1 = time()
        return t1 - t0

    def _step(self, i: int):
        raise NotImplementedError()

    def save(self, filename: str = None):
        if filename is None:
            filename = self._name
        else:
            filename = filename.replace('.npz', '')
        np.savez_compressed(os.path.join(__stat_dir__, "%s.npz" % filename), **self._stats)

    def __call__(self, save=False):
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
        return "Simulation %d/%d" % (self._iteration + 1, self._nb_iterations)

    @property
    def stats(self):
        return self._stats

    @property
    def nb_frames(self):
        return self._nb_iterations

    @property
    def frame(self):
        return self._iteration

    @property
    def name(self):
        return self._name


class VisualNavigationSimulation(Simulation):

    def __init__(self, route, agent=None, sky=None, world=None, nb_ommatidia=None, nb_scans=7,
                 calibrate=False, frequency=False, free_motion=True, **kwargs):
        kwargs.setdefault('nb_iterations', int(2.5 * route.shape[0]))
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

        self._calibrate = calibrate
        self._free_motion = free_motion

    def reset(self):
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

    def _step(self, i: int):

        if i < self._route.shape[0]:  # outbound path
            x, y, z, yaw = self._route[i]
            self._agent(sky=self._sky, scene=self._world, act=False, callback=self.update_stats)
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)

        else:  # inbound path
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

    def update_stats(self, a: VisualNavigationAgent):
        eye = a.sensors[0]
        mem = a.brain[0]
        self._stats["ommatidia"].append(eye.responses.copy())
        self._stats["PN"].append(mem.r_cs.copy())
        self._stats["KC"].append(mem.r_kc.copy())
        self._stats["MBON"].append(mem.r_mbon.copy())
        self._stats["DAN"].append(mem.r_dan.copy())
        self._stats["path"].append([a.x, a.y, a.z, a.yaw])
        self._stats["L"].append(np.linalg.norm(a.xyz - self._route[-1, :3]))
        self._stats["capacity"].append(mem.w_k2m.sum() / float(mem.w_k2m.size))
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
            pn_diff = np.absolute(self._mem.r_cs[0]).mean()
            kc_diff = np.absolute(self._mem.r_kc[0]).mean()
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
        return self._agent

    @property
    def world(self):
        return self._world

    @property
    def route(self):
        return self._route

    @property
    def eye(self):
        return self._eye

    @property
    def mem(self):
        return self._mem

    @property
    def familiarity(self):
        fam_array = self._agent.familiarity
        return fam_array[len(fam_array) // 2] if self._iteration < self._route.shape[0] else fam_array.min()

    @property
    def capacity(self):
        return self._mem.w_k2m.sum() / float(self._mem.w_k2m.size)

    @property
    def d_nest(self):
        return (self._stats["L"][-1] if len(self._stats["L"]) > 0
                else np.linalg.norm(self._route[-1, :3] - self._route[0, :3]))

    @property
    def calibrate(self):
        return self._calibrate

    @property
    def free_motion(self):
        return self._free_motion
