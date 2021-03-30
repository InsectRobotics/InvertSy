__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

from env.seville2009 import __root__
from env import Sky, Seville2009
from agent import PathIntegrationAgent, VisualNavigationAgent
from sim.simulation import VisualNavigationSimulation, Simulation

from invertsensing import CompoundEye

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.path import Path

import matplotlib.pyplot as plt
import numpy as np
import os

__anim_dir__ = os.path.abspath(os.path.join(__root__, "..", "..", "OneDrive", "PhD", "IncentiveCircuit"))
__stat_dir__ = os.path.abspath(os.path.join(__root__, "data", "animation", "stats"))


class Animation(object):

    def __init__(self, sim: Simulation, fps=15, width=11, height=5, name="animation"):
        self._fig = plt.figure(name, figsize=(width, height))
        self._sim = sim
        self._fps = fps
        self._ani = None
        self._name = name
        self._lines = []
        self._iteration = 0

    def __call__(self, save=False, save_type="gif", save_name=None, save_stats=True, show=True):
        self._animate(0)
        self._ani = animation.FuncAnimation(self._fig, self.__animate, init_func=self.__initialise,
                                            frames=self.nb_frames, interval=int(1000 / self._fps), blit=True)
        try:
            if save:
                if save_name is None:
                    save_name = "%s.%s" % (self._name, save_type.lower())
                self.ani.save(os.path.join(__anim_dir__, save_name), fps=self._fps)

            if show:
                plt.show()
        except KeyboardInterrupt:
            print("Animation interrupted by keyboard!")
        finally:
            if save_stats:
                self._sim.save()

    def _initialise(self):
        self._animate(0)

    def _animate(self, i: int):
        raise NotImplementedError()

    def __initialise(self):
        self._initialise()
        return tuple(self._lines)

    def __animate(self, i: int):
        time = self._animate(i)
        if isinstance(time, float):
            print(self.sim.message() + " - time: %.2f sec" % time)
        return tuple(self._lines)

    @property
    def fig(self):
        return self._fig

    @property
    def sim(self):
        return self._sim

    @property
    def nb_frames(self):
        return self._sim.nb_frames

    @property
    def fps(self):
        return self._fps

    @property
    def ani(self):
        return self._ani

    @property
    def name(self):
        return self._name


class RouteAnimation(Animation):

    def __init__(self, route, eye=None, sky=None, world=None, cmap="Greens_r", *args, **kwargs):
        self._route = route
        kwargs.setdefault('nb_iterations', route.shape[0])
        name = kwargs.get('name', None)
        super().__init__(*args, **kwargs)

        if eye is None:
            eye = CompoundEye(nb_input=5000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10), omm_res=5.,
                              c_sensitive=[0, 0., 1., 0., 0.])
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

        omm = create_eye_axis(eye, cmap=cmap, subplot=221)
        line, pos, self._marker = create_map_axis(world=world, subplot=122)[:3]

        plt.tight_layout()

        self._lines.extend([omm, line, pos])

        r = eye(sky=sky, scene=world).mean(axis=1)
        omm.set_array(r.T.flatten())

    def _animate(self, i: int):
        self._eye._xyz = self._route[i, :3]
        self._eye._ori = R.from_euler('Z', self._route[i, 3], degrees=True)

        r = self._eye(sky=self._sky, scene=self._world).mean(axis=1)

        self.omm.set_array(r.T.flatten())
        self.line.set_data(self._route[:(i+1), 1], self._route[:(i+1), 0])
        self.pos.set_offsets(np.array([self._eye.y, self._eye.x]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self._eye.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

    def _print_message(self):
        return super()._print_message() + " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f" % tuple(self._route[self._iteration])

    @property
    def omm(self):
        return self._lines[0]

    @property
    def line(self):
        return self._lines[1]

    @property
    def pos(self):
        return self._lines[2]


class VisualNavigationAnimation(Animation):

    def __init__(self, sim: VisualNavigationSimulation, cmap="Greens_r", show_history=True, show_weights=False,
                 *args, **kwargs):
        super().__init__(sim, *args, **kwargs)

        if show_history:
            ax_dict = self.fig.subplot_mosaic(
                """
                AABB
                AABB
                CFBB
                DGBB
                EHBB
                """
            )
        else:
            ax_dict = self.fig.subplot_mosaic(
                """
                AB
                CB
                CB
                """
            )

        omm = create_eye_axis(self.sim.eye, cmap=cmap, ax=ax_dict["A"])
        line_c, line_b, pos, self._marker, cal, poi = create_map_axis(
            world=self.sim.world, ax=ax_dict["B"], nest=self.sim.route[-1, :2], feeder=self.sim.route[0, :2])

        self._lines.extend([omm, line_c, line_b, pos, cal, poi])

        if show_history:
            pn = create_pn_history(self.sim.agent, self.nb_frames, sep=self.sim.route.shape[0],
                                   cmap="Greys", ax=ax_dict["C"])
            kc = create_kc_history(self.sim.agent, self.nb_frames, sep=self.sim.route.shape[0],
                                   cmap="Greys", ax=ax_dict["D"])
            fam_all, fam_line = create_familiarity_response_history(
                self.sim.agent, self.nb_frames, sep=self.sim.route.shape[0], cmap="Greys", ax=ax_dict["E"])
            dist = create_single_line_history(
                self.nb_frames, sep=self.sim.route.shape[0], title="d_nest (m)", ylim=8, ax=ax_dict["F"])
            cap = create_capacity_history(self.nb_frames, sep=self.sim.route.shape[0], ax=ax_dict["G"])
            fam = create_familiarity_history(self.nb_frames, sep=self.sim.route.shape[0], ax=ax_dict["H"])

            self._lines.extend([pn, kc, fam, dist, cap, fam_all, fam_line])
        else:
            pn, kc, fam = create_mem_axis(self.sim.agent, cmap="Greys", ax=ax_dict["C"])

            self._lines.extend([pn, kc, fam])

        plt.tight_layout()

        self._show_history = show_history
        self._show_weights = show_weights

    def _animate(self, i: int):
        if i == 0:
            xyzs = self.sim.reset()
            if xyzs is not None:
                self.cal.set_offsets(np.array(xyzs)[:, [1, 0]])
        elif i == self.sim.route.shape[0]:
            self.line_b.set_data(np.array(self.sim.stats["path"])[..., 1], np.array(self.sim.stats["path"])[..., 0])
            self.sim.init_inbound()

        time = self.sim.step(i)

        r = self.sim.stats["ommatidia"][-1].mean(axis=1)
        x, y = np.array(self.sim.stats["path"])[..., :2].T

        self.omm.set_array(r.T.flatten())
        self.line_c.set_data(y, x)
        self.pos.set_offsets(np.array([y[-1], x[-1]]))
        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))
        if "replace" in self.sim.stats and self.sim.stats["replace"][-1]:
            pois = self.poi.get_offsets()
            self.poi.set_offsets(np.vstack([pois, np.array([[y[-1], x[-1]]])]))
        if self._show_history:
            pn = self.pn.get_array()
            pn[:, i] = self.sim.mem.r_cs[0].T.flatten()
            self.pn.set_array(pn)
            kc = self.kc.get_array()
            if self._show_weights:
                kc[:, i] = self.sim.mem.w_k2m.T.flatten()
            else:
                kc[:, i] = self.sim.mem.r_kc[0].T.flatten()
            self.kc.set_array(kc)
            fam = self.fam.get_data()
            fam[1][i] = self.sim.familiarity * 100
            self.fam.set_data(*fam)

            if self.dist is not None:
                dist = self.dist.get_data()
                dist[1][i] = self.sim.d_nest
                self.dist.set_data(*dist)
            if self.capacity is not None:
                cap = self.capacity.get_data()
                cap[1][i] = self.sim.capacity * 100.
                self.capacity.set_data(*cap)
            if self.fam_all is not None:
                fam_all = self.fam_all.get_array()
                fam_all[:, i] = self.sim.agent.familiarity
                self.fam_all.set_array(fam_all)
                if self.fam_line is not None:
                    sep = self.sim.route.shape[0]
                    if self.sim.frame > sep:
                        self.fam_line.set_data(np.arange(sep, self.sim.frame),
                                               np.nanargmin(fam_all[:, sep:self.sim.frame], axis=0))

        else:
            self.pn.set_array(self.sim.mem.r_cs[0].T.flatten())
            if self._show_weights:
                self.kc.set_array(self.sim.mem.w_k2m.T.flatten())
            else:
                self.kc.set_array(self.sim.mem.r_kc[0].T.flatten())
            self.fam.set_array(self.sim.agent.familiarity.T.flatten())

        return time

    @property
    def sim(self) -> VisualNavigationSimulation:
        return self._sim

    @property
    def omm(self):
        return self._lines[0]

    @property
    def line_c(self):
        return self._lines[1]

    @property
    def line_b(self):
        return self._lines[2]

    @property
    def pos(self):
        return self._lines[3]

    @property
    def cal(self):
        return self._lines[4]

    @property
    def poi(self):
        return self._lines[5]

    @property
    def pn(self):
        return self._lines[6]

    @property
    def kc(self):
        return self._lines[7]

    @property
    def fam(self):
        return self._lines[8]

    @property
    def dist(self):
        if len(self._lines) > 9:
            return self._lines[9]
        else:
            return None

    @property
    def capacity(self):
        if len(self._lines) > 10:
            return self._lines[10]
        else:
            return None

    @property
    def fam_all(self):
        if len(self._lines) > 11:
            return self._lines[11]
        else:
            return None

    @property
    def fam_line(self):
        if len(self._lines) > 12:
            return self._lines[12]
        else:
            return None

#
# class VisualNavigationAnimation_2(Animation):
#
#     def __init__(self, route, agent=None, sky=None, world=None, cmap="Greens_r", show_history=True, show_weights=False,
#                  calibrate=False, frequency=False, free_motion=True, nb_ommatidia=None, nb_scans=7, *args, **kwargs):
#         self._route = route
#         kwargs.setdefault('nb_iterations', int(2.5 * route.shape[0]))
#         name = kwargs.get('name', None)
#         super().__init__(*args, **kwargs)
#
#         if agent is None:
#             saturation = 5.
#             eye = None
#             if nb_ommatidia is not None:
#                 eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
#                                   omm_res=saturation, c_sensitive=[0, 0., 1., 0., 0.])
#             agent = VisualNavigationAgent(eye=eye, saturation=saturation, freq_trans=frequency, nb_scans=nb_scans,
#                                           speed=0.01)
#         self._agent = agent
#
#         if sky is None:
#             sky = Sky(30, 180, degrees=True)
#         self._sky = sky
#         self._world = world
#
#         if name is None:
#             name = world.name
#         self._name = name
#
#         self._eye = agent.sensors[0]
#         self._mem = agent.brain[0]
#         self._stats = {
#             "path": [],
#             "L": [],  # straight distance from the nest
#             "C": [],  # distance towards the nest that the agent has covered
#         }
#
#         if show_history:
#             ax_dict = self.fig.subplot_mosaic(
#                 """
#                 AABB
#                 AABB
#                 CFBB
#                 DGBB
#                 EHBB
#                 """
#             )
#         else:
#             ax_dict = self.fig.subplot_mosaic(
#                 """
#                 AB
#                 CB
#                 CB
#                 """
#             )
#
#         omm = create_eye_axis(self._eye, cmap=cmap, ax=ax_dict["A"])
#         line_c, line_b, pos, self._marker, cal, poi = create_map_axis(world=world, ax=ax_dict["B"],
#                                                                       nest=route[-1, :2], feeder=route[0, :2])
#
#         self._lines.extend([omm, line_c, line_b, pos, cal, poi])
#
#         if show_history:
#             pn = create_pn_history(self._agent, self.nb_frames, sep=route.shape[0], cmap="Greys", ax=ax_dict["C"])
#             kc = create_kc_history(self._agent, self.nb_frames, sep=route.shape[0], cmap="Greys", ax=ax_dict["D"])
#             fam_all, fam_line = create_familiarity_response_history(self._agent, self.nb_frames, sep=route.shape[0],
#                                                                     cmap="Greys", ax=ax_dict["E"])
#
#             dist = create_single_line_history(self.nb_frames, sep=route.shape[0], title="d_nest (m)", ylim=8,
#                                               ax=ax_dict["F"])
#             cap = create_capacity_history(self.nb_frames, sep=route.shape[0], ax=ax_dict["G"])
#             fam = create_familiarity_history(self.nb_frames, sep=route.shape[0], ax=ax_dict["H"])
#
#             self._lines.extend([pn, kc, fam, dist, cap, fam_all, fam_line])
#         else:
#             pn, kc, fam = create_mem_axis(self._agent, cmap="Greys", ax=ax_dict["C"])
#
#             self._lines.extend([pn, kc, fam])
#
#         plt.tight_layout()
#
#         self._show_history = show_history
#         self._show_weights = show_weights
#         self._calibrate = calibrate
#         self._free_motion = free_motion
#
#     def _animate(self, i: int):
#         if i == 0:
#             self._stats["ommatidia"] = []
#             self._stats["PN"] = []
#             self._stats["KC"] = []
#             self._stats["MBON"] = []
#             self._stats["DAN"] = []
#             self._stats["path"] = []
#             self._stats["L"] = []  # straight distance from the nest
#             self._stats["C"] = []  # distance that the agent has covered
#             self._stats["capacity"] = []
#             self._stats["familiarity"] = []
#             if self._calibrate and not self._agent.is_calibrated:
#                 self._agent.xyz = self._route[-1, :3]
#                 self._agent.ori = R.from_euler('Z', self._route[-1, 3], degrees=True)
#                 self._agent.update = False
#                 xyzs, _ = self._agent.calibrate(self._sky, self._world, nb_samples=32, radius=2.)
#                 self.cal.set_offsets(np.array(xyzs)[:, [1, 0]])
#
#             self._agent.xyz = self._route[0, :3]
#             self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
#             self._agent.update = True
#         elif i == self._route.shape[0]:
#             self._agent.xyz = self._route[0, :3]
#             self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
#             self._agent.update = False
#
#             self.line_b.set_data(self.line_c.get_data())
#             # create a separate line
#             self._stats["outbound"] = self._stats["path"]
#             self._stats["L_out"] = self._stats["L"]
#             self._stats["C_out"] = self._stats["C"]
#             self._stats["capacity_out"] = self._stats["capacity"]
#             self._stats["familiarity_out"] = self._stats["familiarity"]
#             self._stats["path"] = []
#             self._stats["L"] = []
#             self._stats["C"] = []
#             self._stats["capacity"] = []
#             self._stats["familiarity"] = []
#             self._stats["replace"] = []
#
#         # outbound path
#         if i < self._route.shape[0]:
#             x, y, z, yaw = self._route[i]
#             self._agent(sky=self._sky, scene=self._world, act=False, callback=self.callback)
#             self._agent.xyz = [x, y, z]
#             self._agent.ori = R.from_euler('Z', yaw, degrees=True)
#         else:
#             # inbound path
#             act = not (len(self._stats["L"]) > 0 and self._stats["L"][-1] <= 0.01)
#             self._agent(sky=self._sky, scene=self._world, act=act, callback=self.callback)
#             if not act:
#                 self._agent.rotate(R.from_euler('Z', 1, degrees=True))
#             if not self._free_motion and "replace" in self._stats:
#                 d_route = np.linalg.norm(self._route[:, :3] - self._agent.xyz, axis=1)
#                 point = np.argmin(d_route)
#                 if d_route[point] > 0.1:  # move for more than 10cm away from the route
#                     self._agent.xyz = self._route[point, :3]
#                     self._agent.ori = R.from_euler('Z', self._route[point, 3], degrees=True)
#                     self._stats["replace"].append(True)
#                     print(" ~ REPLACE ~")
#                 else:
#                     self._stats["replace"].append(False)
#
#         r = self._eye.responses.mean(axis=1)
#
#         self.omm.set_array(r.T.flatten())
#         self.line_c.set_data(np.array(self._stats["path"])[..., 1], np.array(self._stats["path"])[..., 0])
#         self.pos.set_offsets(np.array([self._agent.y, self._agent.x]))
#         vert, codes = self._marker
#         vertices = R.from_euler('Z', -self._agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
#         self.pos.set_paths((Path(vertices[:, :2], codes),))
#         if "replace" in self._stats and self._stats["replace"][-1]:
#             pois = self.poi.get_offsets()
#             x, y, _, _ = self._stats["path"][-1]
#             self.poi.set_offsets(np.vstack([pois, np.array([[y, x]])]))
#         if self._show_history:
#             pn = self.pn.get_array()
#             pn[:, i] = self._mem.r_cs[0].T.flatten()
#             self.pn.set_array(pn)
#             kc = self.kc.get_array()
#             if self._show_weights:
#                 kc[:, i] = self._mem.w_k2m.T.flatten()
#             else:
#                 kc[:, i] = self._mem.r_kc[0].T.flatten()
#             self.kc.set_array(kc)
#             fam = self.fam.get_data()
#             fam[1][i] = self._familiarity * 100
#             self.fam.set_data(*fam)
#
#             if self.dist is not None:
#                 dist = self.dist.get_data()
#                 dist[1][i] = self._d_nest
#                 self.dist.set_data(*dist)
#             if self.capacity is not None:
#                 cap = self.capacity.get_data()
#                 cap[1][i] = self._capacity * 100.
#                 self.capacity.set_data(*cap)
#             if self.fam_all is not None:
#                 fam_all = self.fam_all.get_array()
#                 fam_all[:, i] = self._agent.familiarity
#                 self.fam_all.set_array(fam_all)
#                 if self.fam_line is not None:
#                     sep = self._route.shape[0]
#                     if self._iteration > sep:
#                         self.fam_line.set_data(np.arange(sep, self._iteration),
#                                                np.nanargmin(fam_all[:, sep:self._iteration], axis=0))
#
#         else:
#             self.pn.set_array(self._mem.r_cs[0].T.flatten())
#             if self._show_weights:
#                 self.kc.set_array(self._mem.w_k2m.T.flatten())
#             else:
#                 self.kc.set_array(self._mem.r_kc[0].T.flatten())
#             self.fam.set_array(self._agent.familiarity.T.flatten())
#
#     def _print_message(self):
#         x, y, z = self._agent.xyz
#         phi = self._agent.yaw_deg
#         fam = self._familiarity
#         pn_diff = np.absolute(self._mem.r_cs[0] - self.pn.get_array()[:, self._iteration-1]).mean()
#         kc_diff = np.absolute(self._mem.r_kc[0] - self.kc.get_array()[:, self._iteration-1]).mean()
#         capacity = self._capacity
#         d_nest = self._d_nest
#         d_trav = (self._stats["C"][-1] if len(self._stats["C"]) > 0
#                   else (self._stats["C_out"][-1] if "C_out" in self._stats else 0.))
#         return (super()._print_message() +
#                 " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f"
#                 " - PN (change): %.2f%%, KC (change): %.2f%%, familiarity: %.2f%%,"
#                 " capacity: %.2f%%, L: %.2fm, C: %.2fm") % (
#             x, y, z, phi, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100., d_nest, d_trav)
#
#     def callback(self, a: PathIntegrationAgent):
#         self._stats["ommatidia"].append(self._eye.responses)
#         self._stats["PN"].append(self._mem.r_cs)
#         self._stats["KC"].append(self._mem.r_kc)
#         self._stats["MBON"].append(self._mem.r_mbon)
#         self._stats["DAN"].append(self._mem.r_dan)
#         self._stats["path"].append([a.x, a.y, a.z, a.yaw])
#         self._stats["L"].append(np.linalg.norm(a.xyz - self._route[-1, :3]))
#         self._stats["capacity"].append(self._mem.w_k2m.sum() / float(self._mem.w_k2m.size))
#         self._stats["familiarity"].append(self._familiarity)
#         c = self._stats["C"][-1] if len(self._stats["C"]) > 0 else 0.
#         if len(self._stats["path"]) > 1:
#             step = np.linalg.norm(np.array(self._stats["path"][-1])[:3] - np.array(self._stats["path"][-2])[:3])
#         else:
#             step = 0.
#         self._stats["C"].append(c + step)
#
#     @property
#     def _familiarity(self):
#         fam_array = self._agent.familiarity
#         return fam_array[len(fam_array) // 2] if self._iteration < self._route.shape[0] else fam_array.min()
#
#     @property
#     def _capacity(self):
#         return self._mem.w_k2m.sum() / float(self._mem.w_k2m.size)
#
#     @property
#     def _d_nest(self):
#         return (self._stats["L"][-1] if len(self._stats["L"]) > 0
#                 else np.linalg.norm(self._route[-1, :3] - self._route[0, :3]))
#
#     @property
#     def omm(self):
#         return self._lines[0]
#
#     @property
#     def line_c(self):
#         return self._lines[1]
#
#     @property
#     def line_b(self):
#         return self._lines[2]
#
#     @property
#     def pos(self):
#         return self._lines[3]
#
#     @property
#     def cal(self):
#         return self._lines[4]
#
#     @property
#     def poi(self):
#         return self._lines[5]
#
#     @property
#     def pn(self):
#         return self._lines[6]
#
#     @property
#     def kc(self):
#         return self._lines[7]
#
#     @property
#     def fam(self):
#         return self._lines[8]
#
#     @property
#     def dist(self):
#         if len(self._lines) > 9:
#             return self._lines[9]
#         else:
#             return None
#
#     @property
#     def capacity(self):
#         if len(self._lines) > 10:
#             return self._lines[10]
#         else:
#             return None
#
#     @property
#     def fam_all(self):
#         if len(self._lines) > 11:
#             return self._lines[11]
#         else:
#             return None
#
#     @property
#     def fam_line(self):
#         if len(self._lines) > 12:
#             return self._lines[12]
#         else:
#             return None


class PathIntegrationAnimation(Animation):

    def __init__(self, route, agent=None, sky=None, world=None, cmap="coolwarm", *args, **kwargs):
        self._route = route
        kwargs.setdefault('fps', 100)
        kwargs.setdefault('nb_iterations', int(2.5 * route.shape[0]))
        name = kwargs.get('name', None)
        super().__init__(*args, **kwargs)

        if agent is None:
            agent = PathIntegrationAgent()
        self._agent = agent

        if sky is None:
            sky = Sky(30, 180, degrees=True)
        self._sky = sky
        self._world = world

        if name is None:
            name = world.name
        self._name = name

        self._compass_sensor = agent.sensors[0]
        self._compass_model, self._cx = agent.brain
        self._stats = {
            "path": [],
            "L": [],  # straight distance from the nest
            "C": [],  # distance towards the nest that the agent has covered
        }

        omm, tb1, cl1, cpu1, cpu4, cpu4mem = create_cx_axis(agent, cmap=cmap, subplot=121)
        line_c, line_b, pos, self._marker = create_map_axis(world=world, subplot=122)[:4]

        plt.tight_layout()

        self._lines.extend([omm, tb1, cl1, cpu1, cpu4, cpu4mem, line_c, line_b, pos])

        r = self._compass_model.r_pol
        omm.set_array(r.T.flatten())

    def _animate(self, i: int):
        if i == 0:
            self._stats["path"] = []
            self._stats["L"] = []
            self._stats["C"] = []

        # outbound path
        if i < self._route.shape[0]:
            x, y, z, yaw = self._route[i]

            self._agent(sky=self._sky, act=False, callback=self.callback_outbound)
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)

        else:
            self._agent(sky=self._sky, act=True, callback=self.callback_inbound)

        self.omm.set_array(self._compass_model.r_pol.T.flatten())
        self.tb1.set_array(self._cx.r_tb1.T.flatten())
        self.cl1.set_array(self._cx.r_cl1.T.flatten())
        self.cpu1.set_array(self._cx.r_cpu1.T.flatten())
        self.cpu4.set_array(self._cx.r_cpu4.T.flatten())
        self.cpu4mem.set_array(self._cx.cpu4_mem.T.flatten())
        self.line_c.set_data(np.array(self._stats["path"])[..., 1], np.array(self._stats["path"])[..., 0])
        self.pos.set_offsets(np.array([self._agent.y, self._agent.x]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self._agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

    def _print_message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        return super()._print_message() + " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f" % (x, y, z, phi)

    def callback_all(self, a: PathIntegrationAgent):
        self._stats["path"].append([a.x, a.y, a.z, a.yaw])
        self._stats["L"].append(np.linalg.norm(a.xyz - self._route[0, :3]))

    def callback_outbound(self, a: PathIntegrationAgent):
        self.callback_all(a)
        self._stats["C"].append(0.)

    def callback_inbound(self, a: PathIntegrationAgent):
        self.callback_all(a)
        self._stats["C"].append(self._stats["C"][-1] + a.step_size)

    @property
    def omm(self):
        return self._lines[0]

    @property
    def tb1(self):
        return self._lines[1]

    @property
    def cl1(self):
        return self._lines[2]

    @property
    def cpu1(self):
        return self._lines[3]

    @property
    def cpu4(self):
        return self._lines[4]

    @property
    def cpu4mem(self):
        return self._lines[5]

    @property
    def line_c(self):
        return self._lines[6]

    @property
    def line_b(self):
        return self._lines[7]

    @property
    def pos(self):
        return self._lines[8]


def create_map_axis(world=None, nest=None, feeder=None, subplot=111, ax=None):

    if ax is None:
        ax = plt.subplot(subplot)

    line_b, = ax.plot([], [], 'grey', lw=2)

    if world is not None:
        for polygon, colour in zip(world.polygons, world.colours):
            x = polygon[[0, 1, 2, 0], 0]
            y = polygon[[0, 1, 2, 0], 1]
            ax.plot(y, x, c=colour)

    if nest is not None:
        ax.scatter([nest[1]], [nest[0]], marker='o', s=50, c='black')
        ax.text(nest[1] - 1, nest[0] - .5, "Nest")

    if feeder is not None:
        ax.scatter([feeder[1]], [feeder[0]], marker='o', s=50, c='black')
        ax.text(feeder[1] + .2, feeder[0] + .2, "Feeder")

    ax.set_ylim([0, 10])
    ax.set_xlim([0, 10])
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', labelsize=8)

    cal = ax.scatter([], [], marker='.', s=50, c='orange')

    poi = ax.scatter([], [], marker='.', s=100, c='blue')

    line_c, = ax.plot([], [], 'r', lw=2)
    pos = ax.scatter([], [], marker=(3, 2, 0), s=100, c='red')

    points = [0, 2, 3, 4, 6]
    vert = np.array(pos.get_paths()[0].vertices)[points]
    vert[0] *= 2
    codes = pos.get_paths()[0].codes[points]
    vert = np.hstack([vert, np.zeros((vert.shape[0], 1))])

    return line_c, line_b, pos, (vert, codes), cal, poi


def create_eye_axis(eye: CompoundEye, cmap="Greys_r", subplot=111, ax=None):
    if ax is None:
        ax = plt.subplot(subplot)
    ax.set_yticks(np.sin([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]))
    ax.set_yticklabels([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim([-1, 1])
    ax.set_xlim([-180, 180])
    ax.tick_params(axis='both', labelsize=8)

    yaw, pitch, roll = eye.omm_ori.as_euler('ZYX', degrees=True).T
    eye_size = 5000. / eye.nb_ommatidia * eye.omm_area * 80
    omm = ax.scatter(yaw.tolist(), (np.sin(np.deg2rad(-pitch))).tolist(), s=eye_size,
                     c=np.zeros(yaw.shape[0], dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    return omm


def create_mem_axis(agent: VisualNavigationAgent, cmap="Greys", subplot=111, ax=None):
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, 5])
    ax.set_xlim([0, 13])
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    mem = agent.brain[0]

    size = 400.
    ax.text(.1, 4.8, "PN", fontsize=10)
    pn = ax.scatter(np.linspace(.3, 12.7, mem.nb_cs), np.full(mem.nb_cs, 4.5), s=size / mem.nb_cs,
                    c=np.zeros(mem.nb_cs, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 3.8, "KC", fontsize=10)
    nb_rows = 50
    nb_cols = int(mem.nb_kc / nb_rows) + 1
    x = np.array([np.linspace(.3, 12.7, nb_cols)] * nb_rows).flatten()[:mem.nb_kc]
    y = np.array([np.linspace(1.3, 3.5, nb_rows)] * nb_cols).T.flatten()[:mem.nb_kc]
    kc = ax.scatter(x, y, s=size / mem.nb_kc, c=np.zeros(mem.nb_kc, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 0.8, "familiarity", fontsize=10)
    nb_fam = len(agent.pref_angles)
    fam = ax.scatter(np.linspace(.3, 12.7, nb_fam), np.full(nb_fam, 0.5), s=size / nb_fam,
                     c=np.zeros(nb_fam, dtype='float32'), cmap=cmap, vmin=0, vmax=mem.nb_cs * mem.sparseness)

    return pn, kc, fam


def create_pn_history(agent: VisualNavigationAgent, nb_frames: int, sep: float = None, cmap="Greys",
                      subplot=111, ax=None):
    nb_pn = agent.brain[0].nb_cs
    return create_image_history(nb_pn, nb_frames, sep=sep, title="PN",  cmap=cmap, subplot=subplot, ax=ax)


def create_kc_history(agent: VisualNavigationAgent, nb_frames: int, sep: float = None, cmap="Greys",
                      subplot=111, ax=None):
    nb_kc = agent.brain[0].nb_kc
    return create_image_history(nb_kc, nb_frames, sep=sep, title="KC",  cmap=cmap, subplot=subplot, ax=ax)


def create_familiarity_response_history(agent: VisualNavigationAgent, nb_frames: int, sep: float = None, cmap="Greys",
                                        subplot=111, ax=None):
    nb_scans = agent.nb_scans

    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, nb_scans-1])
    ax.set_xlim([0, nb_frames-1])
    ax.set_yticks([0, nb_scans//2, nb_scans-1])
    ax.set_yticklabels([int(agent.pref_angles[0]), int(agent.pref_angles[nb_scans//2]), int(agent.pref_angles[-1])])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("familiarity", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    fam = ax.imshow(np.zeros((nb_scans, nb_frames), dtype='float32'), cmap=cmap, vmin=0, vmax=.2,
                    interpolation="none", aspect="auto")

    fam_line, = ax.plot([], [], 'red', lw=.5, alpha=.5)

    if sep is not None:
        ax.plot([sep, sep], [0, nb_scans-1], 'grey', lw=1)

    return fam, fam_line


def create_familiarity_history(nb_frames: int, sep: float = None, subplot=111, ax=None):
    return create_single_line_history(nb_frames, sep=sep, title="familiarity (%)", ylim=20, subplot=subplot, ax=ax)


def create_capacity_history(nb_frames: int, sep: float = None, subplot=111, ax=None):
    return create_single_line_history(nb_frames, sep=sep, title="capacity (%)", ylim=100, subplot=subplot, ax=ax)


def create_cx_axis(agent: PathIntegrationAgent, cmap="coolwarm", subplot=111, ax=None):
    omm_x, omm_y, omm_z = agent.sensors[0].omm_xyz.T

    if ax is None:
        ax = plt.subplot(subplot)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, 5])
    ax.set_xlim([0, 5])
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    size = 20.
    ax.text(.1, 4.8, "POL", fontsize=10)
    omm = ax.scatter((omm_y + .8).tolist(), (omm_x + 4.5).tolist(), s=size,
                     c=np.zeros(omm_y.shape[0], dtype='float32'), cmap=cmap, vmin=-.5, vmax=.5)

    ax.text(1.5, 4.8, "TB1", fontsize=10)
    tb1 = ax.scatter(np.linspace(2, 4.5, 8), np.full(8, 4.5), s=2 * size,
                     c=np.zeros_like(agent.brain[1].r_tb1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 3.8, "CL1", fontsize=10)
    cl1 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 3.5), s=2 * size,
                     c=np.zeros_like(agent.brain[1].r_cl1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 2.8, "CPU1", fontsize=10)
    cpu1 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 2.5), s=2 * size,
                      c=np.zeros_like(agent.brain[1].r_cpu1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 1.8, "CPU4", fontsize=10)
    cpu4 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 1.5), s=2 * size,
                      c=np.zeros_like(agent.brain[1].r_cpu4), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, .8, "CPU4 (mem)", fontsize=10)
    cpu4mem = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, .5), s=2 * size,
                         c=np.zeros_like(agent.brain[1].r_cpu4), cmap=cmap, vmin=0, vmax=1)

    return omm, tb1, cl1, cpu1, cpu4, cpu4mem


def create_image_history(nb_values: int, nb_frames: int, sep: float = None, title: str = None, cmap="Greys", subplot=111, ax=None):
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, nb_values-1])
    ax.set_xlim([0, nb_frames-1])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if title is not None:
        ax.set_ylabel(title)

    im = ax.imshow(np.zeros((nb_values, nb_frames), dtype='float32'), cmap=cmap, vmin=0, vmax=1,
                   interpolation="none", aspect="auto")

    if sep is not None:
        ax.plot([sep, sep], [0, nb_values-1], 'grey', lw=1)

    return im


def create_single_line_history(nb_frames: int, sep: float = None, title: str = None, ylim: float = 1., subplot=111, ax=None):
    return create_multi_line_history(nb_frames, 1, sep=sep,title=title, ylim=ylim, subplot=subplot, ax=ax)


def create_multi_line_history(nb_frames: int, nb_lines: int, sep: float = None, title: str = None, ylim: float = 1.,
                              subplot=111, ax=None):
    ax = get_axis(ax, subplot)

    ax.set_ylim([0, ylim])
    ax.set_xlim([0, nb_frames])
    ax.tick_params(axis='both', labelsize=8)
    ax.set_aspect('auto', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if sep is not None:
        ax.plot([sep, sep], [0, ylim], 'grey', lw=3)

    lines, = ax.plot(np.full((nb_frames, nb_lines), np.nan), 'k-', lw=2)
    if title is not None:
        ax.text(120, ylim * 1.05, title, fontsize=10)

    return lines


def get_axis(ax=None, subplot=111):
    if ax is None:
        if isinstance(subplot, int):
            ax = plt.subplot(subplot)
        else:
            ax = plt.subplot(*subplot)
    return ax
