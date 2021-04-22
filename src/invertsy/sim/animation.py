"""
Package that contains tools that create the animations.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertsy.__helpers import __root__

from ._helpers import *
from .simulation import RouteSimulation, VisualNavigationSimulation, PathIntegrationSimulation, Simulation

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.path import Path

import matplotlib.pyplot as plt
import numpy as np
import os

__anim_dir__ = os.path.abspath(os.path.join(__root__, "..", "..", "OneDrive", "PhD", "IncentiveCircuit"))
__stat_dir__ = os.path.abspath(os.path.join(__root__, "data", "animation", "stats"))


class Animation(object):

    def __init__(self, sim: Simulation, fps=15, width=11, height=5, name=None):
        self._fig = plt.figure(name, figsize=(width, height))
        self._sim = sim
        self._fps = fps
        self._ani = None
        if name is None:
            name = self._sim.name + "-anim"
        self._name = name
        self._lines = []
        self._iteration = 0

    def __call__(self, save=False, save_type="gif", save_name=None, save_stats=True, show=True):
        self.sim.reset()
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

    def __init__(self, sim: RouteSimulation, cmap="Greens_r", *args, **kwargs):
        super().__init__(sim, *args, **kwargs)

        omm = create_eye_axis(sim.eye, cmap=cmap, subplot=221)
        line, _, pos, self._marker = create_map_axis(world=sim.world, subplot=122)[:4]

        plt.tight_layout()

        self._lines.extend([omm, line, pos])

        omm.set_array(sim.responses)

    def _animate(self, i: int):
        self.sim.step(i)

        self.omm.set_array(self.sim.responses)
        self.line.set_data(self.sim.route[:(i+1), 1], self.sim.route[:(i+1), 0])
        self.pos.set_offsets(np.array([self.sim.eye.y, self.sim.eye.x]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.eye.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

    @property
    def omm(self):
        return self._lines[0]

    @property
    def line(self):
        return self._lines[1]

    @property
    def pos(self):
        return self._lines[2]

    @property
    def sim(self) -> RouteSimulation:
        return self._sim


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

        omm = create_eye_axis(sim.eye, cmap=cmap, ax=ax_dict["A"])
        line_c, line_b, pos, self._marker, cal, poi = create_map_axis(
            world=sim.world, ax=ax_dict["B"], nest=sim.route[-1, :2], feeder=sim.route[0, :2])

        self._lines.extend([omm, line_c, line_b, pos, cal, poi])

        if show_history:
            pn = create_pn_history(sim.agent, self.nb_frames, sep=sim.route.shape[0],
                                   cmap="Greys", ax=ax_dict["C"])
            kc = create_kc_history(sim.agent, self.nb_frames, sep=sim.route.shape[0],
                                   cmap="Greys", ax=ax_dict["D"])
            fam_all, fam_line = create_familiarity_response_history(
                sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap="Greys", ax=ax_dict["E"])
            dist = create_single_line_history(
                self.nb_frames, sep=sim.route.shape[0], title="d_nest (m)", ylim=8, ax=ax_dict["F"])
            cap = create_capacity_history(self.nb_frames, sep=sim.route.shape[0], ax=ax_dict["G"])
            fam = create_familiarity_history(self.nb_frames, sep=sim.route.shape[0], ax=ax_dict["H"])

            self._lines.extend([pn, kc, fam, dist, cap, fam_all, fam_line])
        else:
            pn, kc, fam = create_mem_axis(sim.agent, cmap="Greys", ax=ax_dict["C"])

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


class PathIntegrationAnimation(Animation):

    def __init__(self, sim: PathIntegrationSimulation, show_history=True, cmap="coolwarm", *args, **kwargs):
        kwargs.setdefault('fps', 100)
        super().__init__(sim, *args, **kwargs)

        if show_history:
            ax_dict = self.fig.subplot_mosaic(
                """
                ACCCBBBB
                DDDDBBBB
                EEEEBBBB
                FFFFBBBB
                GGGGBBBB
                """
            )
        else:
            ax_dict = self.fig.subplot_mosaic(
                """
                AB
                AB
                AB
                """
            )

        line_c, line_b, pos, self._marker = create_map_axis(world=sim.world, ax=ax_dict["B"],
                                                            nest=sim.route[0, :2], feeder=sim.route[-1, :2])[:4]

        if show_history:
            omm = create_dra_axis(sim.agent.sensors[0], cmap=cmap, ax=ax_dict["A"])
            tb1 = create_tb1_history(sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap=cmap, ax=ax_dict["C"])
            cl1 = create_cl1_history(sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap=cmap, ax=ax_dict["D"])
            cpu1 = create_cpu1_history(sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap=cmap, ax=ax_dict["E"])
            cpu4 = create_cpu4_history(sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap=cmap, ax=ax_dict["F"])
            cpu4mem = create_cpu4_mem_history(sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap=cmap,
                                              ax=ax_dict["G"])
        else:
            omm, tb1, cl1, cpu1, cpu4, cpu4mem = create_cx_axis(sim.agent, cmap=cmap, ax=ax_dict["A"])

        plt.tight_layout()

        self._lines.extend([omm, tb1, cl1, cpu1, cpu4, cpu4mem, line_c, line_b, pos])

        omm.set_array(sim.r_pol)
        self._show_history = show_history

    def _animate(self, i: int):
        if i == 0:
            self.line_b.set_data([], [])
            self.sim.reset()
        elif i == self.sim.route.shape[0]:
            self.line_b.set_data(np.array(self.sim.stats["path"])[..., 1], np.array(self.sim.stats["path"])[..., 0])
            self.sim.init_inbound()

        time = self.sim.step(i)

        self.omm.set_array(self.sim.r_pol)

        if self._show_history:
            tb1 = np.zeros((self.sim.r_tb1.shape[0], self.nb_frames), dtype=float)
            tb1[:, :i+1] = np.array(self.sim.stats["TB1"]).T
            self.tb1.set_array(tb1)
            cl1 = np.zeros((self.sim.r_cl1.shape[0], self.nb_frames), dtype=float)
            cl1[:, :i+1] = np.array(self.sim.stats["CL1"]).T
            self.cl1.set_array(cl1)
            cpu1 = np.zeros((self.sim.r_cpu1.shape[0], self.nb_frames), dtype=float)
            cpu1[:, :i+1] = np.array(self.sim.stats["CPU1"]).T
            self.cpu1.set_array(cpu1)
            cpu4 = np.zeros((self.sim.r_cpu4.shape[0], self.nb_frames), dtype=float)
            cpu4[:, :i+1] = np.array(self.sim.stats["CPU4"]).T
            self.cpu4.set_array(cpu4)
            cpu4mem = np.zeros((self.sim.cpu4_mem.shape[0], self.nb_frames), dtype=float)
            cpu4mem[:, :i+1] = np.array(self.sim.stats["CPU4mem"]).T
            self.cpu4mem.set_array(cpu4mem)
        else:
            self.tb1.set_array(self.sim.r_tb1)
            self.cl1.set_array(self.sim.r_cl1)
            self.cpu1.set_array(self.sim.r_cpu1)
            self.cpu4.set_array(self.sim.r_cpu4)
            self.cpu4mem.set_array(self.sim.cpu4_mem)

        x, y = np.array(self.sim.stats["path"])[..., :2].T
        self.line_c.set_data(y, x)
        self.pos.set_offsets(np.array([y[-1], x[-1]]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        return time

    @property
    def sim(self) -> PathIntegrationSimulation:
        return self._sim

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
