__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

from env.seville2009 import __root__
from agent import PathIntegrationAgent, VisualNavigationAgent
from invertbrain import CelestialCompass
from sim.simulation import RouteSimulation, VisualNavigationSimulation, PathIntegrationSimulation, Simulation

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


def create_dra_axis(sensor: CelestialCompass, cmap="coolwarm", centre=None, scale=1., draw_axis=True,
                    subplot=111, ax=None):
    omm_x, omm_y, omm_z = sensor.omm_xyz.T

    if ax is None:
        ax = plt.subplot(subplot)

    if centre is None:
        centre = [.5, .5]

    if draw_axis:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.set_aspect('equal', 'box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    size = 20. * scale
    ax.text(centre[0] - .7, centre[1] + .3, "POL", fontsize=10)
    omm = ax.scatter((omm_y * scale + centre[0]).tolist(), (omm_x * scale + centre[1]).tolist(), s=size,
                     c=np.zeros(omm_y.shape[0], dtype='float32'), cmap=cmap, vmin=-.5, vmax=.5)

    return omm


def create_tb1_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                       subplot=111, ax=None):
    nb_tb1 = agent.brain[1].nb_tb1
    return create_image_history(nb_tb1, nb_frames, sep=sep, title="TB1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cl1_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                       subplot=111, ax=None):
    nb_cl1 = agent.brain[1].nb_cl1
    return create_image_history(nb_cl1, nb_frames, sep=sep, title="CL1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu1_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                        subplot=111, ax=None):
    nb_cpu1 = agent.brain[1].nb_cpu1
    return create_image_history(nb_cpu1, nb_frames, sep=sep, title="CPU1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                        subplot=111, ax=None):
    nb_cpu4 = agent.brain[1].nb_cpu4
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_mem_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                            subplot=111, ax=None):
    nb_cpu4 = agent.brain[1].nb_cpu4
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4 (mem)", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cx_axis(agent: PathIntegrationAgent, cmap="coolwarm", subplot=111, ax=None):
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
    omm = create_dra_axis(agent.sensors[0], cmap=cmap, centre=[.8, 4.5], draw_axis=False, ax=ax)

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


def create_image_history(nb_values: int, nb_frames: int, sep: float = None, title: str = None, cmap="Greys",
                         subplot=111, vmin=0, vmax=1, ax=None):
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

    im = ax.imshow(np.zeros((nb_values, nb_frames), dtype='float32'), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="none", aspect="auto")

    if sep is not None:
        ax.plot([sep, sep], [0, nb_values-1], 'grey', lw=1)

    return im


def create_single_line_history(nb_frames: int, sep: float = None, title: str = None, ylim: float = 1., subplot=111, ax=None):
    return create_multi_line_history(nb_frames, 1, sep=sep, title=title, ylim=ylim, subplot=subplot, ax=ax)


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
