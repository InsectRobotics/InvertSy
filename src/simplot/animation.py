from env import Sky, Seville2009
from agent import PathIntegrationAgent, VisualNavigationAgent

from invertsensing import CompoundEye

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.path import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import os

__anim_dir__ = os.path.abspath("../../../OneDrive/PhD/IncentiveCircuit")


class Animation(object):

    def __init__(self, nb_iterations, fps=15, width=11, height=5, name="animation"):
        self._fig = plt.figure(name, figsize=(width, height))
        self._nb_iterations = nb_iterations
        self._fps = fps
        self._ani = None
        self._name = name
        self._lines = []
        self._iteration = 0

    def __call__(self, save=False, save_type="gif", save_name=None, show=True):
        self._animate(0)
        self._ani = animation.FuncAnimation(self._fig, self.__animate, init_func=self.__initialise,
                                            frames=self._nb_iterations, interval=int(1000 / self._fps), blit=True)

        if save:
            if save_name is None:
                save_name = "%s.%s" % (self._name, save_type.lower())
            self.ani.save(os.path.join(__anim_dir__, save_name), fps=self._fps)

        if show:
            plt.show()

    def _initialise(self):
        self._animate(0)

    def _animate(self, i: int):
        raise NotImplementedError()

    def _print_message(self):
        return "Animation %d/%d" % (self._iteration + 1, self._nb_iterations)

    def __initialise(self):
        self._initialise()
        return tuple(self._lines)

    def __animate(self, i: int):
        self._iteration = i
        t0 = time()
        self._animate(i)
        t1 = time()
        print(self._print_message() + " - time: %.2f sec" % (t1 - t0))
        return tuple(self._lines)

    @property
    def fig(self):
        return self._fig

    @property
    def nb_frames(self):
        return self._nb_iterations

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

    def __init__(self, route, agent=None, sky=None, world=None, cmap="Greens_r", show_history=False, show_weights=False,
                 calibrate=False, frequency=False, nb_scans=7, *args, **kwargs):
        self._route = route
        kwargs.setdefault('nb_iterations', int(2 * route.shape[0]))
        name = kwargs.get('name', None)
        super().__init__(*args, **kwargs)

        if agent is None:
            agent = VisualNavigationAgent(saturation=4., freq_trans=frequency, nb_scans=nb_scans)
        self._agent = agent

        if sky is None:
            sky = Sky(30, 180, degrees=True)
        self._sky = sky
        self._world = world

        if name is None:
            name = world.name
        self._name = name

        self._eye = agent.sensors[0]
        self._mem = agent.brain[0]
        self._stats = {
            "path": [],
            "L": [],  # straight distance from the nest
            "C": [],  # distance towards the nest that the agent has covered
        }

        if show_history:
            ax_dict = self.fig.subplot_mosaic(
                """
                AB
                AB
                CB
                DB
                EB
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

        omm = create_eye_axis(self._eye, cmap=cmap, ax=ax_dict["A"])
        line_c, line_b, pos, self._marker, cal = create_map_axis(world=world, ax=ax_dict["B"])

        if show_history:
            pn = create_pn_history(self._agent, self.nb_frames, cmap="Greys", ax=ax_dict["C"])
            kc = create_kc_history(self._agent, self.nb_frames, cmap="Greys", ax=ax_dict["D"])
            fam = create_familiarity_history(self._agent, self.nb_frames, ax=ax_dict["E"])
        else:
            pn, kc, fam = create_mem_axis(self._agent, cmap="Greys", ax=ax_dict["C"])

        plt.tight_layout()

        self._lines.extend([omm, pn, kc, fam, line_c, line_b, pos, cal])

        self._show_history = show_history
        self._show_weights = show_weights
        self._calibrate = calibrate

    def _animate(self, i: int):
        if i == 0:
            self._stats["path"] = []
            self._stats["L"] = []
            self._stats["C"] = []
            self._stats["capacity"] = []
            self._stats["familiarity"] = []
            if self._calibrate and not self._agent.is_calibrated:
                self._agent.xyz = self._route[0, :3]
                self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
                self._agent.update = False
                xyzs, _ = self._agent.calibrate(self._sky, self._world, nb_samples=32, radius=2.)
                self.cal.set_offsets(np.array(xyzs)[:, [1, 0]])
            self._agent.update = True
        elif i == self._route.shape[0]:
            self._agent.xyz = self._route[0, :3]
            self._agent.ori = R.from_euler('Z', self._route[0, 3], degrees=True)
            self._agent.update = False

            self.line_b.set_data(self.line_c.get_data())
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

        # outbound path
        if i < self._route.shape[0]:
            x, y, z, yaw = self._route[i]
            self._agent(sky=self._sky, scene=self._world, act=False, callback=self.callback_outbound)
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
        else:
            # inbound path
            self._agent(sky=self._sky, scene=self._world, act=True, callback=self.callback_inbound)

        r = self._eye(sky=self._sky, scene=self._world).mean(axis=1)

        self.omm.set_array(r.T.flatten())
        # self.pn.set_array(r.T.flatten())
        self.line_c.set_data(np.array(self._stats["path"])[..., 1], np.array(self._stats["path"])[..., 0])
        self.pos.set_offsets(np.array([self._agent.y, self._agent.x]))
        vert, codes = self._marker
        vertices = R.from_euler('Z', -self._agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        if self._show_history:
            pn = self.pn.get_array()
            pn[:, i] = self._mem.r_cs[0].T.flatten()
            self.pn.set_array(pn)
            kc = self.kc.get_array()
            if self._show_weights:
                kc[:, i] = self._mem.w_k2m.T.flatten()
            else:
                kc[:, i] = self._mem.r_kc[0].T.flatten()
            self.kc.set_array(kc)
            fam = self.fam.get_data()
            fam[1][i] = self._familiarity
            self.fam.set_data(*fam)
        else:
            self.pn.set_array(self._mem.r_cs[0].T.flatten())
            if self._show_weights:
                self.kc.set_array(self._mem.w_k2m.T.flatten())
            else:
                self.kc.set_array(self._mem.r_kc[0].T.flatten())
            self.fam.set_array(self._agent.familiarity.T.flatten())

    def _print_message(self):
        x, y, z = self._agent.xyz
        phi = self._agent.yaw_deg
        fam = self._familiarity
        pn_diff = np.absolute(self._mem.r_cs[0] - self.pn.get_array()[:, self._iteration-1]).mean()
        kc_diff = np.absolute(self._mem.r_kc[0] - self.kc.get_array()[:, self._iteration-1]).mean()
        capacity = self._mem.w_k2m.sum() / float(self._mem.w_k2m.size)
        return (super()._print_message() +
                " - x: %.2f, y: %.2f, z: %.2f, Φ: %.0f"
                " - PN (change): %.2f%%, KC (change): %.2f%%, familiarity: %.2f%%, capacity: %.2f%%") % (
            x, y, z, phi, pn_diff * 100., kc_diff * 100., fam * 100., capacity * 100.)

    def callback_all(self, a: PathIntegrationAgent):
        self._stats["path"].append([a.x, a.y, a.z, a.yaw])
        self._stats["L"].append(np.linalg.norm(a.xyz - self._route[0, :3]))
        self._stats["capacity"].append(self._mem.w_k2m.sum() / float(self._mem.w_k2m.size))
        self._stats["familiarity"].append(self._familiarity)

    def callback_outbound(self, a: PathIntegrationAgent):
        self.callback_all(a)
        self._stats["C"].append(0.)

    def callback_inbound(self, a: PathIntegrationAgent):
        self.callback_all(a)
        c = self._stats["C"][-1] if len(self._stats["C"]) > 0 else 0.
        self._stats["C"].append(c + a.step_size)

    @property
    def _familiarity(self):
        fam_array = self._agent.familiarity
        return fam_array[len(fam_array) // 2] if self._iteration < self._route.shape[0] else fam_array.min()

    @property
    def omm(self):
        return self._lines[0]

    @property
    def pn(self):
        return self._lines[1]

    @property
    def kc(self):
        return self._lines[2]

    @property
    def fam(self):
        return self._lines[3]

    @property
    def line_c(self):
        return self._lines[4]

    @property
    def line_b(self):
        return self._lines[5]

    @property
    def pos(self):
        return self._lines[6]

    @property
    def cal(self):
        return self._lines[7]


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
        line_c, line_b, pos, self._marker, _ = create_map_axis(world=world, subplot=122)

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


def create_map_axis(world=None, subplot=111, ax=None):

    if ax is None:
        ax = plt.subplot(subplot)

    line_b, = ax.plot([], [], 'grey', lw=2)

    if world is not None:
        for polygon, colour in zip(world.polygons, world.colours):
            x = polygon[[0, 1, 2, 0], 0]
            y = polygon[[0, 1, 2, 0], 1]
            ax.plot(y, x, c=colour)

    ax.set_ylim([0, 10])
    ax.set_xlim([0, 10])
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', labelsize=8)

    cal = ax.scatter([], [], marker='.', s=50, c='orange')

    line_c, = ax.plot([], [], 'r', lw=2)
    pos = ax.scatter([], [], marker=(3, 2, 0), s=100, c='red')

    points = [0, 2, 3, 4, 6]
    vert = np.array(pos.get_paths()[0].vertices)[points]
    vert[0] *= 2
    codes = pos.get_paths()[0].codes[points]
    vert = np.hstack([vert, np.zeros((vert.shape[0], 1))])

    return line_c, line_b, pos, (vert, codes), cal


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

    omm = ax.scatter(yaw.tolist(), (np.sin(np.deg2rad(-pitch))).tolist(), s=eye.omm_area * 200,
                     c=np.zeros(yaw.shape[0], dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    return omm


def create_mem_axis(agent: VisualNavigationAgent, cmap="Greys", subplot=111, ax=None):
    if ax is None:
        ax = plt.subplot(subplot)
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


def create_pn_history(agent: VisualNavigationAgent, nb_frames: int, cmap="Greys", subplot=111, ax=None):
    nb_pn = agent.brain[0].nb_cs
    if ax is None:
        ax = plt.subplot(subplot)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, nb_pn])
    ax.set_xlim([0, nb_frames])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("PN")

    pn = ax.imshow(np.zeros((nb_pn, nb_frames), dtype='float32'), cmap=cmap, vmin=0, vmax=1,
                   interpolation="none", aspect="auto")

    return pn


def create_kc_history(agent: VisualNavigationAgent, nb_frames: int, cmap="Greys", subplot=111, ax=None):
    nb_kc = agent.brain[0].nb_kc
    if ax is None:
        ax = plt.subplot(subplot)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, nb_kc])
    ax.set_xlim([0, nb_frames])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("KC")

    kc = ax.imshow(np.zeros((nb_kc, nb_frames), dtype='float32'), cmap=cmap, vmin=0, vmax=1,
                   interpolation="none", aspect="auto")
    return kc


def create_familiarity_history(agent: VisualNavigationAgent, nb_frames: int, subplot=111, ax=None):
    nb_en = agent.brain[0].nb_mbon

    if ax is None:
        if isinstance(subplot, int):
            ax = plt.subplot(subplot)
        else:
            ax = plt.subplot(*subplot)

    ax.set_ylim([0, .2])
    ax.set_xlim([0, nb_frames])
    ax.tick_params(axis='both', labelsize=8)
    ax.set_aspect('auto', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fam, = ax.plot(np.full((nb_frames, nb_en), np.nan), 'k-', lw=2)
    ax.text(120, .21, "familiarity", fontsize=10)

    return fam


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
