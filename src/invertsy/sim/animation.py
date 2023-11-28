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

from collections import namedtuple

from invertsy.__helpers import __root__
from invertsy.agent.agent import RouteFollowingAgent, VisualProcessingAgent, CentralComplexAgent

from ._helpers import *
from .simulation import RouteSimulation, NavigationSimulation, SimulationBase
from .simulation import PathIntegrationSimulation, TwoSourcePathIntegrationSimulation
from .simulation import VisualNavigationSimulation, CentralPointNavigationSimulationBase

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.path import Path

import loguru as lg
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import sys
import os

__default_anim_dir__ = os.path.abspath(os.path.join(__root__, "data", "animation", "vids"))
if not os.path.isdir(__default_anim_dir__):
    os.makedirs(__default_anim_dir__)
if "linux" in sys.platform:
    __drive_dir__ = os.path.abspath(os.path.join(__root__, "..", "..", "OneDrive"))
elif "win" in sys.platform:
    __drive_dir__ = os.path.abspath(os.path.join("C:", "Users", "Odin", "OneDrive - University of Edinburgh"))
else:
    __drive_dir__ = __default_anim_dir__

__anim_dir__ = os.path.join(__drive_dir__, "PhD", "antworld-animations")

if not os.path.isdir(__anim_dir__):
    __anim_dir__ = __default_anim_dir__

"""
Directory where to save the animation videos to 
"""
__stat_dir__ = os.path.abspath(os.path.join(__root__, "data", "animation", "stats"))
"""
Directory where to save the simulation statistics logs
"""


class Animation(object):
    Marker = namedtuple('Marker', ['position', 'codes', 'vert'])

    def __init__(self, mosaic=None, fps=15, width=5, height=5, name=None):

        self.__fps = fps
        self.__width = width
        self.__height = height
        if name is None:
            name = "animation"
        self.__name = name
        self.__fig = plt.figure(name, figsize=(width, height))
        if mosaic is None:
            mosaic = """A"""
        self.__ax_dict = self.__fig.subplot_mosaic(mosaic)
        self.__static_lines = []
        self.__lines = []

        # plt.show()

    def reset(self):
        return tuple(self.lines)

    def __call__(self, *args, **kwargs):
        return tuple(self.lines)

    @property
    def figure(self):
        return self.__fig

    @property
    def panels(self):
        return self.__ax_dict

    @property
    def panel_names(self):
        return list(self.__ax_dict)

    @property
    def frames_per_second(self):
        return self.__fps

    @property
    def fig_width(self):
        return self.__width

    @property
    def fig_height(self):
        return self.__height

    @property
    def name(self):
        return self.__name

    @property
    def lines(self):
        return self.__lines

    @property
    def _static_lines(self):
        return self.__static_lines

    @staticmethod
    def add_directed_position(axis, colour='red', size=100):
        pos = axis.scatter([], [], marker=(3, 2, 0), s=size, c=colour, zorder=100)

        points = [0, 2, 3, 4, 6]
        vert = np.array(pos.get_paths()[0].vertices)[points]
        vert[0] *= 2
        codes = pos.get_paths()[0].codes[points]
        vert = np.hstack([vert, np.zeros((vert.shape[0], 1))])

        return Animation.Marker(pos, codes, vert)

    @staticmethod
    def update_directed_position(marker, x, y, ori):

        marker.position.set_offsets(np.array([x, y]))

        vertices = R.from_euler('Z', ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(marker.vert)
        marker.position.set_paths((Path(vertices[:, :2], marker.codes),))

    def __del__(self):
        plt.close(self.__fig)


class GradientVectorAnimation(Animation):

    def __init__(self, gradient, levels=10, resolution=101, max_time=1000, vectors=False, *args, **kwargs):
        kwargs.setdefault("mosaic", """A""")
        super().__init__(*args, **kwargs)

        self.max_time = max_time
        self.__gradient = gradient
        self.__iteration = 0

        if vectors:
            self.init_panel = self.init_panel_vectors
            self.update_panel = self.update_panel_vectors
        else:
            self.init_panel = self.init_panel_responses
            self.update_panel = self.update_panel_responses

        grid = np.zeros((resolution, resolution), dtype=float)
        for i, x in enumerate(np.linspace(gradient.minx - 2, gradient.maxx + 2, resolution)):
            for j, y in enumerate(np.linspace(gradient.miny - 2, gradient.maxy + 2, resolution)):
                grid[i, j] = self.__gradient(x, y)

        cont = self.panels["A"].contourf(np.linspace(gradient.minx - 2, gradient.maxx + 2, resolution),
                                         np.linspace(gradient.miny - 2, gradient.maxy + 2, resolution),
                                         grid.T, levels=levels, cmap='YlOrBr', vmin=0, vmax=1)
        self._static_lines.append(cont)

        # self.panels["A"].plot(gradient.route[:, 0], gradient.route[:, 1], 'k-', lw=2)
        # self.line, = self.panels["A"].plot([], [], 'k-', lw=1, alpha=0.5)
        self.quiver = []
        for _ in range(max_time + 1):
            self.quiver.append(self.panels["A"].quiver(-100, -100, 1, 0, color='black', zorder=50,
                                                       lw=.5, alpha=0.2, headwidth=1, headlength=2))
        self.panels["A"].set_aspect("equal")
        self.panels["A"].set_xlim([self.__gradient.minx - 2, self.__gradient.maxx + 2])
        self.panels["A"].set_ylim([self.__gradient.miny - 2, self.__gradient.maxy + 2])
        # self.target_bl = self.panels["A"].scatter([0], [0], s=50, c='red', marker='x')
        # self.target_br = self.panels["A"].scatter([0], [0], s=50, c='green', marker='x')
        # self.target_cl = self.panels["A"].scatter([0], [0], s=50, c='red', marker='o')
        # self.target_cr = self.panels["A"].scatter([0], [0], s=50, c='green', marker='o')
        self.marker = self.add_directed_position(self.panels["A"], colour='orange', size=100)
        self.update_directed_position(self.marker, x=0, y=0, ori=R.from_euler("Z", 0))
        self.lines.append(self.marker.position)
        # self.lines.append(self.line)
        self.lines.extend(self.quiver)
        # self.lines.append(self.target_bl)
        # self.lines.append(self.target_br)
        # self.lines.append(self.target_cl)
        # self.lines.append(self.target_cr)

        self.line_g = None
        self.res_epg = None
        self.res_pfn = None
        self.res_hd = None
        self.res_fc = None
        self.res_pfl2 = None
        self.res_pfl3 = None
        self.line_tan = None
        self.line_tar = None
        self.line_vec = None
        self.line_ang = None
        self.line_phi = None
        self.line_dis = None

        if "B" in self.panel_names:
            self.line_g, = self.panels["B"].plot([], [], 'k-', lw=2)
            self.panels["B"].set_ylim(-.1, 1.1)
            self.panels["B"].set_ylabel("familiarity")
            self.lines.append(self.line_g)

        if "E" in self.panel_names:
            self.res_hdb = self.init_panel("E", title=r"$h\Delta$", bottom=False, cmap='Reds', pop_size=8)
            self.lines.append(self.res_hdb)

        if "F" in self.panel_names:
            self.res_pfl2 = self.init_panel("F", title=r"$PFL2$", bottom=False, cmap='Oranges', pop_size=8)
            self.lines.append(self.res_pfl2)

        if "G" in self.panel_names:
            self.res_pfn = self.init_panel("G", title=r"$PFN_{d/v}$", bottom=False, cmap='Purples')
            self.lines.append(self.res_pfn)

        if "I" in self.panel_names:
            self.res_hd = self.init_panel("I", title=r"$h\Delta$", bottom=False, cmap='Reds', pop_size=8)
            self.lines.append(self.res_hd)

        if "J" in self.panel_names:
            self.res_pfl3 = self.init_panel("J", title=r"$PFL3$", bottom=False, cmap='Greens')
            self.lines.append(self.res_pfl3)

        if "L" in self.panel_names:
            self.res_fc = self.init_panel("L", title=r"$FC$", bottom=False, cmap='Oranges', pop_size=8)
            self.lines.append(self.res_fc)

        if "O" in self.panel_names:
            self.res_epg = self.init_panel("O", title=r"$EPG$", cmap="Blues", bottom=False)
            self.lines.append(self.res_epg)

        if "M" in self.panel_names:
            self.line_tar, = self.panels["M"].plot([], [], color='orange', lw=2)
            self.line_vec, = self.panels["M"].plot([], [], 'r-', lw=2)
            self.line_tan, = self.panels["M"].plot([], [], 'k--', lw=2)
            self.line_ang, = self.panels["M"].plot([], [], 'k-', lw=1, alpha=0.5)
            self.line_phi, = self.panels["M"].plot([], [], 'r-', lw=1, alpha=0.5)  # direction of motion
            self.panels["M"].set_ylim(-np.pi, np.pi)
            self.panels["M"].set_yticks([-np.pi, 0, np.pi])
            self.panels["M"].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
            self.panels["M"].set_ylabel("yaw")
            self.lines.append(self.line_tan)
            self.lines.append(self.line_tar)
            self.lines.append(self.line_vec)
            self.lines.append(self.line_ang)
            self.lines.append(self.line_phi)

        if "N" in self.panel_names:
            self.line_dis, = self.panels["N"].plot([], [], 'k-', lw=2)
            self.panels["N"].set_ylim(-.1, 3.1)
            self.panels["N"].set_ylabel("distance")
            self.lines.append(self.line_dis)

        plt.tight_layout()

    def reset(self, agent=None):
        self.__iteration = 0

        self.update_agent(agent)
        for q in self.quiver:
            q.set_offsets([-100, -100])
            q.set_UVC(1, 0)

        if self.line_g is not None:
            self.line_g.set_data([], [])
            self.panels["B"].set_xlim(0, 0)

        if self.line_dis is not None:
            self.line_dis.set_data([], [])
            self.panels["N"].set_xlim(0, 0)
            self.panels["N"].set_ylim(-.1, +.1)

        if self.line_tan is not None:
            self.line_tan.set_data([], [])
            self.panels["M"].set_xlim(0, 0)

        if self.line_ang is not None:
            self.line_ang.set_data([], [])
            self.panels["M"].set_xlim(0, 0)

        if self.line_phi is not None:
            self.line_phi.set_data([], [])
            self.panels["M"].set_xlim(0, 0)

        if self.line_vec is not None:
            self.line_vec.set_data([], [])
            self.panels["M"].set_xlim(0, 0)

        if self.line_tar is not None:
            self.line_tar.set_data([], [])
            self.panels["M"].set_xlim(0, 0)

        if self.res_epg is not None:
            self.res_epg.set_array(-100 * np.ones((16, self.max_time + 1), dtype='float32'))
            self.panels["O"].set_xlim(-0.5, 0.5)

        if self.res_pfn is not None:
            self.res_pfn.set_array(-100 * np.ones((16, self.max_time + 1), dtype='float32'))
            self.panels["G"].set_xlim(-0.5, 0.5)

        if self.res_hd is not None:
            self.res_hd.set_array(-100 * np.ones((16, self.max_time + 1), dtype='float32'))
            self.panels["I"].set_xlim(-0.5, 0.5)

        if self.res_fc is not None:
            self.res_fc.set_array(-100 * np.ones((16, self.max_time + 1), dtype='float32'))
            self.panels["L"].set_xlim(-0.5, 0.5)

        if self.res_pfl3 is not None:
            self.res_pfl3.set_array(-100 * np.ones((16, self.max_time + 1), dtype='float32'))
            self.panels["J"].set_xlim(-0.5, 0.5)

        return super(GradientVectorAnimation, self).reset()

    def __call__(self, agent, grad=None, epg=None, pfn=None,
                 h_delta=None, fc=None, pfl2=None, pfl3=None, phi=None):

        if self.__iteration >= self.max_time:
            return self.reset(agent)

        lg.logger.debug(f"i = {self.__iteration:03d}", end="; ")
        self.update_agent(agent)
        self.update_target(agent, np.exp(agent.yaw * 1j) * pfl2, np.exp(1j * agent.yaw) * (fc - epg * np.abs(fc)))
        self.update_gradient(grad)
        self.update_distance(self.__gradient.last_distance)
        self.update_epg(epg)
        self.update_pfn(pfn)
        self.update_hd(h_delta)
        self.update_df(fc)
        self.update_pfl2(pfl2)
        self.update_pfl3(pfl3)
        # self.update_yaw(agent.yaw, phi=phi, target=[np.angle(dfc[1] + dfc[0]), np.angle(dfb[1] + dfb[0])])
        self.update_yaw(agent.yaw, target=np.angle(np.sum(fc)))

        self.__iteration += 1

        return super(GradientVectorAnimation, self).__call__()

    def update_agent(self, agent):
        self.update_directed_position(self.marker, x=agent.x, y=agent.y, ori=agent.ori)
        # x, y = self.line.get_data()
        # self.line.set_data(list(x) + [agent.x], list(y) + [agent.y])
        self.quiver[self.__iteration].set_offsets([[agent.x, agent.y]])
        self.quiver[self.__iteration].set_UVC([-np.sin(agent.yaw)], [np.cos(agent.yaw)])
        # self.panels["A"].quiver(agent.x, agent.y, -np.sin(agent.yaw), np.cos(agent.yaw))

    def update_target(self, agent, dfb, dfc):
        # self.target_bl.set_offsets([agent.x + np.real(np.exp(1j * np.pi/2) * dfb[0]),
        #                             agent.y + np.imag(np.exp(1j * np.pi/2) * dfb[0])])
        # self.target_br.set_offsets([agent.x + np.real(np.exp(1j * np.pi/2) * dfb[1]),
        #                             agent.y + np.imag(np.exp(1j * np.pi/2) * dfb[1])])
        # self.target_cl.set_offsets([agent.x + np.real(np.exp(1j * np.pi/2) * dfc[0]),
        #                             agent.y + np.imag(np.exp(1j * np.pi/2) * dfc[0])])
        # self.target_cr.set_offsets([agent.x + np.real(np.exp(1j * np.pi/2) * dfc[1]),
        #                             agent.y + np.imag(np.exp(1j * np.pi/2) * dfc[1])])
        pass

    def update_gradient(self, grad):
        if self.line_g is not None and grad is not None:
            t, g = self.line_g.get_data()
            t_last = len(t) / self.frames_per_second
            self.line_g.set_data(list(t) + [t_last], list(g) + [grad])
            self.panels["B"].set_xlim(0, t_last)

    def update_distance(self, distance):
        if self.line_dis is not None and distance is not None:
            t, d = self.line_dis.get_data()
            t_last = len(t) / self.frames_per_second
            data = list(d) + [distance]
            self.line_dis.set_data(list(t) + [t_last], data)
            self.panels["N"].set_xlim(0, t_last)
            self.panels["N"].set_ylim(-.1 * np.max(data), 1.1 * np.max(data))

    def update_yaw(self, yaw, phi=None, target=None):
        if self.line_ang is not None and yaw is not None:
            t_ang, ang = self.line_ang.get_data()
            t_phi, phi_ = self.line_phi.get_data()
            t_tar, tar = self.line_tar.get_data()
            t_vec, vec = self.line_vec.get_data()
            t_tan, tan = self.line_tan.get_data()
            t_last = np.isfinite(t_ang).sum() / self.frames_per_second

            def add_value(times, values, val_c):
                # fix potential jump
                if len(values) > 0 and np.abs(val_c - values[-1]) > np.pi:
                    t_llast = times[-1]
                    times.append(t_last)
                    if val_c - values[-1] > 0:
                        values.append(val_c - 2 * np.pi)
                    else:
                        values.append(val_c + 2 * np.pi)
                    times.append(t_last)
                    values.append(np.nan)
                    times.append(t_llast)
                    if val_c - values[-2] > 0:
                        values.append(values[-3] + 2 * np.pi)
                    else:
                        values.append(values[-3] - 2 * np.pi)
                times.append(t_last)
                values.append(val_c)

            t_tan = list(t_tan)
            tan = list(tan)
            tan_c = (self.__gradient.last_tangent + np.pi) % (2 * np.pi) - np.pi
            add_value(t_tan, tan, tan_c)
            self.line_tan.set_data(t_tan, tan)

            t_ang = list(t_ang)
            ang = list(ang)
            ang_c = (yaw + np.pi) % (2 * np.pi) - np.pi
            add_value(t_ang, ang, ang_c)
            self.line_ang.set_data(t_ang, ang)

            if phi is not None:
                t_phi = list(t_phi)
                phi_ = list(phi_)
                phi_c = (phi + np.pi) % (2 * np.pi) - np.pi
                add_value(t_phi, phi_, phi_c)
                self.line_phi.set_data(t_phi, phi_)

            if target is not None:
                t_tar = list(t_tar)
                tar = list(tar)
                if type(target) is list:
                    t_vec = list(t_vec)
                    vec = list(vec)
                    tar_c = (target[0] + np.pi) % (2 * np.pi) - np.pi
                    vec_c = (target[1] + np.pi) % (2 * np.pi) - np.pi
                    add_value(t_vec, vec, vec_c)
                    self.line_vec.set_data(t_vec, vec)
                else:
                    tar_c = (target + np.pi) % (2 * np.pi) - np.pi
                add_value(t_tar, tar, tar_c)
                self.line_tar.set_data(t_tar, tar)

            self.panels["M"].set_xlim(0, t_last)

    def update_epg(self, epg):
        self.update_panel("O", self.res_epg, epg)

    def update_pfn(self, pfn):
        self.update_panel("G", self.res_pfn, pfn)

    def update_hd(self, hd):
        self.update_panel("I", self.res_hd, hd)

    def update_df(self, fc):
        self.update_panel("L", self.res_fc, fc)

    def update_pfl2(self, pfl3):
        self.update_panel("F", self.res_pfl2, pfl3)

    def update_pfl3(self, pfl3):
        self.update_panel("J", self.res_pfl3, pfl3)

    def update_panel_vectors(self, panel, handler, new_value):
        h1, h2 = handler
        if handler is not None and new_value is not None:
            t1, v1 = h1.get_data()
            t2, v2 = h2.get_data()
            if len(t1) > 0:
                nb_nans = np.isnan(v1).sum()
                t_last = (len(v1) - 3 * nb_nans) / self.frames_per_second
            else:
                t_last = 0.
            new_v1 = np.angle(new_value[0])
            t1, v1 = self.add_value(list(t1), t_last, list(v1), new_v1)
            h1.set_data(t1, v1)
            new_v2 = np.angle(new_value[1])
            t2, v2 = self.add_value(list(t2), t_last, list(v2), new_v2)
            h2.set_data(t2, v2)
            self.panels[panel].set_xlim(0, t_last)

    def update_panel_responses(self, panel, handler, new_value):
        if handler is not None and new_value is not None:
            values = handler.get_array()
            t = int(np.all(values > -100, axis=0).sum())

            angles = -np.linspace(-np.pi, np.pi, 8, endpoint=False)
            nb_col = values.shape[0] // 8
            if nb_col > 1:
                abs_0, ang_0 = np.abs(new_value[0]), np.angle(new_value[0])
                abs_1, ang_1 = np.abs(new_value[1]), np.angle(new_value[1])
                abs_m = 1
                # abs_m = np.maximum(abs_0, abs_1)
                values[:, t] = np.r_[abs_0 / abs_m * np.cos(ang_0 + angles), abs_1 / abs_m * np.cos(ang_1 + angles)]
            else:
                values[:, t] = np.abs(new_value) * np.cos(np.angle(new_value) + angles)
            handler.set_array(values)
            self.panels[panel].set_xlim(-0.5, t + 0.5)

    def init_panel_vectors(self, panel, title="", left=True, bottom=True, cmap=None):
        ax1, = self.panels[panel].plot([], 'r-', lw=1)
        ax2, = self.panels[panel].plot([], 'g-', lw=1)
        self.panels[panel].plot([0, self.max_time / self.frames_per_second], [np.pi, np.pi], 'k-', lw=0.5)
        self.panels[panel].set_ylim(-np.pi, np.pi)
        if left:
            self.panels[panel].set_yticks([-np.pi, 0, np.pi])
            self.panels[panel].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        else:
            self.panels[panel].set_yticks([-np.pi, 0, np.pi])
            self.panels[panel].set_yticklabels(["", "", ""])
        if not bottom:
            self.panels[panel].set_xticks([])
            self.panels[panel].set_xticklabels([])
        self.panels[panel].set_ylabel(title)

        self.lines.append([ax1, ax2])

        return ax1, ax2

    def init_panel_responses(self, panel, title="", left=True, bottom=True, pop_size=16, cmap='inferno'):
        ax = self.panels[panel].imshow(
            -100 * np.ones((pop_size, self.max_time + 1), dtype='float32'), cmap=cmap, vmin=-1, vmax=1, origin="upper",
            interpolation="none", aspect="auto")
        self.panels[panel].set_ylim(-0.5, pop_size - 0.5)
        self.panels[panel].set_xlim(-0.5, self.max_time - 0.5)
        if left:
            if pop_size > 8:
                self.panels[panel].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5])
                self.panels[panel].set_yticklabels(["", "R", "", "L", ""])
            else:
                self.panels[panel].set_yticks([-0.5, pop_size // 2 - 0.5, pop_size - 0.5])
                self.panels[panel].set_yticklabels(["1", "", f"{pop_size}"])
        else:
            self.panels[panel].set_yticks([-0.5, 7.5, 15.5])
            self.panels[panel].set_yticklabels(["", "", ""])
        if not bottom:
            self.panels[panel].set_xticks([])
        self.panels[panel].set_ylabel(title)
        self.panels[panel].plot([-0.5, self.max_time - 0.5], [7.5, 7.5], 'k-', lw=0.5)

        self.lines.append(ax)

        return ax

    @staticmethod
    def add_value(times, t_last, values, val_c):
        # fix potential jump
        if len(values) > 0 and np.abs(val_c - values[-1]) > np.pi:
            t_llast = times[-1]
            times.append(t_last)
            if val_c - values[-1] > 0:
                values.append(val_c - 2 * np.pi)
            else:
                values.append(val_c + 2 * np.pi)
            times.append(t_last)
            values.append(np.nan)
            times.append(t_llast)
            if val_c - values[-2] > 0:
                values.append(values[-3] + 2 * np.pi)
            else:
                values.append(values[-3] - 2 * np.pi)
        times.append(t_last)
        values.append(val_c)

        return times, values


class GradientAnimation(Animation):

    def __init__(self, gradient, levels=10, resolution=101, max_time=1000, *args, **kwargs):
        kwargs.setdefault("mosaic", """A""")
        super().__init__(*args, **kwargs)

        self.max_time = max_time
        self.__gradient = gradient

        grid = np.zeros((resolution, resolution), dtype=float)
        for i, x in enumerate(np.linspace(gradient.minx-2, gradient.maxx+2, resolution)):
            for j, y in enumerate(np.linspace(gradient.miny-2, gradient.maxy+2, resolution)):
                grid[i, j] = gradient(x, y)

        cont = self.panels["A"].contourf(np.linspace(gradient.minx - 2, gradient.maxx + 2, resolution),
                                         np.linspace(gradient.miny - 2, gradient.maxy + 2, resolution),
                                         grid.T, levels=levels, cmap='YlOrBr', vmin=0, vmax=1)
        self.line, = self.panels["A"].plot([], [], 'k-', lw=1, alpha=0.5)
        self.panels["A"].set_aspect("equal")
        self._static_lines.append(cont)
        self.marker = self.add_directed_position(self.panels["A"], colour='black', size=100)
        self.update_directed_position(self.marker, x=0, y=0, ori=R.from_euler("Z", 0))
        self.lines.append(self.marker)
        self.lines.append(self.line)

        if "B" in self.panel_names:
            self.line_g, = self.panels["B"].plot([], [], 'k-', lw=2)
            self.panels["B"].set_ylim(-.1, 1.1)
            self.panels["B"].set_ylabel("gradient")
            self.lines.append(self.line_g)
        else:
            self.line_g = None

        if "C" in self.panel_names:
            self.res_pfn_d = self.init_panel_responses("C", title=r"$PFN_d$", bottom=False)
        else:
            self.res_pfn_d = None

        if "D" in self.panel_names:
            self.res_pfn_v = self.init_panel_responses("D", title=r"$PFN_v$", bottom=False)
        else:
            self.res_pfn_v = None

        if "E" in self.panel_names:
            self.res_hdb = self.init_panel_responses("E", title=r"$h\Delta B$", bottom=False)
        else:
            self.res_hdb = None

        if "F" in self.panel_names:
            self.res_pfl2 = self.init_panel_responses("F", title=r"$PFL2$")
        else:
            self.res_pfl2 = None

        if "G" in self.panel_names:
            self.res_pfn_a = self.init_panel_responses("G", title=r"$PFN_a$", bottom=False, left=False)
        else:
            self.res_pfn_a = None

        if "H" in self.panel_names:
            self.res_pfn_p = self.init_panel_responses("H", title=r"$PFN_p$", bottom=False, left=False)
        else:
            self.res_pfn_p = None

        if "I" in self.panel_names:
            self.res_hdc = self.init_panel_responses("I", title=r"$h\Delta C$", bottom=False, left=False)
        else:
            self.res_hdc = None

        if "J" in self.panel_names:
            self.res_pfl3 = self.init_panel_responses("J", title=r"$PFL3$", left=False)
        else:
            self.res_pfl3 = None

        if "K" in self.panel_names:
            self.res_dfb = self.init_panel_responses("K", title=r"$\Delta\Phi B$", bottom=False)
        else:
            self.res_dfb = None

        if "L" in self.panel_names:
            self.res_dfc = self.init_panel_responses("L", title=r"$\Delta\Phi C$", bottom=False, left=False)
        else:
            self.res_dfc = None

        if "O" in self.panel_names:
            self.res_epg = self.init_panel_responses("O", title=r"$EPG$")
        else:
            self.res_epg = None

        if "M" in self.panel_names:
            self.line_tar, = self.panels["M"].plot([], [], 'g-', lw=2)
            self.line_tan, = self.panels["M"].plot([], [], 'k--', lw=2)
            self.line_ang, = self.panels["M"].plot([], [], 'k-', lw=1, alpha=0.5)
            self.line_phi, = self.panels["M"].plot([], [], 'r-', lw=1, alpha=0.5)  # direction of motion
            self.panels["M"].set_ylim(-np.pi, np.pi)
            self.panels["M"].set_yticks([-np.pi, 0, np.pi])
            self.panels["M"].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
            self.panels["M"].set_ylabel("yaw")
            self.lines.append(self.line_tan)
            self.lines.append(self.line_tar)
            self.lines.append(self.line_ang)
            self.lines.append(self.line_phi)
        else:
            self.line_tan = None
            self.line_tar = None
            self.line_ang = None
            self.line_phi = None

        if "N" in self.panel_names:
            self.line_dis, = self.panels["N"].plot([], [], 'k-', lw=2)
            self.panels["N"].set_ylim(-.1, 3.1)
            self.panels["N"].set_ylabel("distance")
            self.lines.append(self.line_dis)
        else:
            self.line_dis = None

    def __call__(self, agent, grad=None, epg=None, pfn_d=None, pfn_v=None, hdb=None, dfb=None, pfl2=None,
                 pfn_a=None, pfn_p=None, hdc=None, dfc=None, pfl3=None, phi=None):
        self.update_agent(agent)
        self.update_gradient(grad)
        self.update_distance(self.__gradient.last_distance)
        self.update_epg(epg)
        self.update_pfn_d(pfn_d)
        self.update_pfn_v(pfn_v)
        self.update_hdb(hdb)
        self.update_dfb(dfb)
        self.update_pfl2(pfl2)
        self.update_pfn_a(pfn_a)
        self.update_pfn_p(pfn_p)
        self.update_hdc(hdc)
        self.update_dfc(dfc)
        self.update_pfl3(pfl3)

        population = np.maximum(dfc[-1][8:] + dfc[-1][:8], 0)
        v = np.sum(population * np.exp(-1j * np.linspace(0, 2 * np.pi, 8, endpoint=False)))

        self.update_yaw(agent.yaw, phi=phi, target=-np.angle(v))
        super(GradientAnimation, self).__call__()

    def update_agent(self, agent):
        self.update_directed_position(self.marker, x=agent.x, y=agent.y, ori=agent.ori)
        x, y = self.line.get_data()
        self.line.set_data(list(x) + [agent.x], list(y) + [agent.y])

    def update_gradient(self, grad):
        if self.line_g is not None and grad is not None:
            t, g = self.line_g.get_data()
            t_last = len(t) / self.frames_per_second
            self.line_g.set_data(list(t) + [t_last], list(g) + [grad])
            self.panels["B"].set_xlim(0, t_last)

    def update_distance(self, distance):
        if self.line_dis is not None and distance is not None:
            t, d = self.line_dis.get_data()
            t_last = len(t) / self.frames_per_second
            data = list(d) + [distance]
            self.line_dis.set_data(list(t) + [t_last], data)
            self.panels["N"].set_xlim(0, t_last)
            self.panels["N"].set_ylim(-.1 * np.max(data), 1.1 * np.max(data))

    def update_yaw(self, yaw, phi=None, target=None):
        if self.line_ang is not None and yaw is not None:
            t_ang, ang = self.line_ang.get_data()
            t_phi, phi_ = self.line_phi.get_data()
            t_tar, tar = self.line_tar.get_data()
            t_tan, tan = self.line_tan.get_data()
            t_last = np.isfinite(t_ang).sum() / self.frames_per_second

            def add_value(times, values, val_c):
                # fix potential jump
                if len(values) > 0 and np.abs(val_c - values[-1]) > np.pi:
                    t_llast = times[-1]
                    times.append(t_last)
                    if val_c - values[-1] > 0:
                        values.append(val_c - 2 * np.pi)
                    else:
                        values.append(val_c + 2 * np.pi)
                    times.append(t_last)
                    values.append(np.nan)
                    times.append(t_llast)
                    if val_c - values[-2] > 0:
                        values.append(values[-3] + 2 * np.pi)
                    else:
                        values.append(values[-3] - 2 * np.pi)
                times.append(t_last)
                values.append(val_c)

            t_tan = list(t_tan)
            tan = list(tan)
            tan_c = (self.__gradient.last_tangent + np.pi) % (2 * np.pi) - np.pi
            add_value(t_tan, tan, tan_c)
            self.line_tan.set_data(t_tan, tan)

            t_ang = list(t_ang)
            ang = list(ang)
            ang_c = (yaw + np.pi) % (2 * np.pi) - np.pi
            add_value(t_ang, ang, ang_c)
            self.line_ang.set_data(t_ang, ang)

            if phi is not None:
                t_phi = list(t_phi)
                phi_ = list(phi_)
                phi_c = (phi + np.pi) % (2 * np.pi) - np.pi
                add_value(t_phi, phi_, phi_c)
                self.line_phi.set_data(t_phi, phi_)

            if target is not None:
                t_tar = list(t_tar)
                tar = list(tar)
                tar_c = (target + np.pi) % (2 * np.pi) - np.pi
                add_value(t_tar, tar, tar_c)
                self.line_tar.set_data(t_tar, tar)

            self.panels["M"].set_xlim(0, t_last)

    def update_epg(self, epg):
        self.update_panel_responses("O", self.res_epg, epg)

    def update_pfn_d(self, pfn_d):
        self.update_panel_responses("C", self.res_pfn_d, pfn_d)

    def update_pfn_v(self, pfn_v):
        self.update_panel_responses("D", self.res_pfn_v, pfn_v)

    def update_hdb(self, hdb):
        self.update_panel_responses("E", self.res_hdb, hdb)

    def update_dfb(self, dfb):
        self.update_panel_responses("K", self.res_dfb, dfb)

    def update_pfl2(self, pfl3):
        self.update_panel_responses("F", self.res_pfl2, pfl3)

    def update_pfn_a(self, pfn_a):
        self.update_panel_responses("G", self.res_pfn_a, pfn_a)

    def update_pfn_p(self, pfn_p):
        self.update_panel_responses("H", self.res_pfn_p, pfn_p)

    def update_hdc(self, hdc):
        self.update_panel_responses("I", self.res_hdc, hdc)

    def update_dfc(self, dfc):
        self.update_panel_responses("L", self.res_dfc, dfc)

    def update_pfl3(self, pfl3):
        self.update_panel_responses("J", self.res_pfl3, pfl3)

    def update_panel_responses(self, panel, handler, new_value):
        if handler is not None and new_value is not None:
            handler.set_data(np.array(new_value).T)
            t_last = len(new_value) / self.frames_per_second
            if self.panels[panel].get_xticklabels()[-1].get_text() != "":
                self.panels[panel].set_xticklabels([f"{0:.1f}", f"{t_last:.1f}"])
            # self.panels[panel].set_xlim(-0.5, len(new_value) - 0.5)

    def init_panel_responses(self, panel, title="", left=True, bottom=True):
        ax = self.panels[panel].imshow(
            np.zeros((16, self.max_time), dtype='float32'), cmap='inferno', vmin=-1, vmax=1, origin="upper",
            interpolation="none", aspect="auto")
        self.panels[panel].set_ylim(-0.5, 15.5)
        self.panels[panel].set_xlim(-0.5, self.max_time - 0.5)
        if left:
            self.panels[panel].set_yticks([-0.5, 3.5, 7.5, 11.5, 15.5])
            self.panels[panel].set_yticklabels(["", "L", "", "R", ""])
        else:
            self.panels[panel].set_yticks([-0.5, 7.5, 15.5])
            self.panels[panel].set_yticklabels(["", "", ""])
        if bottom:
            self.panels[panel].set_xticks([0, self.max_time - 1])
            self.panels[panel].set_xticklabels(["0.0", "0.0"])
        else:
            self.panels[panel].set_xticks([0, self.max_time - 1])
            self.panels[panel].set_xticklabels(["", ""])
        self.panels[panel].set_title(title)
        self.panels[panel].plot([-0.5, self.max_time-0.5], [7.5, 7.5], 'k-', lw=0.5)

        self.lines.append(ax)

        return ax


class AnimationBase(object):

    def __init__(self, sim, fps=15, width=11, height=5, name=None):
        """
        Visualises the simulation and creates videos of it.

        Parameters
        ----------
        sim: SimulationBase
            the simulation to run
        fps: float, optional
            the frames per second of the visualisations. Default is 15 fps
        width: float, optional
            the width of the figure (in inches). Default is 11 inches
        height: float, optional
            the height of the figure (in inches). Default is 5 inches
        name: str, optional
            the name of the animation. Default is the name of the simulation + '-anim'
        """
        self._sim = sim
        self._fps = fps
        self._ani = None
        self._lines = []
        self.__lines = []
        self._iteration = 0

        if name is None:
            name = self.name
        else:
            sim.set_name(name)
        self._fig = plt.figure(name, figsize=(width, height))

    def __call__(self, save=False, save_type="gif", save_name=None, save_stats=True, show=True):
        """
        Creates the animation and runs the simulation. Saves the animation in the given file format (if applicable),
        saves the logged stats (if applicable) and visualises the animation (if applicable).

        Parameters
        ----------
        save: bool, optional
            if True, it saves the animation in the given format. Default is False
        save_type: str, optional
            the type of encoding for the saved animation. Default is 'gif'
        save_name: str, optional
            the name of the file where the animation will be saved to. Default is {animation-name}.{save_type}
        save_stats: bool, optional
            if True, it saves the logged statistics from the simulation. Default is True
        show: bool, optional
            if True, it visualises the animation live. Default is True
        """
        # self.sim.reset()
        self._animate(0)
        self._ani = animation.FuncAnimation(self.fig, self.__animate, init_func=self.__initialise,
                                            frames=self.nb_frames, interval=int(1000 / self._fps), blit=True)
        try:
            if save:
                if save_name is None:
                    save_name = "%s.%s" % (self.name, save_type.lower())
                filepath = os.path.join(__anim_dir__, save_name)
                lg.logger.info("Saving video in '%s'." % filepath)
                self.ani.save(filepath, fps=self._fps)

            if show:
                plt.show()
        except KeyboardInterrupt:
            lg.logger.error("Animation interrupted by keyboard!")
        finally:
            if save_stats:
                self._sim.save()

    def _initialise(self):
        """
        Initialises the animation by running the first step.
        """
        self.__lines = [line for line in self._lines if line is not None]
        self._animate(0)

    def _animate(self, i):
        """
        Animates the given iteration of the simulation.

        Parameters
        ----------
        i: int
            the iteration to animate.

        Returns
        -------
        float
            the time (in seconds) that the iteration needed to run

        Raises
        ------
        NotImplementedError
            this method has to be implemented by the sub-classes
        """
        raise NotImplementedError()

    def __initialise(self):
        """
        Initialises the animation and returns the produced lines of the figure.

        Returns
        -------
        tuple
            the figure lines
        """
        self._initialise()
        return tuple(self.__lines)

    def __animate(self, i):
        """
        Runs the animation of the given iteration and prints a message showing the progress.

        Parameters
        ----------
        i: int
            the iteration to run

        Returns
        -------
        tuple
            the lines of the figure
        """
        time = self._animate(i)
        if isinstance(time, float):
            lg.logger.info(self.sim.message() + " - time: %.2f sec" % time)
        return tuple(self.__lines)

    @property
    def fig(self):
        """
        The figure where the animation is illustrated.

        Returns
        -------
        plt.Figure
        """
        return self._fig

    @property
    def sim(self):
        """
        The simulation that runs in the background.

        Returns
        -------
        SimulationBase
        """
        return self._sim

    @property
    def nb_frames(self):
        """
        The total number of frames of the animation.

        Returns
        -------
        int
        """
        return self._sim.nb_frames

    @property
    def fps(self):
        """
        The frames per second of the animation.

        Returns
        -------
        float
        """
        return self._fps

    @property
    def ani(self):
        """
        The matplotlib animation instance.

        Returns
        -------
        animation.Animation
        """
        return self._ani

    @property
    def name(self):
        """
        The name of the animation.

        Returns
        -------
        str
        """
        return self.sim.name + "-anim"


class RouteAnimation(AnimationBase):

    def __init__(self, sim, cmap="Greens_r", *args, **kwargs):
        """
        Animation of the route simulation. It visualised the current view of the agent, and its current and previous
        positions on the map along with the vegetation of the world.

        Parameters
        ----------
        sim: RouteSimulation
            the route simulation instance
        cmap: str, optional
            the colour map to show the intensity of the ommatidia photo-receptors' responses. Default it 'Greens_r'
        """
        kwargs.setdefault("width", 7)
        kwargs.setdefault("height", 3)
        super().__init__(sim, *args, **kwargs)

        ax_dict = self.fig.subplot_mosaic(
            """
            AB
            CB
            """
        )
        omm = create_eye_axis(sim.eye, cmap=cmap, ax=ax_dict['A'])
        line, _, pos, self._marker = create_map_axis(world=sim.world, ax=ax_dict['B'])[:4]
        create_side_axis(sim.world, ax=ax_dict['C'])

        plt.tight_layout()

        self._lines.extend([omm, line, pos])

        omm.set_array(sim.responses)

    def _animate(self, i):
        """
        Runs the given simulation iteration and updates the values of the elements in the figure.

        Parameters
        ----------
        i: int
            the iteration ID of the current step
        """
        self.sim.step(i)

        self.omm.set_array(self.sim.responses)
        self.line.set_data(self.sim.route[:(i+1), 1], self.sim.route[:(i+1), 0])
        self.pos.set_offsets(np.array([self.sim.eye.y, self.sim.eye.x]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.eye.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        # id = {40: 1, 400: 2, 800: 3, 819: 4}
        # lg.logger.debug(f"Iteration: {i}")
        # if i in id:
        #     fig = plt.figure(f"{self.name}-view-{id[i]}", figsize=(7, 3))
        #     ax_dict = fig.subplot_mosaic(
        #         """
        #         AABB
        #         CDBB
        #         """
        #     )
        #
        #     omm = create_eye_axis(self.sim.eye, cmap="Greens_r", ax=ax_dict['A'])
        #     omm_t = create_sphere_eye_axis(self.sim.eye, cmap="Greens_r", side="top", ax=ax_dict['C'])
        #     omm_s = create_sphere_eye_axis(self.sim.eye, cmap="Greens_r", side="side", ax=ax_dict['D'])
        #     line, _, pos, self._marker = create_map_axis(world=self.sim.world, ax=ax_dict['B'])[:4]
        #
        #     omm.set_array(self.sim.responses)
        #     omm_t.set_array(self.sim.responses[self.sim.eye.omm_xyz[:, 2] >= 0])
        #     omm_s.set_array(self.sim.responses[self.sim.eye.omm_xyz[:, 1] >= 0])
        #     line.set_data(self.sim.route[:(i + 1), 1], self.sim.route[:(i + 1), 0])
        #     pos.set_offsets(np.array([self.sim.eye.y, self.sim.eye.x]))
        #     pos.set_paths((Path(vertices[:, :2], codes),))
        #
        #     plt.show()

    @property
    def omm(self):
        """
        The collection of ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[0]

    @property
    def line(self):
        """
        The line representing the path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[1]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[2]

    @property
    def sim(self):
        """
        The simulation instance that drives the agent to follow the given route.

        Returns
        -------
        RouteSimulation
        """
        return self._sim


class MapResponsesAnimation(AnimationBase):
    def __init__(self, sim, *args, **kwargs):
        kwargs.setdefault('fps', 100)
        super().__init__(sim, *args, **kwargs)

        has_visual = hasattr(sim, 'agent') and isinstance(sim.agent, VisualProcessingAgent)
        has_pi = hasattr(sim, 'agent') and isinstance(sim.agent, CentralComplexAgent)
        mosaic = """
        """
        if has_visual:
            mosaic += """
            BBBBAAAA
            BBBBAAAA
            """

        if has_pi and has_visual:
            mosaic += """CDDDAAAA
            FFEEAAAA
            GGHHAAAA
            IIJJAAAA
            KKNNMMLL
            """
        elif has_visual:
            mosaic += """IILLAAAA
            JJMMAAAA
            KKNNAAAA
            """
        elif has_pi:
            mosaic += """
            CDDDAAAA
            EEEEAAAA
            FFFFAAAA
            GGGGAAAA
            HHHHAAAA
            """

        ax_dict = self.fig.subplot_mosaic(mosaic)

        if isinstance(sim, TwoSourcePathIntegrationSimulation):
            nest = sim.central_point[:2]
            feeders = [sim.feeder_a[:2], sim.feeder_b[:2]]
        elif isinstance(sim, CentralPointNavigationSimulationBase):
            nest = sim.central_point[:2]
            feeders = [sim.distant_point[:2]]
        else:
            nest = None
            feeders = None

        all_lines = create_map_axis(world=sim.world, ax=ax_dict["A"], nest=nest, feeders=feeders)
        line_c, line_b, pos, self._marker, cal, poi, feeders_text = all_lines[:7]
        omm, pol, tb1, cl1, cpu1, cpu4, cpu4mem, pn, mbon, dan, dist, cap, fam = [None] * 13

        if "B" in mosaic and hasattr(sim.agent, "eye"):
            omm = create_eye_axis(sim.agent.eye, cmap="Greens_r", ax=ax_dict["B"])
        if "C" in mosaic and hasattr(sim.agent, "pol_sensor"):
            pol = create_dra_axis(sim.agent.pol_sensor, cmap="coolwarm", ax=ax_dict["C"])
            pol.set_array(sim.r_pol)
        if "D" in mosaic:
            tb1 = create_tb1_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["D"])
        if "E" in mosaic:
            cl1 = create_cl1_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["E"])
        if "F" in mosaic:
            cpu1 = create_cpu1_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["F"])
        if "G" in mosaic:
            cpu4 = create_cpu4_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["G"])
        if "H" in mosaic:
            cpu4mem = create_cpu4_mem_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["H"])
        if "I" in mosaic:
            pn = create_pn_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["I"])
        if "J" in mosaic:
            mbon = create_mbon_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["J"])
        if "K" in mosaic:
            dan = create_dan_history(sim.agent, self.nb_frames, cmap="coolwarm", ax=ax_dict["K"])
        if "L" in mosaic:
            dist = create_single_line_history(self.nb_frames, title="d_nest (m)", ylim=8, ax=ax_dict["L"])
        if "M" in mosaic:
            cap = create_free_space_history(self.nb_frames, ax=ax_dict["M"])
        if "N" in mosaic:
            fam = create_familiarity_history(self.nb_frames, ax=ax_dict["N"])

        self._lines.extend([line_c, line_b, pos, cal, poi,
                            omm, pol, tb1, cl1, cpu1, cpu4, cpu4mem,
                            pn, mbon, dan, dist, cap, fam] + feeders_text)

        plt.tight_layout()

    def _animate(self, i):
        """
        Runs the current iteration of the simulation and updates the data from the figure.

        Parameters
        ----------
        i : int
        """

        if i == 0:
            if self.line_b is not None:
                self.line_b.set_data([], [])
            if self.poi is not None:
                self.poi.set_offsets(np.empty((0, 2)))

            xyz = self.sim.reset()
            # xyz = self.sim.reset(nb_samples_calibrate=3)
            if xyz is not None and self.cal is not None:
                self.cal.set_offsets(np.array(xyz)[:, [1, 0]])
        elif "xyz_out" in self.sim.stats and self.line_b is not None:
            xyz = np.array(self.sim.stats["xyz_out"])
            self.line_b.set_data(xyz[..., 1], xyz[..., 0])

        time = self.sim.step(i)

        x, y = np.array(self.sim.stats["xyz"])[..., :2].T
        if self.line_c is not None:
            self.line_c.set_data(y, x)
        if self.pos is not None:
            self.pos.set_offsets(np.array([y[-1], x[-1]]))

            vert, codes = self._marker
            vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
            self.pos.set_paths((Path(vertices[:, :2], codes),))
        if (self.poi is not None and "replace" in self.sim.stats and
                len(self.sim.stats["replace"]) > 0 and self.sim.stats["replace"][-1]):
            points = self.poi.get_offsets()
            self.poi.set_offsets(np.vstack([points, np.array([[y[-1], x[-1]]])]))

        if self.omm is not None and "ommatidia" in self.sim.stats:
            r = self.sim.stats["ommatidia"][-1].mean(axis=1)
            self.omm.set_array(r.T.flatten())
        if self.pol is not None and "POL" in self.sim.stats:
            self.pol.set_array(np.array(self.sim.stats["POL"][-1]))
        if self.tb1 is not None and "TB1" in self.sim.stats:
            tb1 = np.zeros((self.sim.r_tb1.shape[0], self.nb_frames), dtype=float)
            tb1[:, :i + 1] = np.array(self.sim.stats["TB1"]).T
            self.tb1.set_array(tb1)
        if self.cl1 is not None and "CL1" in self.sim.stats:
            cl1 = np.zeros((self.sim.r_cl1.shape[0], self.nb_frames), dtype=float)
            cl1[:, :i + 1] = np.array(self.sim.stats["CL1"]).T
            self.cl1.set_array(cl1)
        if self.cpu1 is not None and "CPU1" in self.sim.stats:
            cpu1 = np.zeros((self.sim.r_cpu1.shape[0], self.nb_frames), dtype=float)
            cpu1[:, :i + 1] = np.array(self.sim.stats["CPU1"]).T
            self.cpu1.set_array(cpu1)
        if self.cpu4 is not None and "CPU4" in self.sim.stats:
            cpu4 = np.zeros((self.sim.r_cpu4.shape[0], self.nb_frames), dtype=float)
            cpu4[:, :i + 1] = np.array(self.sim.stats["CPU4"]).T
            self.cpu4.set_array(cpu4)
        if self.cpu4mem is not None and "CPU4mem" in self.sim.stats:
            cpu4mem = np.zeros((self.sim.cpu4_mem.shape[0], self.nb_frames), dtype=float)
            cpu4mem[:, :i + 1] = np.array(self.sim.stats["CPU4mem"]).T
            self.cpu4mem.set_array(cpu4mem)

        if self.pn is not None and "PN" in self.sim.stats:
            pn = np.zeros((self.sim.agent.mushroom_body.nb_cs, self.nb_frames), dtype=float)
            pn[:, :i + 1] = np.array(self.sim.stats["PN"]).T
            self.pn.set_array(pn)
        if self.mbon is not None and "MBON" in self.sim.stats:
            mbon = np.zeros((self.sim.agent.mushroom_body.nb_mbon, self.nb_frames), dtype=float)
            mbon[:, :i + 1] = np.array(self.sim.stats["MBON"]).T
            self.mbon.set_array(mbon)
        if self.dan is not None and "DAN" in self.sim.stats:
            dan = np.zeros((self.sim.agent.mushroom_body.nb_dan, self.nb_frames), dtype=float)
            dan[:, :i + 1] = np.array(self.sim.stats["DAN"]).T
            self.dan.set_array(dan)
        if self.dist is not None and "L" in self.sim.stats:
            dist = np.full(self.nb_frames, np.nan, dtype=float)
            if "L_out" in self.sim.stats:
                out = np.array(self.sim.stats["L_out"]).T
            else:
                out = np.array([])
            dist[:i + 1] = np.r_[out, np.array(self.sim.stats["L"]).T]
            self.dist.set_data(np.arange(self.nb_frames), dist)
        if self.capacity is not None and "capacity" in self.sim.stats:
            cap = np.full(self.nb_frames, np.nan, dtype=float)
            cap[:i + 1] = np.array(self.sim.stats["capacity"]).T
            self.capacity.set_data(np.arange(self.nb_frames), cap)
        if self.fam is not None and "familiarity" in self.sim.stats:
            fam = np.full(self.nb_frames, np.nan, dtype=float)
            fam[:i + 1] = np.array(self.sim.stats["familiarity"]).T
            self.fam.set_data(np.arange(self.nb_frames), fam)

        return time

    @property
    def sim(self):
        """

        Returns
        -------
        NavigationSimulation
        """
        return super().sim

    @property
    def line_c(self):
        """
        The line representing the ongoing path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[0]

    @property
    def line_b(self):
        """
        The line representing the finished path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[1]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[2]

    @property
    def cal(self):
        """
        The positions the figure used for calibration in.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[3]

    @property
    def poi(self):
        """
        The positions in the figure where the agent was brought back to the route.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[4]

    @property
    def omm(self):
        """
        The collection of ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[5]

    @property
    def pol(self):
        """
        The collection of the DRA ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[6]

    @property
    def tb1(self):
        """
        The history of the TB1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[7]

    @property
    def cl1(self):
        """
        The history of the CL1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[8]

    @property
    def cpu1(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[9]

    @property
    def cpu4(self):
        """
        The history of the CPU4 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[10]

    @property
    def cpu4mem(self):
        """
        The history of the CPU4 memory in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[11]

    @property
    def pn(self):
        """
        The history of the PN response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[12]

    @property
    def mbon(self):
        """
        The history of the MBON response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[13]

    @property
    def dan(self):
        """
        The history of the DAN response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[14]

    @property
    def dist(self):
        """
        The history of the distance from the goal (nest) in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[15]

    @property
    def capacity(self):
        """
        The history of the memory capacity in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[16]

    @property
    def fam(self):
        """
        The history of familiarity in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[17]

    @property
    def feeders_text(self):
        """
        The text next to each feeder in the map.

        Returns
        -------
        list[matplotlib.text.Text]
        """
        return self._lines[18:]


class VisualNavigationAnimation(AnimationBase):

    def __init__(self, sim, cmap="Greens_r", show_history=True, show_weights=False, *args, **kwargs):
        """
        Animation for the visual navigation simulation. It visualised the current view of the agent, its current and
        previous positions on the map along with the vegetation, and the statistics according to the parameters.

        Parameters
        ----------
        sim: VisualNavigationSimulation
            the visual navigation simulation instance
        cmap: str, optional
            the colour map to be used for the responses from the ommatidia. Default is 'Greens_r'
        show_history: bool, optional
            if True, the whole history of the neurons is visualised instead of just the current values. Default is True
        show_weights: bool, optional
            if True, the KC-MBON synaptic weights are visualised instead of the the KC responses. Default is False
        """
        super().__init__(sim, *args, **kwargs)

        if show_history:
            ax_dict = self.fig.subplot_mosaic(
                """
                AABB
                AABB
                CFBB
                DGBB
                DHBB
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
            world=sim.world, ax=ax_dict["B"], nest=sim.route[-1, :2], feeders=[sim.route[0, :2]])[:6]

        self._lines.extend([omm, line_c, line_b, pos, cal, poi])

        if show_history:
            pn = create_pn_history(sim.agent, self.nb_frames, sep=sim.route.shape[0],
                                   cmap="Greys", ax=ax_dict["C"])
            kc = create_kc_history(sim.agent, self.nb_frames, sep=sim.route.shape[0],
                                   cmap="Greys", ax=ax_dict["D"])
            # fam_all, fam_line = create_familiarity_response_history(
            #     sim.agent, self.nb_frames, sep=sim.route.shape[0], cmap="Greys", ax=ax_dict["E"])
            dist = create_single_line_history(
                self.nb_frames, sep=sim.route.shape[0], title="d_nest (m)", ylim=8, ax=ax_dict["F"])
            cap = create_free_space_history(self.nb_frames, sep=sim.route.shape[0], ax=ax_dict["G"])
            fam = create_familiarity_history(self.nb_frames, sep=sim.route.shape[0], ax=ax_dict["H"])

            self._lines.extend([pn, kc, fam, dist, cap])
            # self._lines.extend([pn, kc, fam, dist, cap, fam_all, fam_line])
        else:
            pn, kc, fam = create_mem_axis(sim.agent, cmap="Greys", ax=ax_dict["C"])

            self._lines.extend([pn, kc, fam])

        plt.tight_layout()

        self._show_history = show_history
        self._show_weights = show_weights

    def _animate(self, i):
        """
        Runs the current iteration of the simulation and updates the data for the figure.

        Parameters
        ----------
        i: int
            the current iteration
        """
        if i == 0:
            xyzs = self.sim.reset()
            if xyzs is not None:
                self.cal.set_offsets(np.array(xyzs)[:, [1, 0]])
        elif i == self.sim.route.shape[0]:
            self.line_b.set_data(np.array(self.sim.stats["xyz"])[..., 1], np.array(self.sim.stats["xyz"])[..., 0])

        time = self.sim.step(i)

        r = self.sim.stats["ommatidia"][-1].mean(axis=1)
        x, y = np.array(self.sim.stats["xyz"])[..., :2].T

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
            pn[:, i] = self.sim.mem.r_cs[0, 0].T.flatten()
            self.pn.set_array(pn)
            kc = self.kc.get_array()
            if self._show_weights:
                kc[:, i] = self.sim.mem.w_k2m.T.flatten()
            else:
                kc[:, i] = self.sim.mem.r_kc[0, 0].T.flatten()
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
            # if self.fam_all is not None:
            #     fam_all = self.fam_all.get_array()
            #     fam_all[:, i] = np.roll(self.sim.agent.familiarity, len(self.sim.agent.pref_angles) // 2)
            #     self.fam_all.set_array(fam_all)
            #     if self.fam_line is not None:
            #         sep = self.sim.route.shape[0]
            #         if self.sim.frame > sep:
            #             self.fam_line.set_data(np.arange(sep, self.sim.frame), np.nanargmin(
            #                 np.roll(fam_all[:, sep:self.sim.frame], len(self.sim.agent.pref_angles) // 2, axis=0),
            #                 axis=0))

        else:
            self.pn.set_array(self.sim.mem.r_cs[0].T.flatten())
            if self._show_weights:
                self.kc.set_array(self.sim.mem.w_k2m.T.flatten())
            else:
                self.kc.set_array(self.sim.mem.r_kc[0].T.flatten())
            self.fam.set_array(self.sim.agent.familiarity.T.flatten())

        return time

    @property
    def sim(self):
        """
        The simulation that runs on the background

        Returns
        -------
        VisualNavigationSimulation
        """
        return self._sim

    @property
    def omm(self):
        """
        The collection of ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[0]

    @property
    def line_c(self):
        """
        The line representing the ongoing path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[1]

    @property
    def line_b(self):
        """
        The line representing the finished path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[2]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[3]

    @property
    def cal(self):
        """
        The positions the figure used for calibration in.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[4]

    @property
    def poi(self):
        """
        The positions in the figure where the agent was brought back to the route.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[5]

    @property
    def pn(self):
        """
        The history of the PN response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[6]

    @property
    def kc(self):
        """
        The history of the KC response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[7]

    @property
    def fam(self):
        """
        The history of familiarity in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[8]

    @property
    def dist(self):
        """
        The history of the distance from the goal (nest) in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        if len(self._lines) > 9:
            return self._lines[9]
        else:
            return None

    @property
    def capacity(self):
        """
        The history of the memory capacity in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        if len(self._lines) > 10:
            return self._lines[10]
        else:
            return None

    @property
    def fam_all(self):
        """
        The history of familiarity in the figure for all the scanning directions.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 11:
            return self._lines[11]
        else:
            return None

    @property
    def fam_line(self):
        """
        The history of the most familiar direction in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        if len(self._lines) > 12:
            return self._lines[12]
        else:
            return None


class VisualFamiliarityAnimation(AnimationBase):

    def __init__(self, sim, cmap="Greens_r", *args, **kwargs):
        """
        Animation for the visual navigation simulation. It visualised the current view of the agent, its current and
        previous positions on the map along with the vegetation, and the statistics according to the parameters.

        Parameters
        ----------
        sim: VisualFamiliarityTestSimulation | VisualFamiliarityGridExplorationSimulation
            the visual navigation simulation instance
        cmap: str, optional
            the colour map to be used for the responses from the ommatidia. Default is 'Greens_r'
        """
        super().__init__(sim, *args, **kwargs)

        ax_dict = self.fig.subplot_mosaic(
            """
            AABB
            AABB
            CEBB
            DEBB
            """
        )

        omm = create_eye_axis(sim.eye, cmap=cmap, ax=ax_dict["A"])
        line_c, line_b, pos, self._marker, cal, poi = create_map_axis(
            world=sim.world, ax=ax_dict["B"], nest=sim.route[-1, :2], feeder=sim.route[0, :2])

        self._lines.extend([omm, line_c, line_b, pos, cal, poi])

        pn = create_pn_history(sim.agent, self.nb_frames, sep=sim.route.shape[0],
                               cmap="Greys", ax=ax_dict["C"])
        kc = create_kc_history(sim.agent, self.nb_frames, sep=sim.route.shape[0],
                               cmap="Greys", ax=ax_dict["D"])
        fam_all, fam_qui = create_familiarity_map(sim.nb_cols, sim.nb_rows, cmap="RdPu", ax=ax_dict["E"])

        self._lines.extend([pn, kc, fam_all, fam_qui])

        plt.tight_layout()

    def _animate(self, i):
        """
        Runs the current iteration of the simulation and updates the data for the figure.

        Parameters
        ----------
        i: int
            the current iteration
        """
        if i == 0:
            xyzs = self.sim.reset()
            if xyzs is not None:
                self.cal.set_offsets(np.array(xyzs)[:, [1, 0]])
        elif i == self.sim.route.shape[0]:
            self.line_b.set_data(np.array(self.sim.stats["position"])[..., 1],
                                 np.array(self.sim.stats["position"])[..., 0])

        time = self.sim.step(i)

        r = self.sim.stats["ommatidia"][-1].mean(axis=1)
        x, y = np.array(self.sim.stats["position"])[..., :2].T

        # draw the ommatidia responses
        self.omm.set_array(r.T.flatten())

        # draw current path
        if i < self.sim.route.shape[0]:
            self.line_c.set_data(y, x)

        # draw current position and orientation
        self.pos.set_offsets(np.array([y[-1], x[-1]]))
        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        # draw the PN activity
        pn = self.pn.get_array()
        pn[:, i] = self.sim.mem.r_inp[0].T.flatten()
        self.pn.set_array(pn)

        # draw the KC activity
        kc = self.kc.get_array()
        kc[:, i] = self.sim.mem.r_hid[0].T.flatten()
        self.kc.set_array(kc)

        # draw familiarity map
        if self.fam_all is not None:

            fam_all = self.fam_all.get_array()
            fam_all[:] = np.max(1 - (1 - self.sim.familiarity_map) / (1 - self.sim.familiarity_map).max(), axis=2)
            self.fam_all.set_array(fam_all)
            # z = ring2complex(self.sim.familiarity_map, axis=2)
            # u, v = z.real, z.imag
            # self.fam_qui.set_UVC(u, v)

        return time

    @property
    def sim(self):
        """
        The simulation that runs on the background

        Returns
        -------
        VisualFamiliarityTestSimulation
        """
        return self._sim

    @property
    def omm(self):
        """
        The collection of ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[0]

    @property
    def line_c(self):
        """
        The line representing the ongoing path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[1]

    @property
    def line_b(self):
        """
        The line representing the finished path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[2]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[3]

    @property
    def cal(self):
        """
        The positions the figure used for calibration in.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[4]

    @property
    def poi(self):
        """
        The positions in the figure where the agent was brought back to the route.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[5]

    @property
    def pn(self):
        """
        The history of the PN response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[6]

    @property
    def kc(self):
        """
        The history of the KC response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[7]

    @property
    def fam_all(self):
        """
        The map of familiarity in the figure for all the visited positions.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 8:
            return self._lines[8]
        else:
            return None

    @property
    def fam_qui(self):
        """
        The direction of the maximum familiarity in the figure for all the visited directions.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 9:
            return self._lines[9]
        else:
            return None


class PathIntegrationAnimation(AnimationBase):

    def __init__(self, sim, show_history=True, cmap="coolwarm", *args, **kwargs):
        """
        Animation for the path integration simulation. Shows the POL neurons responses in the Dorsal Rim Area, the
        position and history of positions of the agent on the map (with vegetation if provided) and the responses of
        the CX neurons (and their history if requested).

        Parameters
        ----------
        sim: PathIntegrationSimulation, TwoSourcePathIntegrationSimulation
            the path integration simulation isnstance
        show_history: bool, optional
            if True, it shows the history instead of just the current responses. Default is True
        cmap: str, optional
            the colour map for the responses of the POL neurons. Default is 'coolwarm'
        """
        kwargs.setdefault('fps', 100)
        super().__init__(sim, *args, **kwargs)

        if show_history:
            mosaic = """
                ACCCBBBB
                DDDDBBBB
                EEEEBBBB
                FFFFBBBB
                GGGGBBBB
                """
            if isinstance(sim, TwoSourcePathIntegrationSimulation) or isinstance(sim.agent, RouteFollowingAgent):
                mosaic += """HHHHBBBB
                """
            ax_dict = self.fig.subplot_mosaic(mosaic)
        else:
            ax_dict = self.fig.subplot_mosaic(
                """
                AB
                AB
                AB
                """
            )

        if isinstance(sim, TwoSourcePathIntegrationSimulation):
            nest = sim.central_point[:2]
            feeders = [sim.feeder_a[:2], sim.feeder_b[:2]]
            route = sim.route_a
            lg.logger.debug(feeders)
        elif isinstance(sim, PathIntegrationSimulation):
            nest = sim.central_point[:2]
            feeders = [sim.distant_point[:2]]
            route = sim.route
        else:
            nest = None
            feeders = None
            route = None

        line_c, line_b, pos, self._marker = create_map_axis(world=sim.world, ax=ax_dict["B"],
                                                            nest=nest, feeders=feeders)[:4]
        vec, mbon = None, None
        if show_history:
            omm = create_dra_axis(sim.compass_sensor, cmap=cmap, ax=ax_dict["A"])
            tb1 = create_tb1_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["C"])
            cl1 = create_cl1_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["D"])
            cpu1 = create_cpu1_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["E"])
            cpu4 = create_cpu4_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["F"])
            cpu4mem = create_cpu4_mem_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap,
                                              ax=ax_dict["G"])
            if isinstance(sim, TwoSourcePathIntegrationSimulation):
                vec = create_vec_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["H"])
            if isinstance(sim.agent, RouteFollowingAgent):
                mbon = create_mbon_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["H"])
        else:
            omm, tb1, cl1, cpu1, cpu4, cpu4mem = create_bcx_axis(sim.agent, cmap=cmap, ax=ax_dict["A"])

        plt.tight_layout()

        self._lines.extend([omm, tb1, cl1, cpu1, cpu4, cpu4mem, line_c, line_b, pos, vec, mbon])

        omm.set_array(sim.r_pol)
        self._show_history = show_history

    def _animate(self, i):
        """
        Runs the current iteration of the simulation and updates the data from the figure.

        Parameters
        ----------
        i: int
            the current iteration number
        """

        if isinstance(self.sim, PathIntegrationSimulation):
            route_size = self.sim.route.shape[0]
        elif isinstance(self.sim, TwoSourcePathIntegrationSimulation):
            route_size = self.sim.route_a.shape[0]
        else:
            route_size = -1
            lg.logger.debug("None INSTANCE!")
        if i == 0:
            self.line_b.set_data([], [])
            # self.sim.reset(nb_samples_calibrate=10)
            self.sim.reset()
        elif "xyz_out" in self.sim.stats:
            self.line_b.set_data(np.array(self.sim.stats["xyz_out"])[..., 1],
                                 np.array(self.sim.stats["xyz_out"])[..., 0])
            # self.sim.init_inbound()

        time = self.sim.step(i)

        self.omm.set_array(np.array(self.sim.stats["POL"][-1]))

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
            if self.vec is not None:
                vec = np.zeros((self.sim.r_vec.shape[0], self.nb_frames), dtype=float)
                vec[:, :i+1] = np.array(self.sim.stats["vec"]).T
                self.vec.set_array(vec)
            if self.mbon is not None:
                mbon = np.zeros((self.sim.r_mbon.shape[-1], self.nb_frames), dtype=float)
                mbon[:, :i+1] = np.array(self.sim.stats["MBON"]).T
                self.mbon.set_array(mbon)
        else:
            self.tb1.set_array(self.sim.r_tb1)
            self.cl1.set_array(self.sim.r_cl1)
            self.cpu1.set_array(self.sim.r_cpu1)
            self.cpu4.set_array(self.sim.r_cpu4)
            self.cpu4mem.set_array(self.sim.cpu4_mem)

        x, y = np.array(self.sim.stats["xyz"])[..., :2].T
        self.line_c.set_data(y, x)
        self.pos.set_offsets(np.array([y[-1], x[-1]]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        return time

    @property
    def sim(self):
        """
        The path integration simulation instance.

        Returns
        -------
        PathIntegrationSimulation
        """
        return self._sim

    @property
    def omm(self):
        """
        The collection of the DRA ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[0]

    @property
    def tb1(self):
        """
        The history of the TB1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[1]

    @property
    def cl1(self):
        """
        The history of the CL1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[2]

    @property
    def cpu1(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[3]

    @property
    def cpu4(self):
        """
        The history of the CPU4 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[4]

    @property
    def cpu4mem(self):
        """
        The history of the CPU4 memory in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[5]

    @property
    def line_c(self):
        """
        The line representing the ongoing path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[6]

    @property
    def line_b(self):
        """
        The line representing the finished path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[7]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[8]

    @property
    def vec(self):
        """
        The history of the Vector neurons in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 8:
            return self._lines[9]
        else:
            return None

    @property
    def mbon(self):
        """
        The history of the Vector neurons in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 9:
            return self._lines[10]
        else:
            return None


class NavigationAnimation(AnimationBase):

    def __init__(self, sim, cmap="coolwarm", *args, **kwargs):
        """
        Animation for the path integration simulation. Shows the POL neurons responses in the Dorsal Rim Area, the
        position and history of positions of the agent on the map (with vegetation if provided) and the responses of
        the CX neurons (and their history if requested).

        Parameters
        ----------
        sim: NavigationSimulation
            the path integration simulation isnstance
        cmap: str, optional
            the colour map for the responses of the POL neurons. Default is 'coolwarm'
        """
        kwargs.setdefault('fps', 100)
        # kwargs.setdefault('width', 15)
        kwargs.setdefault('height', 6)
        super().__init__(sim, *args, **kwargs)

        ax_dict = self.fig.subplot_mosaic(
            """
            ACCCBBBBBB
            DDDDBBBBBB
            EEEEBBBBBB
            FFFFBBBBBB
            GGGGBBBBBB
            HHHHBBBBBB
            JJKKBBBBBB
            """
        )
        nest = sim.route[0, :2]
        feeders = [route[-1, :2] for route in sim.routes]
        route = sim.route
        odour_spread = []
        for odour in sim.odours:
            odour_spread.append(odour.spread)

        all_lines = create_map_axis(world=sim.world, ax=ax_dict["B"],
                                    nest=nest, feeders=feeders, odour_spread=odour_spread)
        line_c, line_b, pos, self._marker, cal = all_lines[:5]
        feeders_text = all_lines[-1]

        omm = create_dra_axis(sim.agent.pol_sensor, cmap=cmap, ax=ax_dict["A"])
        tb1 = create_tb1_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["C"])
        cl1 = create_cl1_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["D"])
        cpu1 = create_cpu1_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["E"])
        cpu4 = create_cpu4_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["F"])
        cpu4mem = create_cpu4_mem_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["G"])
        pn = create_pn_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["H"])
        mbon = create_mbon_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["J"])
        dan = create_dan_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["K"])

        plt.tight_layout()

        self._lines.extend([omm, tb1, cl1, cpu1, cpu4, cpu4mem, pn, mbon, dan, line_c, line_b, pos, cal] + feeders_text)

        self._nb_lines = 0
        omm.set_array(sim.r_pol)

    def _animate(self, i: int):
        """
        Runs the current iteration of the simulation and updates the data from the figure.

        Parameters
        ----------
        i: int
            the current iteration number
        """

        nb_lines = 0
        for suf in ["in", "out"]:
            j = 0
            while f"xyz_{suf}_{j}" in self.sim.stats:
                j += 1
            nb_lines += j

        if i == 0:
            self.line_b.set_data([], [])
            self.sim.reset()
            self._nb_lines = 0

            xyzs = self.sim.reset()

            if xyzs is not None:
                self.cal.set_offsets(np.array(xyzs)[:, [1, 0]])
        elif nb_lines > self._nb_lines:
            xs, ys = [], []
            for suf in ["in", "out"]:
                j = 0
                while f"xyz_{suf}_{j}" in self.sim.stats:
                    xs.append(np.r_[np.array(self.sim.stats[f"xyz_{suf}_{j}"])[:, 1], np.nan])
                    ys.append(np.r_[np.array(self.sim.stats[f"xyz_{suf}_{j}"])[:, 0], np.nan])
                    j += 1

            if len(xs) > 0 and len(ys) > 0:
                xs = np.hstack(xs)
                ys = np.hstack(ys)

            self.line_b.set_data(xs, ys)
            self._nb_lines = nb_lines

        time = self.sim.step(i)

        self.omm.set_array(np.array(self.sim.stats["POL"][-1]))

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
        pn = np.zeros((self.sim.r_pn.shape[0], self.nb_frames), dtype=float)
        pn[:, :i+1] = np.array(self.sim.stats["PN"]).T
        self.pn.set_array(pn)
        mbon = np.zeros((self.sim.r_mbon.shape[0], self.nb_frames), dtype=float)
        mbon[:, :i+1] = np.array(self.sim.stats["MBON"]).T
        self.mbon.set_array(mbon)
        dan = np.zeros((self.sim.r_dan.shape[0], self.nb_frames), dtype=float)
        dan[:, :i+1] = np.array(self.sim.stats["DAN"]).T
        self.dan.set_array(dan)

        x, y = np.array(self.sim.stats["xyz"])[..., :2].T
        self.line_c.set_data(y, x)
        self.pos.set_offsets(np.array([y[-1], x[-1]]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        for i, feeder_text in enumerate(self.feeders_text):
            feeder_text.set_text(f"Crumbs: {self.sim.food_supply[i]}")

        return time

    @property
    def sim(self):
        """
        The path integration simulation instance.

        Returns
        -------
        NavigationSimulation
        """
        return self._sim

    @property
    def omm(self):
        """
        The collection of the DRA ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[0]

    @property
    def tb1(self):
        """
        The history of the TB1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[1]

    @property
    def cl1(self):
        """
        The history of the CL1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[2]

    @property
    def cpu1(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[3]

    @property
    def cpu4(self):
        """
        The history of the CPU4 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[4]

    @property
    def cpu4mem(self):
        """
        The history of the CPU4 memory in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[5]

    @property
    def pn(self):
        """
        The history of the PN response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[6]

    @property
    def mbon(self):
        """
        The history of the MBON response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[7]

    @property
    def dan(self):
        """
        The history of the DAN response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[8]

    @property
    def line_c(self):
        """
        The line representing the ongoing path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[9]

    @property
    def line_b(self):
        """
        The line representing the finished path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[10]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[11]

    @property
    def cal(self):
        return self._lines[12]

    @property
    def feeders_text(self):
        """
        The text next to each feeder in the map.

        Returns
        -------
        list[matplotlib.text.Text]
        """
        return self._lines[13:]
