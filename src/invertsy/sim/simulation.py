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

from collections import namedtuple
from abc import ABC

from ._helpers import col2x, row2y, yaw2ori, x2col, y2row, ori2yaw

from invertsy.__helpers import __data__, RNG
from invertsy.env import UniformSky, Sky, Seville2009, WorldBase, StaticOdour
from invertsy.agent import VisualNavigationAgent, VectorMemoryAgent, NavigatingAgent, RouteFollowingAgent, Agent
from invertsy.agent.agent import VisualProcessingAgent

from invertpy.sense import CompoundEye
from invertpy.sense.polarisation import PolarisationSensor
from invertpy.brain.preprocessing import Preprocessing
from invertpy.brain.compass import PolarisationCompass
from invertpy.brain.memory import MemoryComponent
from invertpy.brain.centralcomplex import CentralComplexBase
from invertpy.brain.preprocessing import MentalRotation

from scipy.spatial.transform import Rotation as R
from scipy.special import expit

import numpy as np

from time import time
from copy import copy

import os

__stat_dir__ = os.path.abspath(os.path.join(__data__, "animation", "stats"))
__outb_dir__ = os.path.abspath(os.path.join(__data__, "animation", "outbounds"))
if not os.path.isdir(__stat_dir__):
    os.makedirs(__stat_dir__)
if not os.path.isdir(__outb_dir__):
    os.makedirs(__outb_dir__)


class Simulation(object):
    def __init__(self, agent=None, noise=0., rng=RNG, name="simulation"):
        """

        Parameters
        ----------
        agent: Agent, optional
            the agent. Default is a `Agent(speed=.01)`
        noise : float
            the noise amplitude in the system
        rng : np.random.RandomState
            the random number generator
        name: str, optional
            a unique name for the simulation. Default is 'simulation'
        """
        if agent is None:
            agent = Agent(speed=.02)
        if name is None:
            name = "simulation"
        self.__agent = agent
        self.__noise = noise
        self.__rng = rng
        self.__name = name

        self.__callback = None

    def __call__(self, linear_velocity=None, angular_velocity=None, **kwargs):
        if linear_velocity is not None:
            self.agent.move_towards(linear_velocity)
        if angular_velocity is not None:
            self.agent.rotate(angular_velocity)

        return self.callback(self.agent, **kwargs)

    @property
    def agent(self):
        return self.__agent

    @property
    def callback(self):
        if self.__callback is None:
            return lambda x, **kwargs: None
        else:
            return self.__callback

    @callback.setter
    def callback(self, value):
        self.__callback = value

    @property
    def noise(self):
        return self.__noise

    @property
    def rng(self):
        return self.__rng

    @property
    def name(self):
        return self.__name

    def __repr__(self):
        return f"Simulation(agent_type={self.agent.__class__.__name__}, noise={self.noise:.2f}, name={self.name})"


class GradientVectorSimulation(Simulation):

    def __init__(self, gradient=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if gradient is None:
            gradient = Gradient([[0, 0]], sigma=1, grad_type='gaussian')

        #     s.g, u.g, m.d,  o.g, g.p, n.u,   exp, bs
        # 01: 0.3, 0.1, 0.03, 0.2, 0.0, False, 1.0, 0.1 --- small S
        # 02: 0.3, 0.1, 0.05, 0.2, 0.0, False, 1.0, 0.1 --- small S
        # 03: 0.3, 0.1, 0.07, 0.2, 0.0, False, 1.0, 0.1 --- I-shape
        # 04: 0.3, 0.1, 0.10, 0.2, 0.0, False, 1.0, 0.1 --- small S
        # 05: 0.3, 0.1, 0.10, 0.2, 0.0, False, 1.0, 0.1 --- I-shape
        # 06: 0.3, 0.1, 0.10, 0.2, 0.0, False, 1.0, 0.1 --- large S
        self.__steering_gain = 0.3
        self.__update_gain = 0.1
        self.__memory_decay = 0.1
        self.__oscillation_gain = .2
        self.__g_power = 0.
        self.__normalise_update = False
        gradient.exp = 1.
        gradient.baseline = 0.1

        self.__gradient = gradient
        self.__hist = {
            'g': [],
            'x': [], 'y': [], 'yaw': [], 'phi': [],
            'epg': [],
            'pfn_d': [], 'pfn_v': [], 'pfn_a': [], 'pfn_p': [],
            'hdb': [], 'hdc': [],
            'dfb': [], 'dfc': [],
            'pfl2': [], 'pfl3': [],
        }
        self.l_epg = lambda yaw: np.exp(1j * yaw)
        self.r_epg = lambda yaw: np.exp(1j * yaw)
        self.l_pfn = lambda l_epg, r_lno: (1 - r_lno) * np.exp(+1j * np.pi / 4) * l_epg
        self.r_pfn = lambda r_epg, l_lno: (1 - l_lno) * np.exp(-1j * np.pi / 4) * r_epg
        self.l_hdb = lambda l_pfn_d, l_pfn_v: l_pfn_d + np.exp(-4j * np.pi / 4) * l_pfn_v
        self.r_hdb = lambda r_pfn_d, r_pfn_v: r_pfn_d + np.exp(+4j * np.pi / 4) * r_pfn_v
        self.l_hdc = lambda l_pfn_a, l_pfn_p: l_pfn_a + np.exp(-4j * np.pi / 4) * l_pfn_p
        self.r_hdc = lambda r_pfn_a, r_pfn_p: r_pfn_a + np.exp(+4j * np.pi / 4) * r_pfn_p
        self.l_pfl2 = lambda l_dfb, l_epg: np.exp(-4j * np.pi / 4) * l_epg + l_dfb
        self.r_pfl2 = lambda r_dfb, r_epg: np.exp(+4j * np.pi / 4) * r_epg + r_dfb
        # self.l_pfl3 = lambda l_dfc, l_epg: np.exp(+0j * np.pi / 4) * l_dfc + np.exp(+3j * np.pi / 4) * l_epg
        # self.r_pfl3 = lambda r_dfc, r_epg: np.exp(-0j * np.pi / 4) * r_dfc + np.exp(-3j * np.pi / 4) * r_epg
        self.l_pfl3 = lambda l_dfc, l_epg: np.exp(+0j * np.pi / 4) * l_dfc + np.exp(-3j * np.pi / 4) * l_epg
        self.r_pfl3 = lambda r_dfc, r_epg: np.exp(-0j * np.pi / 4) * r_dfc + np.exp(+3j * np.pi / 4) * r_epg

        self.r_nod = 0.
        self.l_nod = 0.

        self.osc_b = 0.  # side-walk
        self.osc_c = 0.  # steering

    def __call__(self, *args, **kwargs):
        self.sense()
        lv, av, grad = self.act()

        return super(GradientVectorSimulation, self).__call__(
            linear_velocity=lv, angular_velocity=av, grad=grad, epg=self.__hist["epg"][-1],
            pfn_d=self.__hist["pfn_d"][-1], pfn_v=self.__hist["pfn_v"][-1],
            hdb=self.__hist["hdb"][-1], dfb=self.__hist["dfb"][-1], pfl2=self.__hist["pfl2"][-1],
            pfn_a=self.__hist["pfn_a"][-1], hdc=self.__hist["hdc"][-1],
            dfc=self.__hist["dfc"][-1], pfl3=self.__hist["pfl3"][-1], phi=self.__hist["phi"][-1])

    def act(self):

        front = 1
        side = self.osc_b

        yaw = self.__hist['yaw'][-1]

        if len(self.__hist['yaw']) > 1:
            d_x = self.__hist['x'][-1] - self.__hist['x'][-2]
            d_y = self.__hist['y'][-1] - self.__hist['y'][-2]
            d_yaw = self.__hist['yaw'][-1] - self.__hist['yaw'][-2]
            phi = (self.__hist['yaw'][-1] + np.arctan2(d_x, d_y) - d_yaw + np.pi) % (2 * np.pi) - np.pi
        else:
            phi = 0.

        pfl2_v = (self.__hist["pfl2"][-1][1] + self.__hist["pfl2"][-1][0])
        pfl2_m = np.abs(pfl2_v)
        pfl2_a = np.angle(pfl2_v)  # - phi
        l_pfl3, r_pfl3 = self.__hist["pfl3"][-1]
        pfl3_v = l_pfl3 + r_pfl3
        pfl3_m = np.abs(pfl3_v)
        pfl3_a = self.__steering_gain * np.angle(pfl3_v)
        # pfl3_a = (np.abs(self.__hist["pfl3"][-1][0]) - np.abs(self.__hist["pfl3"][-1][1])) * np.pi / 4
        print(f"yaw = {np.rad2deg(yaw):.2f}", end=";  ")
        # print(f"PFL2_ang = {np.rad2deg(pfl2_a):.2f}", end=";  ")
        print(f"PFL3_mag = {pfl3_m:.2f}, PFL3_ang = {np.rad2deg(pfl3_a):.2f}")

        osc_w = self.__oscillation_gain
        rand = 0.0 * (np.random.rand(3) * 2 - 1)
        lv = [0, front, 0]
        # lv = [osc_w * side, front, 0]
        # lv = [osc_w * side - np.sin(pfl2_a) + rand[0], front + rand[1], 0]
        # lv = [osc_w * side - 1 * np.sin(pfl2_a) + rand[0], osc_w * front + np.cos(pfl2_a) + rand[1], 0]
        # av = R.from_euler("Z", osc_w * self.osc_c)
        av = R.from_euler("Z", osc_w * self.osc_c +
                          np.clip(pfl3_m * pfl3_a, -np.pi/4, np.pi/4) +
                          rand[2] * np.pi / 2)

        # lv = [side * (1 - w * np.clip(pfl2_m, 0, 1)), front * (1 + w * np.clip(pfl2_m, 0, 1)), 0]
        # lv = [side, front * (1 - w * np.clip(pfl2_m, 0, 1)), 0]
        # av = R.from_euler("Z", np.clip(1 - pfl3_m, 0.2, 1) * self.osc_v + np.clip(pfl3_m, 0, .8) * pfl3_a)
        grad = self.__hist['g'][-1]
        # lv = av.apply(lv)

        return lv, av, grad

    def sense(self):
        x, y = self.agent.x, self.agent.y
        yaw = self.agent.yaw
        g = self.gradient(x, y, yaw)

        self.__hist['g'].append(g)
        self.__hist['x'].append(x)
        self.__hist['y'].append(y)
        self.__hist['yaw'].append(yaw)

        if len(self.__hist['g']) > 1:
            d_g = self.__hist['g'][-1] - self.__hist['g'][-2]
            d_x = self.__hist['x'][-1] - self.__hist['x'][-2]
            d_y = self.__hist['y'][-1] - self.__hist['y'][-2]
            d_yaw = self.__hist['yaw'][-1] - self.__hist['yaw'][-2]
        else:
            d_g = 0.
            d_x = 0.
            d_y = 0.
            d_yaw = 0.

        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi

        phi = np.angle(np.exp(-1j * np.pi / 2) * (d_x + d_y * 1j))
        self.__hist["phi"].append(phi)

        if len(self.__hist['g']) > 1:
            d_phi = phi - self.__hist['yaw'][-2]
            # d_phi = self.__hist['phi'][-1] - self.__hist['phi'][-1] - d_yaw
        else:
            d_phi = 0.
            # d_phi = -d_yaw

        # calculate the relative direction of holonomic motion (excluding the heading)
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

        # calculate the relative magnitude (speed) of motion
        # rho = np.sqrt(np.square(d_x) + np.square(d_y)) * 20
        rho = 1
        # print(f"d_phi = {np.rad2deg(phi):.2f}, d_yaw={np.rad2deg(d_yaw):.2f}", end="; ")

        r_lno1 = rho * np.clip(0.5 + 0.5 * np.sin(d_phi) * (0.5 + 0.5 * np.cos(d_phi)), 0.1, 0.9)
        l_lno1 = rho * np.clip(0.5 - 0.5 * np.sin(d_phi) * (0.5 + 0.5 * np.cos(d_phi)), 0.1, 0.9)
        r_lno2 = rho * np.clip(0.5 + 0.5 * np.sin(d_phi) * (0.5 - 0.5 * np.cos(d_phi)), 0.1, 0.9)
        l_lno2 = rho * np.clip(0.5 - 0.5 * np.sin(d_phi) * (0.5 - 0.5 * np.cos(d_phi)), 0.1, 0.9)
        # print(f"_lLNO1 = {l_lno1:.2f}, _rLNO1 = {r_lno1:.2f}, _lLNO2 = {l_lno2:.2f}, _rLNO2  = {r_lno2:.2f}", end="; ")

        # r_lno_a = float(d_yaw < 0) * np.maximum(np.cos(d_yaw + np.pi/2), 0)
        # l_lno_a = float(d_yaw > 0) * np.maximum(np.cos(d_yaw - np.pi/2), 0)
        r_lno_a = 0 * np.cos(d_yaw + np.pi / 4) * np.clip(np.cos(d_yaw + np.pi / 2), 0., 0.9)
        l_lno_a = 0 * np.cos(d_yaw - np.pi / 4) * np.clip(np.cos(d_yaw - np.pi / 2), 0., 0.9)
        r_lno_p = 1 - r_lno_a
        l_lno_p = 1 - l_lno_a
        # r_lno_a = np.clip(-0.5 * (np.cos(d_yaw + 3 * np.pi / 4) + 1), 0.01, 0.9)
        # l_lno_a = np.clip(-0.5 * (np.cos(d_yaw - 3 * np.pi / 4) + 1), 0.01, 0.9)
        # l_lno_p = np.clip(+0.5 * (np.cos(d_yaw + np.pi / 4) + 1), 0.01, 0.9)
        # r_lno_p = np.clip(+0.5 * (np.cos(d_yaw - np.pi / 4) + 1), 0.01, 0.9)

        # r_lno_a = np.clip(-np.sin(d_yaw - np.pi/4), 0.01, 0.9)
        # l_lno_a = np.clip(+np.sin(d_yaw + np.pi/4), 0.01, 0.9)
        print(f"g = {np.power(g, self.__g_power):.4f}, _lLNO_a = {l_lno_a:.2f}, _rLNO_a = {r_lno_a:.2f}", end=", ")
        print(f"_lLNO_p = {l_lno_p:.2f}, _rLNO_p = {r_lno_p:.2f}, d_yaw = {np.rad2deg(d_yaw):.2f}", end="; ")
        print(f"d_phi = {np.rad2deg(d_phi):.2f}", end="; ")

        l_epg_b = self.l_epg(yaw)
        r_epg_b = self.r_epg(yaw)
        self.__hist['epg'].append(np.r_[l_epg_b, r_epg_b])

        l_pfn_d = self.l_pfn(l_epg_b, r_lno2)
        r_pfn_d = self.r_pfn(r_epg_b, l_lno2)
        self.__hist['pfn_d'].append(np.r_[l_pfn_d, r_pfn_d])

        l_pfn_v = self.l_pfn(l_epg_b, r_lno1)
        r_pfn_v = self.r_pfn(r_epg_b, l_lno1)
        self.__hist['pfn_v'].append(np.r_[l_pfn_v, r_pfn_v])

        l_epg_c = self.l_epg(yaw)
        r_epg_c = self.r_epg(yaw)
        l_pfn_a = self.l_pfn(l_epg_c, r_lno_a)
        r_pfn_a = self.r_pfn(r_epg_c, l_lno_a)
        l_pfn_p = self.l_pfn(l_epg_c, r_lno_p)
        r_pfn_p = self.r_pfn(r_epg_c, l_lno_p)
        self.__hist['pfn_a'].append(np.r_[l_pfn_a, r_pfn_a])

        l_hdb = np.sign(d_g) * self.l_hdb(l_pfn_d, l_pfn_v)
        r_hdb = np.sign(d_g) * self.r_hdb(r_pfn_d, r_pfn_d)
        self.__hist['hdb'].append(np.r_[l_hdb, r_hdb])

        # l_hdc = float(d_yaw <= 0) * np.sign(d_g) * self.l_hdc(l_pfn_a, l_pfn_p)
        # r_hdc = float(d_yaw >= 0) * np.sign(d_g) * self.r_hdc(r_pfn_a, r_pfn_p)
        l_hdc = self.l_hdc(l_pfn_a, l_pfn_p)
        r_hdc = self.r_hdc(r_pfn_a, r_pfn_p)
        self.__hist['hdc'].append(np.r_[l_hdc, r_hdc])

        delta_phi_b = np.sign(d_g) * np.r_[l_hdb, r_hdb]
        # delta_phi_c = np.sign(d_g) * np.r_[l_hdc, r_hdc]
        delta_phi_c = self.__update_gain * np.sign(d_g) * np.r_[l_hdc, r_hdc]
        if len(self.__hist['dfb']) > 0:
            delta_phi_b = 2 * self.__hist['dfb'][-1] + delta_phi_b
            delta_phi_c = np.power(g, self.__g_power) * (1 - self.__memory_decay) * self.__hist['dfc'][-1] + delta_phi_c
            # delta_phi_c = 0.98 * self.__hist['dfc'][-1] + delta_phi_c
            if self.__normalise_update:
                delta_phi_b /= (np.abs(delta_phi_b) + np.finfo(float).eps)
                delta_phi_c /= (np.abs(delta_phi_c) + np.finfo(float).eps)
        l_dfb = delta_phi_b[0]
        r_dfb = delta_phi_b[1]
        l_dfc = delta_phi_c[0]
        r_dfc = delta_phi_c[1]
        self.__hist['dfb'].append(delta_phi_b)  # side-walk
        self.__hist['dfc'].append(delta_phi_c)  # steering

        l_pfl2 = np.exp(+0j * np.pi / 4) * l_dfb - np.exp(+2j * np.pi / 4) * l_epg_b
        r_pfl2 = np.exp(-0j * np.pi / 4) * r_dfb - np.exp(-2j * np.pi / 4) * r_epg_b
        self.__hist['pfl2'].append(np.r_[l_pfl2, r_pfl2])
        # l_pfl3 = l_dfc - np.exp(+1j * np.pi / 4) * l_epg_c
        # r_pfl3 = r_dfc - np.exp(-1j * np.pi / 4) * r_epg_c
        l_pfl3 = self.l_pfl3(0.5 * (l_dfc + r_dfc), 0.9 * l_epg_c * np.abs(0.5 * (l_dfc + r_dfc)))
        r_pfl3 = self.r_pfl3(0.5 * (r_dfc + l_dfc), 0.9 * r_epg_c * np.abs(0.5 * (r_dfc + l_dfc)))
        self.__hist['pfl3'].append(np.r_[l_pfl3, r_pfl3])

        # default oscillation angle
        angle = (len(self.__hist['g']) + 4.5) * np.pi / 10

        self.osc_b = np.sin(angle)  # side-walk oscillation
        self.osc_c = (np.abs((angle + np.pi / 2) % (2 * np.pi) - np.pi) - np.pi / 2) / 4  # steering oscillation

    @property
    def gradient(self):
        return self.__gradient

    @staticmethod
    def get_angle(population):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        n = population.shape[0] // 2
        lv = np.sum(population[:n] * np.exp(-1j * angles))
        rv = np.sum(population[n:] * np.exp(-1j * angles))

        return np.angle(lv + rv)


class GradientSimulation(Simulation):

    def __init__(self, gradient=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if gradient is None:
            gradient = Gradient([[0, 0]], sigma=1, grad_type='gaussian')

        self.__gradient = gradient
        self.__hist = {
            'g': [],
            'x': [], 'y': [], 'yaw': [], 'phi': [],
            'epg': [],
            'pfn_d': [], 'pfn_v': [], 'pfn_a': [], 'pfn_p': [],
            'hdb': [], 'hdc': [],
            'dfb': [], 'dfc': [],
            'pfl2': [], 'pfl3': [],
        }
        self.l_epg = lambda yaw: np.cos(yaw + np.linspace(0, 2 * np.pi, 8, endpoint=False))
        self.r_epg = lambda yaw: np.cos(yaw + np.linspace(0, 2 * np.pi, 8, endpoint=False))
        self.l_pfn = lambda l_epg, r_lno: (1 - r_lno) * np.roll(l_epg, shift=-1)  # this is working
        self.r_pfn = lambda r_epg, l_lno: (1 - l_lno) * np.roll(r_epg, shift=+1)  # this is working
        self.l_hdb = lambda l_pfn_d, l_pfn_v: l_pfn_d + np.roll(l_pfn_v, shift=-4)
        self.r_hdb = lambda r_pfn_d, r_pfn_v: r_pfn_d + np.roll(r_pfn_v, shift=+4)
        self.l_hdc = lambda l_pfn_a, l_pfn_p: l_pfn_a + np.roll(l_pfn_p, shift=-4)
        self.r_hdc = lambda r_pfn_a, r_pfn_p: r_pfn_a + np.roll(r_pfn_p, shift=+4)
        self.dfb = lambda hdb, d_mbon: d_mbon * hdb
        self.dfc = lambda hdc, d_mbon: d_mbon * hdc
        self.l_pfl2 = lambda l_epg, l_dfb: np.roll(l_dfb + l_epg * np.max(l_dfb), shift=-1)
        self.r_pfl2 = lambda r_epg, r_dfb: np.roll(r_dfb + r_epg * np.max(r_dfb), shift=+1)
        self.l_pfl3 = lambda l_epg, l_dfc: np.roll(l_dfc, shift=-1) + np.roll(l_epg, shift=-1)  # this is working
        self.r_pfl3 = lambda r_epg, r_dfc: np.roll(r_dfc, shift=+1) + np.roll(r_epg, shift=+1)  # this is working

        self.r_nod = 0.
        self.l_nod = 0.

        self.osc_b = 0.  # side-walk
        self.osc_c = 0.  # steering

    def __call__(self, *args, **kwargs):
        self.sense()
        lv, av, grad = self.act()

        super(GradientSimulation, self).__call__(linear_velocity=lv, angular_velocity=av, grad=grad,
                                                 epg=self.__hist["epg"],
                                                 pfn_d=self.__hist["pfn_d"], pfn_v=self.__hist["pfn_v"],
                                                 hdb=self.__hist["hdb"], dfb=self.__hist["dfb"],
                                                 pfl2=self.__hist["pfl2"],
                                                 pfn_a=self.__hist["pfn_a"], pfn_p=self.__hist["pfn_p"],
                                                 hdc=self.__hist["hdc"], dfc=self.__hist["dfc"],
                                                 pfl3=self.__hist["pfl3"], phi=self.__hist["phi"][-1])

    def act(self):

        front = 1
        side = self.osc_b

        yaw = self.__hist['yaw'][-1]

        if len(self.__hist['yaw']) > 1:
            d_x = self.__hist['x'][-1] - self.__hist['x'][-2]
            d_y = self.__hist['y'][-1] - self.__hist['y'][-2]
            d_yaw = self.__hist['yaw'][-1] - self.__hist['yaw'][-2]
            phi = (self.__hist['yaw'][-1] + np.arctan2(d_x, d_y) - d_yaw + np.pi) % (2 * np.pi) - np.pi
        else:
            phi = 0.

        pfl2 = np.maximum(self.__hist["pfl2"][-1][8:] + self.__hist["pfl2"][-1][:8], 0)
        pfl2_v = np.sum(pfl2 * np.exp(-1j * np.linspace(0, 2 * np.pi, 8, endpoint=False)))
        pfl2_m = np.abs(pfl2_v)
        pfl2_a = np.angle(pfl2_v)  # - phi
        pfl3 = np.maximum(self.__hist["pfl3"][-1][8:] + self.__hist["pfl3"][-1][:8], 0)
        pfl3_v = np.sum(pfl3 * np.exp(-1j * np.linspace(0, 2 * np.pi, 8, endpoint=False)))
        pfl3_m = np.abs(pfl3_v)
        pfl3_a = np.angle(pfl3_v)  # - yaw
        print(f"yaw = {np.rad2deg(yaw):.2f}", end=";  ")
        print(f"PFL2_ang = {np.rad2deg(pfl2_a):.2f}, PFL2_mag = {pfl2_m:.2f}", end=";  ")
        print(f"PFL3_ang = {np.rad2deg(pfl3_a):.2f}, PFL3_mag = {pfl3_m:.2f}")

        lv = [0, front, 0]
        # lv = [side, front, 0]
        # av = R.from_euler("Z", self.osc_v)
        w = 1
        # lv = [side * (1 - w * np.imag(pfl2_v)), front * (1 - w * np.real(pfl2_v)), 0]
        # lv = [side * (1 - w * np.clip(pfl2_m, 0, 1)), front * (1 + w * np.clip(pfl2_m, 0, 1)), 0]
        # lv = [side, front * (1 - w * np.clip(pfl2_m, 0, 1)), 0]
        # av = R.from_euler("Z", np.clip(1 - pfl3_m, 0.2, 1) * self.osc_v + np.clip(pfl3_m, 0, .8) * pfl3_a)
        av = R.from_euler("Z", self.osc_c + .5 * np.clip(pfl3_a, -np.pi/2, np.pi/2))
        grad = self.__hist['g'][-1]

        return lv, av, grad

    def sense(self):
        x, y = self.agent.x, self.agent.y
        yaw = self.agent.yaw
        g = self.gradient(x, y, yaw)
        # g = self.gradient(x, y)
        self.__hist['g'].append(g)
        self.__hist['x'].append(x)
        self.__hist['y'].append(y)
        self.__hist['yaw'].append(yaw)

        if len(self.__hist['g']) > 1:
            d_g = self.__hist['g'][-1] - self.__hist['g'][-2]
            d_x = self.__hist['x'][-1] - self.__hist['x'][-2]
            d_y = self.__hist['y'][-1] - self.__hist['y'][-2]
            d_yaw = self.__hist['yaw'][-1] - self.__hist['yaw'][-2]
        else:
            d_g = 0.
            d_x = 0.
            d_y = 0.
            d_yaw = 0.

        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi

        phi = -((np.arctan2(d_x, d_y) + np.pi) % (2 * np.pi) - np.pi)
        self.__hist["phi"].append(phi)

        # calculate the relative direction of holonomic motion (excluding the heading)
        d_phi = (phi - self.__hist['yaw'][-1] + d_yaw + np.pi) % (2 * np.pi) - np.pi

        # calculate the relative magnitude (speed) of motion
        # rho = np.sqrt(np.square(d_x) + np.square(d_y)) * 20
        rho = 1
        # print(f"d_phi = {np.rad2deg(phi):.2f}, d_yaw={np.rad2deg(d_yaw):.2f}")

        p = 1
        nco = lambda a: np.power(.5 * np.cos(a) + .5, p)
        lin = lambda a: np.power(1 - np.clip(np.abs((a + np.pi) % (2 * np.pi) - np.pi) / np.pi, 0, 1), p)

        l_lno1 = rho * lin(d_phi - np.pi / 4) * float(d_yaw >= 0)
        r_lno1 = rho * lin(d_phi + np.pi / 4) * float(d_yaw <= 0)
        l_lno2 = rho * lin(d_phi - 3 * np.pi / 4) * float(d_yaw >= 0)
        r_lno2 = rho * lin(d_phi - 3 * np.pi / 4) * float(d_yaw <= 0)

        l_lno_a = lin(d_yaw - np.pi / 4) * float(d_yaw >= 0)
        r_lno_a = lin(d_yaw + np.pi / 4) * float(d_yaw <= 0)
        l_lno_p = lin(d_yaw - np.pi / 4) * float(d_yaw >= 0)
        r_lno_p = lin(d_yaw + np.pi / 4) * float(d_yaw <= 0)

        # l_lno1 = rho if 0. <= d_phi <= np.pi/2 else 0.
        # r_lno1 = rho if -np.pi/2 <= d_phi <= 0 else 0.
        # l_lno2 = rho if np.pi/2 <= d_phi <= np.pi else 0.
        # r_lno2 = rho if -np.pi <= d_phi <= -np.pi/2 else 0.

        # r_lno_a = (1. if d_yaw <= 0 else 0.) * (1 + np.cos(d_yaw)) / 2
        # l_lno_a = (1. if d_yaw >= 0 else 0.) * (1 + np.cos(d_yaw)) / 2
        # r_lno_p = (1. if d_yaw >= 0 else 0.) * (1 - np.cos(d_yaw)) / 2 if False else 1.
        # l_lno_p = (1. if d_yaw <= 0 else 0.) * (1 - np.cos(d_yaw)) / 2 if False else 1.

        l_epg = self.l_epg(-yaw)
        r_epg = self.r_epg(-yaw)
        self.__hist['epg'].append(np.r_[l_epg, r_epg])

        neg_w = 1
        d_g = np.clip(neg_w * d_g, neg_w * d_g, d_g)

        l_pfn_d = self.l_pfn(l_epg, r_lno2)
        r_pfn_d = self.r_pfn(r_epg, l_lno2)
        self.__hist['pfn_d'].append(np.r_[l_pfn_d, r_pfn_d])

        l_pfn_v = self.l_pfn(l_epg, r_lno1)
        r_pfn_v = self.r_pfn(r_epg, l_lno1)
        self.__hist['pfn_v'].append(np.r_[l_pfn_v, r_pfn_v])

        l_pfn_a = self.l_pfn(l_epg, r_lno_a)
        r_pfn_a = self.r_pfn(r_epg, l_lno_a)
        self.__hist['pfn_a'].append(np.r_[l_pfn_a, r_pfn_a])

        l_pfn_p = self.l_pfn(l_epg, r_lno_p)
        r_pfn_p = self.r_pfn(r_epg, l_lno_p)
        self.__hist['pfn_p'].append(np.r_[l_pfn_p, r_pfn_p])

        l_hdb = self.l_hdb(np.maximum(d_g, 0) * l_pfn_d, np.maximum(-d_g, 0) * l_pfn_v)
        r_hdb = self.r_hdb(np.maximum(d_g, 0) * r_pfn_d, np.maximum(-d_g, 0) * r_pfn_v)
        self.__hist['hdb'].append(np.r_[l_hdb, r_hdb])

        l_hdc = self.l_hdc(np.maximum(d_g, 0) * l_pfn_a, np.maximum(-d_g, 0) * l_pfn_p)
        r_hdc = self.r_hdc(np.maximum(d_g, 0) * r_pfn_a, np.maximum(-d_g, 0) * r_pfn_p)
        self.__hist['hdc'].append(np.r_[l_hdc, r_hdc])

        delta_phi_b = np.r_[l_hdb, r_hdb]
        delta_phi_c = np.r_[l_hdc, r_hdc]
        if len(self.__hist['dfb']) > 0:
            delta_phi_b = self.__hist['dfb'][-1] + delta_phi_b
            delta_phi_c = self.__hist['dfc'][-1] + delta_phi_c
            delta_phi_b /= np.max(delta_phi_b)
            delta_phi_c /= np.max(delta_phi_c)
        l_dfb = delta_phi_b[:8]
        r_dfb = delta_phi_b[8:]
        l_dfc = delta_phi_c[:8]
        r_dfc = delta_phi_c[8:]
        self.__hist['dfb'].append(delta_phi_b)  # side-walk
        self.__hist['dfc'].append(delta_phi_c)  # steering

        l_pfl2 = self.l_pfl2(-l_epg, l_dfb)
        r_pfl2 = self.r_pfl2(-r_epg, r_dfb)
        self.__hist['pfl2'].append(np.r_[l_pfl2, r_pfl2])
        l_pfl3 = self.l_pfl3(l_epg, l_dfc)
        r_pfl3 = self.r_pfl3(r_epg, r_dfc)
        self.__hist['pfl3'].append(np.r_[l_pfl3, r_pfl3])

        # default oscillation angle
        angle = (len(self.__hist['g']) + 4.5) * np.pi / 10

        self.osc_b = np.sin(angle)  # side-walk oscillation
        self.osc_c = (np.abs((angle + np.pi / 2) % (2 * np.pi) - np.pi) - np.pi / 2) / 4  # steering oscillation

    @property
    def gradient(self):
        return self.__gradient

    @staticmethod
    def get_angle(population):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        n = population.shape[0] // 2
        lv = np.sum(population[:n] * np.exp(-1j * angles))
        rv = np.sum(population[n:] * np.exp(-1j * angles))

        return np.angle(lv + rv)


class SimulationBase(object):

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

            while self._iteration < self.nb_frames:
                dt = self.step(self._iteration)
                if self._iteration % self.message_intervals == 0:
                    print(self.message() + " - time: %.2f sec" % dt)

                self._iteration += 1
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


class RouteSimulation(SimulationBase):
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


class NavigationSimulationBase(SimulationBase, ABC):

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
        self._calibration_xyz = None

    def reset(self, nb_samples_calibrate=None):
        super().reset()

        self.agent.xyz = copy(self.central_point)
        self.calibration(nb_samples=nb_samples_calibrate)

    def calibration(self, nb_samples=None):
        if self.needs_calibration and hasattr(self.agent, "calibrate"):
            if hasattr(self.agent, "eye") and nb_samples is None:
                # the number of samples must be at least the same number as the dimensions of the input
                nb_samples = self.agent.eye.nb_ommatidia
            elif nb_samples is None:
                nb_samples = 1000
                
            self.agent.xyz = copy(self.central_point)
            self.agent.update = False

            self._calibration_xyz, _ = self.agent.calibrate(self.sky, self._world, nb_samples=nb_samples, radius=2.)

            self.agent.xyz = copy(self.central_point)
            self.agent.update = True

        return self._calibration_xyz

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
        mb_message = ""
        if hasattr(self.agent, 'mushroom_body'):
            mb = self.agent.mushroom_body
            mb_message = f" - reward: {mb.r_us[0].max():.0f}, "
            mb_message += f"KC: {mb.r_kc[0].max():.2f}, "
            mb_message += f"free-space: {mb.free_space*100:.2f}%, "
            fam = np.power(mb.familiarity[0, 0, ::2].mean(), 8) * 100
            mb_message += f"familiarity: {fam:.2f}%"

        cx_message = ""
        if hasattr(self.agent, 'central_complex'):
            cx = self.agent.central_complex
            if hasattr(cx.memory, "r_tn1"):
                cx_message = f" - Nod: {cx.memory.r_tn1[0]:2f}, {cx.memory.r_tn1[1]:2f}"

        return f"{message} - L: {self.d_central_point:.2f}m, C: {d_trav:.2f}m{cx_message}{mb_message}"

    @property
    def central_point(self):
        return self._central_point

    @property
    def distant_point(self):
        raise NotImplementedError()

    @property
    def d_central_point(self):
        """
        The distance between the agent and the central point.

        Returns
        -------
        float
        """
        return np.linalg.norm(self.agent.xyz - self._central_point)

    @property
    def needs_calibration(self):
        return hasattr(self.agent, "is_calibrated") and not self.agent.is_calibrated


class PathIntegrationSimulation(CentralPointNavigationSimulationBase):

    def __init__(self, route, zero_vector=False, *args, **kwargs):
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
            self._agent = VectorMemoryAgent(nb_feeders=1, speed=.01, rng=self.rng, noise=self._noise)

        self._compass_sensor = None
        for sensor in self.agent.sensors:
            if isinstance(sensor, PolarisationSensor):
                self._compass_sensor = sensor
        self._compass_model, self._cx = self.agent.brain[:2]

        self._foraging = True
        self._distant_point = route[-1, :3]
        self._zero_vector = zero_vector

        self.__file_data = None

        if not isinstance(self._agent, RouteFollowingAgent):
            delattr(self, 'r_mbon')

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
        self._stats["position"] = []
        self._stats["capacity"] = []
        self._stats["familiarity"] = []

        self.__file_data = None

        self.agent.ori = R.from_euler("Z", self.route[0, 3], degrees=True)
        self._foraging = True

    def init_stats(self):
        super().init_stats()
        self._stats["POL"] = []
        self._stats["SOL"] = []
        self._stats["TB1"] = []
        self._stats["CL1"] = []
        self._stats["CPU1"] = []
        self._stats["CPU4"] = []
        self._stats["CPU4mem"] = []

        if hasattr(self.agent, "eye"):
            self._stats["ommatidia"] = []

        if hasattr(self.agent, 'mushroom_body'):
            self._stats["PN"] = []
            self._stats["KC"] = []
            self._stats["MBON"] = []
            self._stats["DAN"] = []
            self._stats["familiarity"] = []
            self._stats["capacity"] = []
            self._stats["replace"] = []

    def init_inbound(self):
        """
        Sets up the inbound phase.
        Changes the labels of the logged stats to their outbound equivalent and resets them for the new phase to come.
        """
        CentralPointNavigationSimulationBase.init_inbound(self)

        if self._zero_vector:
            self.agent.xyz = self.route[0, :3]
            self.agent.ori = R.from_euler("Z", self.route[0, 3], degrees=True)
            self.agent.central_complex.reset_integrator()

        file_path = os.path.join(__outb_dir__, f"{self.name}.npz")
        if not os.path.exists(file_path):
            np.savez(file_path, **self.stats)
            print(f"Outbound stats are saved in: '{file_path}'")

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
        omm_responses = None
        if i < self._route.shape[0]:  # outbound
            x, y, z, yaw = self._route[i]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
            # for process in self.agent.preprocessing:
            #     if isinstance(process, MentalRotation):
            #         process.pref_angles[:] = np.pi

            file_path = os.path.join(__outb_dir__, f"{self.name}.npz")
            if os.path.exists(file_path) and self.__file_data is None:
                print(f"Loading outbound stats from: '{file_path}'")
                data = np.load(file_path, allow_pickle=True)
                self.__file_data = {
                    "ommatidia": data["ommatidia"]
                }
            if self.__file_data is not None:
                omm_responses = self.__file_data["ommatidia"][i]

            self._foraging = True
        elif i == self._route.shape[0]:
            self.init_inbound()
            self._foraging = False
            # for process in self.agent.preprocessing:
            #     if isinstance(process, MentalRotation):
            #         process.pref_angles[:] = 0.
        elif self._foraging and self.distance_from(self.distant_point) < 0.5:
            self.approach_point(self.distant_point)
        elif not self._foraging and not self._zero_vector and self.d_nest < 0.5:
            self.approach_point(self.central_point)
        elif self._foraging and self.distance_from(self.distant_point) < 0.2:
            self._foraging = False
            print("START PI FROM FEEDER")
        elif not self._foraging and not self._zero_vector and self.d_nest < 0.2:
            self._foraging = True
            print("START FORAGING!")

        # if self._foraging:
        #     motivation = np.array([0, 1])
        # else:
        #     motivation = np.array([1, 0])

        if hasattr(self.agent, "mushroom_body"):
            self.agent.mushroom_body.update = self._foraging
        self._agent(sky=self._sky, scene=self._world, act=act, callback=self.update_stats, omm_responses=omm_responses)

        if i > self.route.shape[0] and "replace" in self._stats:
            d_route = np.linalg.norm(self.route[:, :3] - self._agent.xyz, axis=1)
            point = np.argmin(d_route)
            if d_route[point] > 0.2:  # move for more than 20cm away from the route
                self.agent.xyz = self.route[point, :3]
                self.agent.ori = R.from_euler('Z', self.route[point, 3], degrees=True)
                self._stats["replace"].append(True)
                print(" ~ REPLACE ~")
            else:
                self._stats["replace"].append(False)

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

        if hasattr(a, "eye"):
            if self.__file_data is not None and self._iteration < len(self.__file_data["ommatidia"]):
                self._stats["ommatidia"].append(self.__file_data["ommatidia"][self._iteration])
            else:
                self._stats["ommatidia"].append(a.eye.responses.copy())

        if hasattr(a, 'mushroom_body'):
            self._stats["PN"].append(a.mushroom_body.r_cs[0, 0].copy())
            self._stats["KC"].append(a.mushroom_body.r_kc[0, 0].copy())
            self._stats["MBON"].append(a.mushroom_body.r_mbon[0, 0].copy())
            self._stats["DAN"].append(a.mushroom_body.r_dan[0, 0].copy())
            self._stats["familiarity"].append(np.power(a.mushroom_body.familiarity[0, 0, ::2].mean(), 8) * 100)
            self._stats["capacity"].append(a.mushroom_body.free_space * 100)

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
    def r_mbon(self):
        if hasattr(self.agent, "mushroom_body"):
            return self.agent.mushroom_body.r_mbon[0]
        else:
            return None

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
            self._agent = VectorMemoryAgent(nb_feeders=2, speed=.01, rng=self.rng, noise=self._noise)

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

            self.central_complex.reset_current_memory()

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

        vector = self.get_vector()

        self.agent(sky=self._sky, act=act, vec=vector, callback=self.update_stats)

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

    def get_vector(self):
        vector = np.zeros(3, dtype=self.agent.dtype)
        if self._foraging:
            vector[self._forage_id] = 1
        else:
            vector[0] = 1
        return vector

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
    def central_complex(self):
        """

        Returns
        -------
        VectorMemoryCX
        """
        return self.agent.central_complex

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
        kwargs.setdefault('world', Seville2009())
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

        return self.calibration()

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
        vec = np.zeros(self.agent.central_complex.nb_vectors, dtype=self._mb.dtype)

        if self._learning_phase:
            vectors = np.eye(self.agent.mushroom_body.nb_us)
            # if self._foraging and i - self.i_offset < 20:
            if self._foraging:
                reinforcement[:] = [1, 0]
                # reinforcement[:] = vectors[(self._current_route_id + 1) * 2]
                # reinforcement[:] += vectors[1]
            # elif not self._foraging and i - self.i_offset < self.route.shape[0] + 20:
            elif not self._foraging:
                reinforcement[:] = [0, 1]
                # reinforcement[:] += vectors[(self._current_route_id + 1) * 2 + 1]
        # elif self.agent.central_complex.v_change:
        #     reinforcement[::2] = self.agent.central_complex.r_vec

        if np.all(np.isclose(self._food, 0)):
            self._food[:] = np.eye(len(self._odours))[0]

        if self._foraging:
            act = self.forage()
            vec[self._current_route_id + 1] = 1
        else:
            act = self.home()
            vec[0] = 1

        # if (len(self.stats["US"]) >= repeats and
        #     np.any(self.stats["US"][-1] > 0) and
        #     np.any(self.stats["US"][-1] != self.stats["US"][-repeats])):
        #     reinforcement[:] = self.stats["US"][-1]

        # self._food[:] = 0.
        self.agent(sky=self._sky, world=self._world, odours=self._odours, food=self._food, reinforcement=reinforcement,
                   vec=vec, act=act, callback=self.update_stats)

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
            self.agent.central_complex.reset_memory(self._current_route_id + 1)

            # pick up the food if it's there and go home
            stock = self._food_supply[self._current_route_id]
            if stock > 0 and self._food[0] < 1:
                # change the food state
                self._food[:] = np.eye(self.agent.nb_odours)[0]

                # deduct the food from the feeder
                self._food_supply[self._current_route_id] -= 1

                self.init_inbound()
                self._foraging = False

            elif stock == 0:
                # if there is no food in the feeder move to the next feeder
                self._current_route_id = (self._current_route_id + 1) % len(self.routes)
                self._food[:] = np.eye(self.agent.nb_odours)[self._current_route_id + 1]

            if np.isclose(self._food.sum(), 0):
                self._food[1] = 1.  # continue searching towards the first source
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
            self.agent.central_complex.reset_memory(0)
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
        agent: VisualProcessingAgent
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
        self._stats["L"] = []  # straight distance from the nest
        self._stats["C"] = []  # distance towards the nest that the agent has covered

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


class VisualFamiliarityDataCollectionSimulation(SimulationBase):

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


class VisualFamiliarityParallelExplorationSimulation(SimulationBase):

    def __init__(self, data, nb_par, nb_oris, agent=None, calibrate=False, saturation=5.,
                 order="rpo", pre_training=False, **kwargs):
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

        kwargs.setdefault('nb_iterations', int(route.shape[0]) + int(route_par.shape[0]))
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
        self._indices = np.arange(self.nb_frames)
        self._order = order

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
        self.__pre_training_count = 0

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
        self._indices[:] = np.arange(self.nb_frames)

        # default order is to rotate in each position before the agent moves to the next
        # default order: (x11, y11, r1), ..., (x11, y11, rN),
        #                (x12, y12, r1), ..., (x12, y12, rN),
        if "random" in self._order:
            # randomise the order of indices to remove order bias
            self._indices[self._route_length:] = self.rng.permutation(self._indices[self._route_length:])
        elif "r" in self._order and "p" in self._order and "o" in self._order:
            r = self._order.index("r")  # the route order
            p = self._order.index("p")  # parallel disposition
            o = self._order.index("o")  # rotation disposition
            order = [0, 0, 0]
            order[r] = 0
            order[p] = 1
            order[o] = 2
            order = tuple(order[::-1])

            # follow the route before changing to another route
            indices = self._indices[self._route_length:].reshape(-1, self.nb_par, self.nb_orientations)
            self._indices[self._route_length:] = np.transpose(indices, order).flatten()
        else:
            pass  # use the default order

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
        self._stats["xyz_out"] = copy(self._stats["xyz"])
        self._stats["capacity_out"] = copy(self._stats["capacity"])
        self._stats["familiarity_out"] = copy(self._stats["familiarity"])
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
        if self.__pre_training and self.__pre_training_count < int(self.__pre_training) and i % self._route_length == 0:
            self.reset()
            self.__pre_training_count += 1
        elif i >= self._route_length and self._agent.update:  # initialise route following
            self.init_grid()

        i = self._indices[i]

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
        return (super().message() +  # f" {self.mem._ic.r_mbon[0, [2, 3]]}" +
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


class Gradient:
    Grad = namedtuple('Gradient', ['progress', 'intensity'])

    def __init__(self, route, sigma=1, grad_type="gaussian"):
        self.__route = route[:, :2]
        self.__sigma = sigma

        assert grad_type in {"gaussian", "linear"}
        self.__type = eval(f"Gradient.{grad_type}")
        self.__g = 0.
        self.__tan = np.nan
        self.__dist = np.nan
        self.__baseline = 0.1
        self.__exp = 1.

    def __call__(self, x, y, yaw=None):
        g = self.__grad(x, y, yaw=yaw)
        return g.intensity * (g.progress * .2 + .8)

    def __grad(self, x, y, yaw=None):
        d = np.vstack([[x - self.route[:, 0]], [y - self.route[:, 1]]]).T
        m = np.linalg.norm(d, axis=1)
        i = np.argmin(m)
        if yaw is not None:
            if i <= 0:
                i = 1
            elif i >= self.route.shape[0] - 1:
                i = self.route.shape[0] - 1

            tang = -np.arctan2(self.route[i, 0] - self.route[i-1, 0], self.route[i, 1] - self.route[i-1, 1])

            self.__tan = tang
            w = ((1 - self.baseline) * np.exp(-self.exp * np.square((yaw - tang + np.pi) % (2 * np.pi) - np.pi)) +
                 self.baseline)
        else:
            w = 1.
        self.__dist = m[i]
        self.__g = self.Grad(
            progress=i / self.route.shape[0],
            intensity=w * self.__type(m[i], self.sigma))
        return self.__g

    def __repr__(self):
        return f"Gradient(route.length={self.route.shape[0]}, sigma={self.sigma})"

    @property
    def route(self):
        return self.__route

    @property
    def sigma(self):
        return self.__sigma

    @property
    def last_gradient(self):
        return self.__g

    @property
    def last_tangent(self):
        return self.__tan

    @property
    def last_distance(self):
        return self.__dist

    @staticmethod
    def gaussian(m, sigma):
        return np.exp(-0.5 * np.square(m / sigma))

    @staticmethod
    def linear(m, sigma):
        return np.clip((1 - 0.5 * m / sigma), 0, 1)

    @property
    def minx(self):
        return self.route[:, 0].min() - 2 * self.sigma

    @property
    def maxx(self):
        return self.route[:, 0].max() + 2 * self.sigma

    @property
    def miny(self):
        return self.route[:, 1].min() - 2 * self.sigma

    @property
    def maxy(self):
        return self.route[:, 1].max() + 2 * self.sigma

    @property
    def baseline(self):
        return self.__baseline

    @baseline.setter
    def baseline(self, value):
        self.__baseline = value

    @property
    def exp(self):
        return self.__exp

    @exp.setter
    def exp(self, value):
        self.__exp = value


def get_statsdir():
    return __stat_dir__


def set_statsdir(stats_dir):
    global __stat_dir__

    __stat_dir__ = stats_dir


def get_outbdir():
    return __outb_dir__


def set_outbdir(outb_dir):
    global __outb_dir__

    __outb_dir__ = outb_dir
