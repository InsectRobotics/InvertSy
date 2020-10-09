#!/usr/bin/env python

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2019, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

from geometry3 import Route

from scipy.io import loadmat
from matplotlib import cm

import numpy as np
import os

__root__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.realpath(os.path.join(__root__, "..", "data"))
__seville_2009__ = os.path.join(__data__, "Seville2009_world")
WORLD_FILENAME = "world5000_gray.mat"
ROUTES_FILENAME = "AntRoutes.mat"


cmap = cm.get_cmap('brg')

WIDTH = 36
HEIGHT = 10
LENGTH = 36

GRASS_COLOUR = (0, 255, 0)
GROUND_COLOUR = (229, 183, 90)
SKY_COLOUR = (13, 135, 201)

x_terrain = np.linspace(0, 10, 1001, endpoint=True)
y_terrain = np.linspace(0, 10, 1001, endpoint=True)
x_terrain, y_terrain = np.meshgrid(x_terrain, y_terrain)
z_terrain = np.zeros_like(x_terrain)


def load_routes(routes_filename=ROUTES_FILENAME):
    mat = loadmat(os.path.join(__seville_2009__, routes_filename))
    ant, route, key = 1, 1, lambda a, r: "Ant%d_Route%d" % (a, r)
    routes = []
    while key(ant, route) in mat.keys():
        while key(ant, route) in mat.keys():
            mat[key(ant, route)][:, :2] /= 100.  # convert the route data to meters
            xs, ys, phis = mat[key(ant, route)].T
            r = Route(xs, ys, .01, phis=np.deg2rad(phis), agent_no=ant, route_no=route)
            routes.append(r)
            route += 1
        ant += 1
        route = 1
    return routes


def load_route(name):
    return Route.from_file(__data__ + "routes/" + name + ".npz")


def save_route(rt, name):
    rt.save(__data__ + "routes/" + name + ".npz")


def get_terrain(max_altitude=.5, tau=.6, x=None, y=None):
    global z_terrain

    # create terrain
    if x is None or y is None:
        x, y = np.meshgrid(x_terrain, y_terrain)
    try:
        z = np.load("data/terrain-%.2f.npz" % 0.6)["terrain"] * 1000 * max_altitude
    except IOError:
        z = np.random.randn(*x.shape) / 50
        terrain = np.zeros_like(z)

        for i in range(terrain.shape[0]):
            print("%04d / %04d" % (i + 1, terrain.shape[0]),)
            for j in range(terrain.shape[1]):
                k = np.sqrt(np.square(x[i, j] - x) + np.square(y[i, j] - y)) < tau
                terrain[i, j] = z[k].mean()
                if j % 20 == 0:
                    print(".",)
            print()

        np.savez_compressed("terrain-%.2f.npz" % tau, terrain=terrain)
        z = terrain
    z_terrain = z
    return z

