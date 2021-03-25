__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

from ._helpers import RNG, add_noise

from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from matplotlib import cm

import numpy as np
import os

__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
__data__ = os.path.join(__root__, "data")
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


class Seville2009(object):
    def __init__(self, horizon=2., dtype='float32', name="seville_2009"):
        self.__polygons, self.__colours = load_world()
        self.__horizon = horizon
        self.dtype = dtype
        self.name = name

    def __call__(self, pos, ori: R = None, init_c=None, brightness=1., noise=0., eta=None, rng=RNG):
        if init_c is None:
            c = np.full((np.shape(ori)[0], 3), np.nan, dtype=self.dtype)
        else:
            c = luminance2skycolour(init_c)

        yaw, pitch, roll = ori.as_euler('ZYX', degrees=False).T
        # omm_xyz = ori.apply([1, 0, 0])
        # yaw = np.arctan2(omm_xyz[..., 1], omm_xyz[..., 0])
        # pitch = np.pi/2 - np.arctan2(np.linalg.norm(omm_xyz[..., :2], axis=1), omm_xyz[..., 2])

        # Add uniform ground below the horizon
        c[pitch >= 0, :] = np.array(GROUND_COLOUR) / 255.

        # Add the polygons (vegetation)
        # calculate the relative position of each point in the polygon
        xyz = self.polygons - np.nanmean(pos, axis=0)

        # calculate the distance from the closest point of the polygon to the agent's position
        dist = np.min(np.linalg.norm(xyz, axis=2), axis=1)
        visible = dist <= self.horizon
        polygons = xyz[visible]

        # free the space of the polygons that were not rendered
        to_keep = ~np.isnan(np.linalg.norm(polygons, axis=(1, 2)))
        colours = self.colours[visible][to_keep]
        polygons = polygons[to_keep]

        ind = np.argsort(dist[visible])[::-1]

        for polygon, colour in zip(polygons[ind], colours[ind]):
            phi = np.arctan2(polygon[..., 1], polygon[..., 0])
            theta = np.arctan2(np.linalg.norm(polygon[..., :2], axis=1), polygon[..., 2]) - np.pi/2

            poi = np.array([pitch, yaw]).T
            pol = np.array([theta, phi]).T

            # check for the case that the object is on the edge of the screen
            if phi.max() - phi.min() >= np.pi:
                p1, p2 = pol.copy(), pol.copy()
                p1[:, 1][phi < 0] += 2 * np.pi
                p2[:, 1][phi > 0] -= 2 * np.pi
                i = same_side_technique(poi, p1) | same_side_technique(poi, p2)
            else:
                i = same_side_technique(poi, pol)
            c[i] = colour

        if eta is None:
            eta = add_noise(noise=noise, shape=c.shape, rng=rng)
        c[eta] = 0.

        return c

    @property
    def polygons(self):
        return self.__polygons

    @property
    def colours(self):
        return self.__colours

    @property
    def horizon(self):
        return self.__horizon


def luminance2skycolour(y):
    return np.clip(19. * y[..., np.newaxis] + np.array(SKY_COLOUR).reshape((1, -1)), 0, 255)


def same_side_technique(points, triangles):
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    return same_side(points, a, b, c) & same_side(points, b, a, c) & same_side(points, c, a, b)


def same_side(p1, p2, a, b):
    return np.multiply(np.cross(b - a, p1 - a), np.cross(b - a, p2 - a)) >= 0


def load_routes(routes_filename=ROUTES_FILENAME, degrees=False):
    mat = loadmat(os.path.join(__seville_2009__, routes_filename))
    ant, route, key = 1, 1, lambda a, r: "Ant%d_Route%d" % (a, r)
    routes = {"ant_no": [], "route_no": [], "path": []}
    while key(ant, route) in mat.keys():
        while key(ant, route) in mat.keys():
            mat[key(ant, route)][:, :2] /= 100.  # convert the route data to meters
            xs, ys, phis = mat[key(ant, route)].T
            phis = (90 - phis + 180) % 360 - 180
            r = np.zeros((xs.shape[0], 4))
            r[:, 0] = ys
            r[:, 1] = xs
            r[:, 2] = 0.01
            r[:, 3] = phis if degrees else np.deg2rad(phis)
            routes["ant_no"].append(ant)
            routes["route_no"].append(route)
            routes["path"].append(r)
            route += 1
        ant += 1
        route = 1
    return routes


def load_route(name):
    path = os.path.join(__data__, "routes", name + ".npz")
    data = np.load(path)
    return data["path"], data["ant"], data["route"]


def save_route(name, **rt):
    path = os.path.join(__data__, "routes", name + ".npz")
    np.save(path, **rt)


def load_world(world_filename=WORLD_FILENAME, dtype='float32'):
    mat = loadmat(os.path.join(__seville_2009__, world_filename))

    polygons = []
    colours = []
    green = np.array([0, 1, 1], dtype=dtype)
    for xs, ys, zs, col in zip(mat['X'], mat['Y'], mat['Z'], mat['colp']):
        polygons.append([ys, xs, zs])
        colours.append(col * green)

    polygons = np.transpose(np.array(polygons, dtype=dtype), axes=(0, 2, 1))
    colours = np.array(colours, dtype=dtype)
    colours[:, [0, 2]] = 0.

    return polygons, colours


def get_terrain(max_altitude=.5, tau=.6, x=None, y=None):
    global z_terrain

    terrain_path = os.path.join(__seville_2009__, "terrain-%.2f.npz" % tau)
    # create terrain
    if x is None or y is None:
        x, y = np.meshgrid(x_terrain, y_terrain)
    try:
        z = np.load(terrain_path)["terrain"] * 1000 * max_altitude
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

        np.savez_compressed(terrain_path % tau, terrain=terrain)
        z = terrain
    z_terrain = z
    return z

