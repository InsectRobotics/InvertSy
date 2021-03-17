from env.sky import Sky
from env.seville2009 import Seville2009

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.path import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np


def anim_route(eye, route, world=None, sky=None, cmap="Greens", max_intensity=2., title=None,
               fps=5, save=False, show=True):
    if sky is None:
        sky = Sky(30, 180, degrees=True)
    if world is None:
        world = Seville2009()
    if title is None:
        title = world.name

    yaw, pitch, roll = eye.omm_ori.as_euler('ZYX', degrees=True).T

    fig = plt.figure(title, figsize=(11, 5))

    ax1 = plt.subplot(221)
    ax1.set_yticks(np.sin([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]))
    ax1.set_yticklabels([-90, -60, -30, 0, 30, 60, 90])
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.set_ylim([-1, 1])
    ax1.set_xlim([-180, 180])

    ax2 = plt.subplot(122)
    for polygon, colour in zip(world.polygons, world.colours):
        x = polygon[[0, 1, 2, 0], 0]
        y = polygon[[0, 1, 2, 0], 1]
        ax2.plot(y, x, c=colour)

    ax2.set_ylim([0, 10])
    ax2.set_xlim([0, 10])
    ax2.set_aspect('equal', 'box')

    plt.tight_layout()

    r = eye(sky=sky, scene=world).mean(axis=1)

    size = np.sqrt(1000. * (4 * eye.omm_rho/np.pi + 1.) / eye.nb_ommatidia) * 20.
    omm1 = ax1.scatter(yaw.tolist(), (np.sin(np.deg2rad(-pitch))).tolist(), s=size,
                       c=np.zeros(yaw.shape[0], dtype='float32'), cmap=cmap, vmin=0, vmax=max_intensity)
    pos1, = ax2.plot([], [], 'r', lw=2)
    pos2 = ax2.scatter(eye.y, eye.x, marker=(3, 2, 0), s=100, c='red')

    points = [0, 2, 3, 4, 6]
    vert = np.array(pos2.get_paths()[0].vertices)[points]
    vert[0] *= 2
    codes = pos2.get_paths()[0].codes[points]
    marker = np.hstack([vert, np.zeros((vert.shape[0], 1))])

    omm1.set_array(r.T.flatten())

    def init():
        eye._xyz = route[0, :3]
        eye._ori = R.from_euler('Z', route[0, 3], degrees=True)

        r = eye(sky=sky, scene=world).mean(axis=1)
        omm1.set_array(r.T.flatten())

        pos1.set_data(route[0, 1], route[0, 0])
        pos2.set_offsets(np.array([eye.y, eye.x]))
        codes = pos2.get_paths()[0].codes
        vertices = R.from_euler('Z', -eye.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(marker)
        pos2.set_paths((Path(vertices[:, :2], codes),))

        return omm1, pos1, pos2

    def animate(i):
        t0 = time()

        eye._xyz = route[i, :3]
        eye._ori = R.from_euler('Z', route[i, 3], degrees=True)
        # eye._xyz += np.array([.1, 0, 0])

        r = eye(sky=sky, scene=world).mean(axis=1)

        omm1.set_array(r.T.flatten())
        pos1.set_data(route[:(i+1), 1], route[:(i+1), 0])
        pos2.set_offsets(np.array([eye.y, eye.x]))
        vertices = R.from_euler('Z', -eye.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(marker)
        pos2.set_paths((Path(vertices[:, :2], codes),))

        t1 = time()
        interval = t1 - t0
        print("Animation %d/%d - x: %.2f, y: %.2f, z: %.2f, Φ: %.2f - time: %.2f sec" % (
            i, route.shape[0], eye.x, eye.y, eye.z, (route[i, 3] + 180) % 360 - 180, interval))

        return omm1, pos1, pos2

    animate(0)

    ani = animation.FuncAnimation(fig, animate, frames=route.shape[0], interval=1000,
                                  blit=True, init_func=init)

    if save:
        ani.save("%s.gif" % title, fps=fps)

    if show:
        plt.show()

    return ani, ax1, ax2
