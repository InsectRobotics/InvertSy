from invertsensing.comoundeye import CompoundEye
from env.sky import Sky
from env.seville2009 import Seville2009

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from time import time

import matplotlib.pyplot as plt
import numpy as np
import sys


def main(*args):

    eye = CompoundEye(nb_input=5000, omm_pol_op=1, noise=0.,
                      xyz=np.array([0, 5, .01]), omm_rho=np.deg2rad(10),
                      ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
                      c_sensitive=[0, 0., 1., 0., 0.])
    sky = Sky(np.deg2rad(30), np.pi/2)
    world = Seville2009()

    yaw, pitch, roll = eye.omm_ori.as_euler('ZYX', degrees=True).T

    fig = plt.figure("Seville2009", figsize=(12, 5))
    ax1 = plt.subplot(121)
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

    omm1 = ax1.scatter(yaw.tolist(), (np.sin(np.deg2rad(-pitch))).tolist(), s=np.sqrt(4 * eye.omm_rho/np.pi + 1) * 20,
                       c=np.zeros(yaw.shape[0], dtype='float32'), cmap='Greens', vmin=0, vmax=4)
    pos2 = ax2.scatter(eye.y, eye.x, s=50, c='red')

    def init():
        omm1.set_array(np.zeros(yaw.shape[0], dtype='float32').T.flatten())
        pos2.set_offsets(np.array([eye.y, eye.x]))
        return omm1, pos2

    def animate(i):
        t0 = time()

        eye._xyz += np.array([.1, 0, 0])

        r = eye(sky=sky, scene=world)

        # hue = r[..., 0:1] * eye.hue_sensitive
        # rgb = hue[..., 1:4]
        # rgb[:, [0, 2]] += hue[..., 4:5] / 2
        # rgb[:, 0] += hue[..., 0]

        # omm1.set_array(np.clip(rgb, 0., 1.).T.flatten())
        print(r[..., 1].min(), r[..., 0].max())
        omm1.set_array(r[..., 0].T.flatten())
        pos2.set_offsets(np.array([eye.y, eye.x]))

        t1 = time()
        interval = t1 - t0

        print("Animation %d - x: %.2f, y: %.2f, z: %.2f - time: %.2f sec" % (
            i, eye.x, eye.y, eye.z, interval))

        return omm1, pos2

    animate(0)

    ani = animation.FuncAnimation(fig, animate, frames=100, interval=1000,
                                  blit=True, init_func=init)

    # ani.save("simple_nav.gif", fps=1)
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
