from invertsensing.comoundeye import CompoundEye
from sky.sky import Sky
from world.seville2009 import Seville2009, SKY_COLOUR

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import numpy as np
import sys


def main(*args):
    eye = CompoundEye(nb_input=5000, omm_pol_op=1, noise=0.,
                      xyz=np.array([5, 5, .01]),
                      ori=R.from_euler('ZYX', [45, 0, 0], degrees=True))
    sky = Sky(np.deg2rad(89), np.pi)
    world = Seville2009()

    r = eye(sky=sky, scene=world)

    # c = world(eye.omm_xyz, ori=eye.ori * eye.omm_ori)

    yaw, pitch, roll = eye.omm_ori.as_euler('ZYX', degrees=True).T

    plt.figure("Seville2009", figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(yaw, -pitch, s=20, c=r[..., 0], cmap='Greens', vmax=5, vmin=0)  # np.clip(c/255., 0, 1))
    plt.ylim([-90, 90])
    plt.xlim([-180, 180])

    plt.subplot(122)
    for polygon, colour in zip(world.polygons, world.colours):
        x = polygon[[0, 1, 2, 0], 0]
        y = polygon[[0, 1, 2, 0], 1]
        plt.plot(y, x, c=colour)
    plt.scatter(eye.y, eye.x, s=50, c='red')
    plt.ylim([0, 10])
    plt.xlim([0, 10])
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
