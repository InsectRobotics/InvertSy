from invertsy.env.world import Seville2009
from invertsy.sim.simulation import NavigationSimulation
from invertsy.sim.animation import NavigationAnimation

from scipy.spatial.transform import Rotation as R

import numpy as np


def main(*args):
    print("Mushroom Body and Central Complex simulation for vector memory.")
    routes = Seville2009.load_routes(degrees=True)
    ant_no, rt_no, rta = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rta.shape[0]))

    nb_routes = 2
    shift = 20

    rta = rta[::-1]  # inverse route A

    rta[:, :2] -= rta[0, :2]  # modify to fit two routes
    rta[:, :3] = R.from_euler("Z", shift, degrees=True).apply(rta[:, :3]) + np.array([1., 5., 0.])
    rta[:, 3] = (rta[:, 3] + shift) % 360 - 180

    d = np.diff(rta[:, :2], axis=0)
    phi = np.rad2deg(np.arctan2(d[:, 1], d[:, 0]))
    phi = (np.insert(phi, -1, phi[-1]) + 180) % 360 - 180
    rta[:, 3] = phi
    # print(d.shape, phi.shape)
    # print(phi[:10])
    # print(rta[:10, 3])
    #
    # sys.exit()

    rtb = rta.copy()  # mirror route A and create route B
    rtb[:, [1, 3]] = -rtb[:, [1, 3]]
    rtb[:, :3] += rta[0, :3] - rtb[0, :3]

    # correct for the drift
    rtb[:, 3] = (rtb[:, 3] + 180) % 360 - 180
    rta[:, 3] = (rta[:, 3] + 180) % 360 - 180

    sim = NavigationSimulation([rta, rtb], name=f"vector-memory-routes{nb_routes:02d}")
    ani = NavigationAnimation(sim)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
