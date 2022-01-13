from invertsy.env.world import Seville2009
from invertsy.sim.simulation import TwoSourcePathIntegrationSimulation
from invertsy.sim.animation import PathIntegrationAnimation

from scipy.spatial.transform import Rotation as R

import numpy as np


def main(*args):
    routes = Seville2009.load_routes(degrees=True)
    ant_no, rt_no, rtb = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rtb.shape[0]))

    shift = 20

    rtb = rtb[::-1]  # inverse route A

    rtb[:, :2] -= rtb[0, :2]  # modify to fit two routes
    rtb[:, :3] = R.from_euler("Z", shift, degrees=True).apply(rtb[:, :3]) + np.array([1., 5., 0.])
    rtb[:, 3] = (rtb[:, 3] + shift) % 360 - 180

    d = np.diff(rtb[:, :2], axis=0)
    phi = np.rad2deg(np.arctan2(d[:, 1], d[:, 0]))
    phi = (np.insert(phi, -1, phi[-1]) + 180) % 360 - 180
    rtb[:, 3] = phi
    # print(d.shape, phi.shape)
    # print(phi[:10])
    # print(rtb[:10, 3])
    #
    # sys.exit()

    rta = rtb.copy()  # mirror route A and create route B
    rta[:, [1, 3]] = -rta[:, [1, 3]]
    rta[:, :3] += rtb[0, :3] - rta[0, :3]

    # correct for the drift
    rta[:, 3] = (rta[:, 3] + 180) % 360 - 180
    rtb[:, 3] = (rtb[:, 3] + 180) % 360 - 180

    sim = TwoSourcePathIntegrationSimulation(rtb, rta, name="vmpi-ant%d-route%d" % (ant_no, rt_no))
    ani = PathIntegrationAnimation(sim, show_history=True)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
