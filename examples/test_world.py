from invertsy.env.world import SimpleWorld, Seville2009
from invertsy.sim.simulation import RouteSimulation
from invertsy.sim.animation import RouteAnimation

from invertpy.brain.preprocessing import LateralInhibition
from invertpy.sense.vision import CompoundEye

import numpy as np
from copy import copy


def main(*args):
    routes = Seville2009.load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    eye = CompoundEye(nb_input=1000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10),
                      omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
    sim = RouteSimulation(rt, eye=eye, world=Seville2009(),
                          # preprocessing=[LateralInhibition(copy(eye.omm_ori))],
                          name="seville-ant%d-route%d" % (ant_no, rt_no))
    ani = RouteAnimation(sim)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
