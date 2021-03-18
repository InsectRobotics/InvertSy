from invertsensing.vision import CompoundEye
from env.seville2009 import Seville2009, load_routes
from simplot.routes import anim_route

import numpy as np
import sys


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    eye = CompoundEye(nb_input=5000, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10),
                      c_sensitive=[0, 0., 1., 0., 0.])

    world = Seville2009()

    anim_route(eye, rt, world=world, max_intensity=3., save=False, show=True, fps=10,
               title="seville-ant%d-route%d-int5" % (ant_no, rt_no))


if __name__ == '__main__':
    main(*sys.argv)
