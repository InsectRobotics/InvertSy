from invertsensing.comoundeye import CompoundEye
from env.seville2009 import Seville2009, load_routes
from simplot.routes import anim_route

import sys


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    eye = CompoundEye(nb_input=5000, omm_pol_op=0, noise=0.,
                      c_sensitive=[0, 0., 1., 0., 0.])

    world = Seville2009()

    anim_route(eye, rt, world=world, max_intensity=4., save=True, show=False,
               title="seville-ant%d-route%d" % (ant_no, rt_no))


if __name__ == '__main__':
    main(*sys.argv)
