from invertpy.brain.mushroombody import PerfectMemory, WillshawNetwork
from invertpy.sense import CompoundEye

from invertsy.agent import VisualNavigationAgent
from invertsy.env.world import Seville2009
from invertsy.sim.simulation import VisualNavigationSimulation

import numpy as np


def main(*args):
    routes = Seville2009.load_routes(degrees=True)

    replace = True
    calibrate = True

    nb_scans = 121
    nb_ommatidia = 2000

    for ant_no, rt_no, rt in zip(routes['ant_no'], routes['route_no'], routes['path']):
        print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

        mem = PerfectMemory(nb_ommatidia)
        # mem = WillshawNetwork(nb_cs=nb_ommatidia, nb_kc=nb_ommatidia * 40, sparseness=0.01, eligibility_trace=.1)
        agent_name = "vn-%s%s-scan%d-ant%d-route%d%s" % (
            mem.__class__.__name__.lower(),
            "-pca" if calibrate else "",
            "-scan%d" % nb_scans if nb_scans > 1 else "",
            ant_no, rt_no,
            "-mr%d" % nb_mental_rotations if nb_mental_rotations > 1 else "",
            "-replace" if replace else "")
        agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
        print(" - Agent: %s" % agent_name)

        eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                          omm_res=5., c_sensitive=[0, 0., 1., 0., 0.])
        agent = VisualNavigationAgent(eye, mem, nb_scans=nb_scans, speed=.01)
        sim = VisualNavigationSimulation(rt, agent=agent, world=Seville2009(), calibrate=calibrate, nb_scans=nb_scans,
                                         nb_ommatidia=nb_ommatidia, name=agent_name, free_motion=not replace)
        sim(save=True)

        break


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
