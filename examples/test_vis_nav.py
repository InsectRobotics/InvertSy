from invertpy.brain.mushroombody import PerfectMemory, WillshawNetwork
from invertpy.sense import CompoundEye

from invertsy.agent import VisualNavigationAgent
from invertsy.env.seville2009 import load_routes, Seville2009
from invertsy.sim.simulation import VisualNavigationSimulation
from invertsy.sim.animation import VisualNavigationAnimation

import numpy as np


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

    save, show = True, False
    nb_scans = 121
    nb_ommatidia = 4000
    replace = True
    calibrate = True
    mem = PerfectMemory(nb_ommatidia)
    # mem = WillshawNetwork(nb_cs=nb_ommatidia, nb_kc=nb_ommatidia * 40, sparseness=0.01, eligibility_trace=.1)
    agent_name = "vn-%s%s-scan%d-ant%d-route%d%s" % (
        mem.__class__.__name__.lower(), "-pca" if calibrate else "", nb_scans, ant_no, rt_no, "-replace" if replace else "")
    agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
    print(" - Agent: %s" % agent_name, ("- save " if save else "") + ("- show" if show else ""))

    eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(15),
                      omm_res=5., c_sensitive=[0, 0., 1., 0., 0.])
    agent = VisualNavigationAgent(eye, mem, nb_scans=nb_scans, speed=.01)
    sim = VisualNavigationSimulation(rt, agent=agent, world=Seville2009(), calibrate=calibrate, nb_scans=nb_scans,
                                     nb_ommatidia=nb_ommatidia, name=agent_name, free_motion=not replace)
    ani = VisualNavigationAnimation(sim, show_history=True)
    ani(save=save, show=show, save_type="mp4")
    # sim(save=True)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
