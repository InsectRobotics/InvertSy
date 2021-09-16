from invertpy.brain.mushroombody import PerfectMemory, WillshawNetwork
from invertpy.sense import CompoundEye

from invertsy.agent import VisualNavigationAgent
from invertsy.env.world import Seville2009, SimpleWorld
from invertsy.sim.simulation import VisualFamiliaritySimulation
from invertsy.sim.animation import VisualFamiliarityAnimation

import numpy as np


def main(*args):
    routes = Seville2009.load_routes(degrees=True)

    calibrate = True
    save = True
    # world = SimpleWorld()
    world = Seville2009()

    nb_scans = 8
    nb_rows = 50
    nb_cols = 50
    nb_ommatidia = 2000

    print("Heatmap simulation")

    for ant_no, rt_no, rt in zip(routes['ant_no'], routes['route_no'], routes['path']):
        print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

        # mem = PerfectMemory(nb_ommatidia)
        mem = WillshawNetwork(nb_cs=nb_ommatidia, nb_kc=nb_ommatidia * 40, sparseness=0.01, eligibility_trace=.1)
        agent_name = "heatmap-%s%s-scan%d-rows%d-cols%d-ant%d-route%d-%s" % (
            mem.__class__.__name__.lower(),
            "-pca" if calibrate else "",
            nb_scans, nb_rows, nb_cols, ant_no, rt_no, world.__class__.__name__.lower())
        agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
        print(" - Agent: %s" % agent_name)

        eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                          omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
        agent = VisualNavigationAgent(eye, mem, speed=.01)
        sim = VisualFamiliaritySimulation(rt, agent=agent, world=world, calibrate=calibrate,
                                          nb_ommatidia=nb_ommatidia, name=agent_name,
                                          nb_orientations=nb_scans, nb_rows=nb_rows, nb_cols=nb_cols)
        ani = VisualFamiliarityAnimation(sim)
        ani(save=save, show=not save, save_type="mp4", save_stats=save)
        # sim(save=save)

        break


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
