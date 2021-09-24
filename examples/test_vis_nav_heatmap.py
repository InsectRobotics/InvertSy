from invertpy.brain.mushroombody import PerfectMemory, WillshawNetwork
from invertpy.sense import CompoundEye

from invertsy.agent import VisualNavigationAgent
from invertsy.sim.simulation import VisualFamiliarityGridExplorationSimulation
from invertsy.sim.animation import VisualFamiliarityAnimation

import numpy as np


def main(*args):
    data_filename = ""

    calibrate = True
    save = True
    nb_ommatidia, nb_scans, nb_rows, nb_cols, ant_no, rt_no = 1000, 16, 100, 100, 1, 1

    print("Heatmap simulation from data")
    print("File:", data_filename)

    # mem = PerfectMemory(nb_ommatidia)
    mem = WillshawNetwork(nb_cs=nb_ommatidia, nb_kc=nb_ommatidia * 40, sparseness=0.01, eligibility_trace=0.)
    agent_name = "heatmap-%s%s-scan%d-rows%d-cols%d-ant%d-route%d-%s" % (
        mem.__class__.__name__.lower(),
        "-pca" if calibrate else "",
        nb_scans, nb_rows, nb_cols, ant_no, rt_no, "seville2009")
    agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
    print(" - Agent: %s" % agent_name)

    eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                      omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
    agent = VisualNavigationAgent(eye, mem, speed=.01)
    sim = VisualFamiliarityGridExplorationSimulation(data_filename, agent=agent, calibrate=calibrate, name=agent_name)
    ani = VisualFamiliarityAnimation(sim)
    ani(save=save, show=not save, save_type="mp4", save_stats=save)
    # sim(save=save)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
