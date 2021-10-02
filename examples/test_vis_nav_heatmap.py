from invertpy.brain.memory import PerfectMemory, WillshawNetwork
from invertpy.sense import CompoundEye

from invertsy.agent import VisualNavigationAgent
from invertsy.sim.simulation import VisualFamiliarityGridExplorationSimulation
from invertsy.sim.animation import VisualFamiliarityAnimation

import numpy as np

import re


def main(*args):
    pattern = r"dataset-scan([0-9]+)-rows([0-9]+)-cols([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+).npz"
    data_filename = "dataset-scan16-rows100-cols100-ant1-route1-seville2009-omm1000.npz"

    calibrate = True
    save = True

    details = re.match(pattern, data_filename)
    nb_scans = int(details.group(1))
    nb_rows = int(details.group(2))
    nb_cols = int(details.group(3))
    ant_no = int(details.group(4))
    rt_no = int(details.group(5))
    world_name = details.group(6)
    nb_ommatidia = int(details.group(7))

    print("Heatmap simulation from data")
    print("File:", data_filename)

    # mem = PerfectMemory(nb_input=nb_ommatidia, maximum_capacity=813)
    mem = WillshawNetwork(nb_input=nb_ommatidia, nb_sparse=nb_ommatidia * 40, sparseness=0.01, eligibility_trace=0.)
    agent_name = "heatmap-%s%s-scan%d-rows%d-cols%d-ant%d-route%d-%s" % (
        mem.__class__.__name__.lower(),
        "-pca" if calibrate else "",
        nb_scans, nb_rows, nb_cols, ant_no, rt_no, world_name)
    agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
    print("Agent: %s" % agent_name)

    eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                      omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
    agent = VisualNavigationAgent(eye, mem, nb_scans=1, speed=.01)
    sim = VisualFamiliarityGridExplorationSimulation(data_filename, nb_rows=nb_rows, nb_cols=nb_cols, nb_oris=nb_scans,
                                                     agent=agent, calibrate=calibrate, name=agent_name)
    # ani = VisualFamiliarityAnimation(sim)
    # ani(save=save, show=not save, save_type="mp4", save_stats=save)
    sim(save=save)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
