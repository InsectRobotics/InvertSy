from invertpy.sense import CompoundEye

from invertsy.env.world import Seville2009, SimpleWorld
from invertsy.sim.simulation import VisualFamiliarityDataCollectionSimulation

import numpy as np


def main(*args):
    routes = Seville2009.load_routes(degrees=True)

    # world = SimpleWorld()
    world = Seville2009()

    nb_scans = 16
    nb_rows = 100
    nb_cols = 100
    nb_ommatidia = 1000

    print("Create familiarity map data simulation")

    for ant_no, rt_no, rt in zip(routes['ant_no'], routes['route_no'], routes['path']):
        print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

        agent_name = "dataset-scan%d-rows%d-cols%d-ant%d-route%d-%s" % (
            nb_scans, nb_rows, nb_cols, ant_no, rt_no, world.__class__.__name__.lower())
        agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
        print(" - Agent: %s" % agent_name)

        eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                          omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
        sim = VisualFamiliarityDataCollectionSimulation(
            rt, eye=eye, world=world, nb_ommatidia=nb_ommatidia, name=agent_name,
            nb_orientations=nb_scans, nb_rows=nb_rows, nb_cols=nb_cols)
        sim(save=True)

        break


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
