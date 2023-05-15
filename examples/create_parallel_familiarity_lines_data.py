import os.path

from invertpy.sense import CompoundEye

from invertsy.env.world import Seville2009, SimpleWorld
from invertsy.sim.simulation import VisualFamiliarityDataCollectionSimulation, get_statsdir

import numpy as np


def main(*args):
    routes = Seville2009.load_routes(degrees=True)

    # world = SimpleWorld()
    world = Seville2009()

    nb_scans = 36
    nb_parallel = 1
    nb_ommatidia = 1000
    overwrite = False
    order_by = "route_no"

    print("Create familiarity map data simulation")

    order = np.argsort(routes[order_by])
    ant_nos = np.array(routes['ant_no'])[order]
    route_nos = np.array(routes['route_no'])[order]
    paths = np.array(routes['path'])[order]

    for ant_no, rt_no, rt in zip(ant_nos, route_nos, paths):
        print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

        agent_name = "dataset-scan%d-parallel%d-ant%d-route%d-%s" % (
            nb_scans, nb_parallel, ant_no, rt_no, world.__class__.__name__.lower())
        agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
        print(" - Agent: %s" % agent_name)

        if os.path.exists(os.path.join(get_statsdir(), f"{agent_name}.npz")) and not overwrite:
            print(f"File exists: {agent_name}.npz")
            continue

        eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(10),
                          omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
        sim = VisualFamiliarityDataCollectionSimulation(
            rt, eye=eye, world=world, nb_ommatidia=nb_ommatidia, name=agent_name,
            nb_orientations=nb_scans, nb_parallel=nb_parallel, method="parallel")
        sim.message_intervals = 1000
        sim(save=True)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
