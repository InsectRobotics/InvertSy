from invertpy.brain.preprocessing import pca, zca, ZernikeMoments
from invertpy.brain.memory import PerfectMemory, WillshawNetwork, Infomax, IncentiveCircuitMemory
from invertpy.sense import CompoundEye

from invertsy.agent import NavigatingAgent
from invertsy.env.world import Seville2009
from invertsy.sim.simulation import VisualNavigationSimulation
from invertsy.sim.animation import VisualNavigationAnimation

import numpy as np


def main(*args):
    routes = Seville2009.load_routes(degrees=True)

    # model = "perfectmemory"
    model = "incentivecircuit"
    # model = "willshaw"
    # model = "infomax"

    save = False
    replace = True

    lateral_inhibition = True
    calibrate = False
    zernike = False
    ms = 1  # mental scanning
    nb_ommatidia = 1000
    percentile_omm = .1

    if zernike:
        whitening = zca if calibrate else None
        nb_white = nb_ommatidia
        nb_input = ZernikeMoments.get_nb_coeff(16)
    else:
        whitening = pca if calibrate else None
        nb_white = int(nb_ommatidia * percentile_omm)
        # nb_white = 600
        nb_input = nb_white
    if not calibrate:
        nb_white = nb_ommatidia
        nb_input = nb_white

    if model in ["zernike"]:
        calibrate = False

    if model in ["perfectmemory"]:
        mem = PerfectMemory(nb_input=nb_input, maximum_capacity=813, dims=ms)
    elif model in ["infomax"]:
        mem = Infomax(nb_input=nb_input, eligibility_trace=0., dims=ms)
    elif model in ["willshaw"]:
        # the sparse code should be 40 times larger that the input
        nb_sparse = 40 * nb_input
        # nb_sparse = 4000  # fixed number for the KCs
        # if zernike:
        #     nb_sparse = 4000  # The same number as Xuelong Sun uses
        sparseness = 10 / nb_sparse  # force 10 sparse neurons to be active (new)
        # sparseness = 5 / nb_sparse  # force 5 sparse neurons to be active
        mem = WillshawNetwork(nb_input=nb_input, nb_sparse=nb_sparse,
                              sparseness=sparseness, eligibility_trace=0., dims=ms)
        mem.reset()
    else:
        # the sparse code should be 40 times larger that the input
        nb_sparse = 40 * nb_input
        # nb_sparse = 4000  # fixed number for the KCs
        # if zernike:
        #     nb_sparse = 4000  # The same number as Xuelong Sun uses
        sparseness = 10 / nb_sparse  # force 10 sparse neurons to be active (new)
        # sparseness = 5 / nb_sparse  # force 5 sparse neurons to be active
        mem = IncentiveCircuitMemory(nb_input=nb_input, nb_sparse=nb_sparse,
                                     sparseness=sparseness, eligibility_trace=0., ndim=ms)
        mem.reset()

    mem.novelty_mode = ""

    for ant_no, rt_no, rt in zip(routes['ant_no'], routes['route_no'], routes['path']):
        print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

        agent_name = "vn-%s%s-ant%d-route%d%s" % (
            mem.__class__.__name__.lower(),
            "-pca" if calibrate else "",
            ant_no, rt_no,
            "-replace" if replace else "")
        agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
        print(" - Agent: %s" % agent_name)

        eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                          omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
        agent = NavigatingAgent(mem, eye=eye, nb_visual=nb_white, speed=.01, mental_scanning=ms,
                                whitening=whitening, zernike=zernike, lateral_inhibition=lateral_inhibition)
        sim = VisualNavigationSimulation(rt, agent=agent, world=Seville2009(), calibrate=calibrate,
                                         nb_ommatidia=nb_ommatidia, name=agent_name, free_motion=not replace)

        sim.message_intervals = 10
        # sim(save=save)

        ani = VisualNavigationAnimation(sim)

        ani(save=save, save_type="mpeg", show=not save)

        break


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
