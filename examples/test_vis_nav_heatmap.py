from invertpy.brain.memory import PerfectMemory, WillshawNetwork, Infomax
from invertpy.sense import CompoundEye
from invertpy.brain.preprocessing import pca, zca, ZernikeMoments

from invertsy.agent import VisualNavigationAgent
from invertsy.sim.simulation import VisualFamiliarityGridExplorationSimulation
from invertsy.sim.animation import VisualFamiliarityAnimation

import numpy as np

import re


def main(*args):
    pattern = r"dataset-scan([0-9]+)-rows([0-9]+)-cols([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+).npz"
    data_filename = "dataset-scan16-rows100-cols100-ant1-route1-seville2009-omm1000.npz"

    save = True
    # model = "perfectmemory"
    model = "willshawnetwork"
    # model = "infomax"
    calibrate = True
    zernike = False
    ms = 16  # mental scanning

    details = re.match(pattern, data_filename)
    nb_scans = int(details.group(1))
    nb_rows = int(details.group(2))
    nb_cols = int(details.group(3))
    ant_no = int(details.group(4))
    rt_no = int(details.group(5))
    world_name = details.group(6)
    nb_ommatidia = int(details.group(7))
    if zernike:
        whitening = zca if calibrate else None
        nb_white = nb_ommatidia
        nb_input = ZernikeMoments.get_nb_coeff(16)
    else:
        whitening = pca if calibrate else None
        # nb_white = nb_ommatidia
        nb_white = nb_ommatidia
        nb_input = nb_white

    print("Heatmap simulation from data")
    print("File:", data_filename)

    if model in ["zernike"]:
        calibrate = False

    if model in ["perfectmemory"]:
        mem = PerfectMemory(nb_input=nb_input, maximum_capacity=813, dims=ms)
    elif model in ["infomax"]:
        mem = Infomax(nb_input=nb_input, eligibility_trace=0., dims=ms)
    else:
        nb_sparse = 40 * nb_input  # the sparse code should be 40 times larger that the input
        if zernike:
            nb_sparse = 4000  # The same number as Xuelong Sun uses
        sparseness = 10 / nb_sparse  # force 10 sparse neurons to be active (new)
        # sparseness = 5 / nb_sparse  # force 5 sparse neurons to be active
        mem = WillshawNetwork(nb_input=nb_input, nb_sparse=nb_sparse,
                              sparseness=sparseness, eligibility_trace=0., dims=ms)
    mem.novelty_mode = "%s%s%s" % ("zernike" if zernike else "",
                                   "-" if zernike and whitening is not None else "",
                                   "" if whitening is None else f"{whitening.__name__}")

    agent_name = "heatmap-%s%s%s-scan%d-rows%d-cols%d-ant%d-route%d-%s" % (
        mem.__class__.__name__.lower(),
        "-zernike" if zernike else "",
        "" if whitening is None else f"-{whitening.__name__}",
        nb_scans, nb_rows, nb_cols, ant_no, rt_no, world_name)
    agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
    agent_name += (f"x{ms:d}" if ms > 1 else "")
    print("Agent: %s" % agent_name)

    eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                      omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
    agent = VisualNavigationAgent(eye, mem, nb_visual=nb_white, nb_scans=1, speed=.01, mental_scanning=ms,
                                  whitening=whitening, zernike=zernike)
    sim = VisualFamiliarityGridExplorationSimulation(data_filename, nb_rows=nb_rows, nb_cols=nb_cols, nb_oris=nb_scans,
                                                     agent=agent, calibrate=calibrate, name=agent_name)
    sim.message_intervals = 80
    # ani = VisualFamiliarityAnimation(sim)
    # ani(save=save, show=not save, save_type="mp4", save_stats=save)
    sim(save=save)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
