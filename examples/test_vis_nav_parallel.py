from invertpy.brain.memory import PerfectMemory, WillshawNetwork, Infomax
from invertpy.brain.mushroombody import IncentiveCircuitMemory, VisualIncentiveCircuit
from invertpy.sense import CompoundEye
from invertpy.brain.preprocessing import pca, zca, ZernikeMoments

from invertsy.agent import VisualNavigationAgent
from invertsy.sim.simulation import VisualFamiliarityParallelExplorationSimulation, get_statsdir

import numpy as np

from glob import glob

import os
import re


def main(*args):
    pattern = r"dataset-scan([0-9]+)-parallel([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+).npz"
    pattern_par = r"dataset-scan(1)-parallel([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+).npz"
    pattern_rot = r"dataset-scan([0-9]+)-parallel(1)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+).npz"
    # data_filename = "dataset-scan16-parallel21-ant1-route1-seville2009-omm1000.npz"
    # data_filename = "dataset-scan180-parallel1-ant1-route1-seville2009-omm1000.npz"

    save = True
    overwrite = False
    percentile_ommatidia = [.05, .01, .1, .2, .3, .4, .5]
    # models = ["infomax"]
    # models = ["incentivecircuit"]
    models = ["perfectmemory", "willshaw", "incentivecircuit"]
    # models = ["perfectmemory", "willshaw", "incentivecircuit", "incentivecircuitrand"]
    # models = ["perfectmemory", "willshaw", "infomax", "incentivecircuit"]
    lateral_inhibition = False
    calibrate = True
    zernike = False
    ms = 1  # mental scanning
    order = "rpo"  # "random"
    pre_trainings = [5, 4, 3, 2, 1]

    print("\nHeatmap simulation from data")

    for pre_training in pre_trainings:
        for percentile_omm in percentile_ommatidia:
            for filename in glob(os.path.join(get_statsdir(), "dataset-*.npz")):
                filename = filename.split("\\")[-1]
                details = re.match(pattern_par, filename)
                is_par, is_rot = False, False
                if details is None:
                    details = re.match(pattern_rot, filename)
                else:
                    is_par = True
                if details is None:
                    details = re.match(pattern, filename)
                else:
                    is_rot = True
                if details is None or not (is_par or is_rot):
                    continue

                # details = re.match(pattern, data_filename)
                nb_scans = int(details.group(1))
                nb_parallel = int(details.group(2))
                ant_no = int(details.group(3))
                rt_no = int(details.group(4))
                world_name = details.group(5)
                nb_ommatidia = int(details.group(6))

                for model in models:
                    if nb_scans > 1 or model not in ["infomax", "incentivecircuit"]:
                        pre_training_ = 1
                    else:
                        pre_training_ = pre_training

                    if zernike:
                        whitening = zca if calibrate else None
                        nb_white = nb_ommatidia
                        nb_input = ZernikeMoments.get_nb_coeff(16)
                        percentile_omm = float(nb_input) / float(nb_ommatidia)
                    else:
                        whitening = pca if calibrate else None
                        nb_white = int(nb_ommatidia * percentile_omm)
                        # nb_white = 600
                        nb_input = nb_white

                    print("\nRecorded file:", filename)

                    if model in ["zernike"]:
                        calibrate = False

                    nb_sparse = 4000  # fixed number for the KCs
                    # if zernike:
                    #     nb_sparse = 4000  # The same number as Xuelong Sun uses
                    # else:
                    #     # the sparse code should be 40 times larger that the input
                    #     nb_sparse = 40 * nb_input

                    sparseness = 10 / nb_sparse  # force 10 sparse neurons to be active (new)
                    # sparseness = 5 / nb_sparse  # force 5 sparse neurons to be active

                    if model in ["perfectmemory"]:
                        mem = PerfectMemory(nb_input=nb_input, maximum_capacity=1500, ndim=ms)
                    elif model in ["infomax"]:
                        mem = Infomax(nb_input=nb_input, eligibility_trace=0., ndim=ms)
                    elif model in ["willshaw", "willshawrand"]:
                        # the sparse code should be 40 times larger that the input
                        nb_sparse = 40 * nb_input
                        mem = WillshawNetwork(nb_input=nb_input, nb_sparse=nb_sparse,
                                              sparseness=sparseness, eligibility_trace=0., ndim=ms)
                        mem.reset()
                    elif model in ["incentivecircuit", "incentivecircuitrand"]:
                        mem = VisualIncentiveCircuit(nb_input=nb_input, nb_sparse=nb_sparse,
                                                     sparseness=sparseness, eligibility_trace=0., ndim=ms)
                        mem.reset()
                    else:
                        mem = PerfectMemory(nb_input=nb_input, maximum_capacity=1500, ndim=ms)

                    if model in ["willshawrand"]:
                        mem.w_i2s = np.array(mem.rng.random_sample(mem.w_i2s.shape) > 0.5, dtype=mem.dtype)
                    elif model in ["incentivecircuitrand"]:
                        mem.w_c2k = np.array(mem.rng.random_sample(mem.w_c2k.shape) > 0.5, dtype=mem.dtype)

                    # mem.novelty_mode = "%s%s%s" % ("zernike" if zernike else "",
                    #                                "-" if zernike and whitening is not None else "",
                    #                                "" if whitening is None else f"{whitening.__name__}")
                    mem.novelty_mode = ""

                    agent_name = "heatmap-%s%s%s%s-scan%d-par%d%s-ant%d-route%d-%s" % (
                        mem.__class__.__name__.lower() + ("rand" if "rand" in model else ""),
                        "-zernike" if zernike else "",
                        "" if whitening is None else f"-{whitening.__name__}{int(percentile_omm * 100):03d}",
                        "-li" if lateral_inhibition else "",
                        nb_scans, nb_parallel,
                        "" if order == "rpo" else order,
                        ant_no, rt_no, world_name)
                    agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
                    agent_name += (f"x{ms:d}" if ms > 1 else "")
                    if pre_training_ > 1:
                        agent_name += f"-{pre_training_}"
                    print("Agent: %s" % agent_name)

                    if os.path.exists(os.path.join(get_statsdir(), f"{agent_name}.npz")) and not overwrite:
                        print(f"File exists: {agent_name}.npz")
                        continue

                    eye = CompoundEye(nb_input=nb_ommatidia, omm_pol_op=0, noise=0., omm_rho=np.deg2rad(4),
                                      omm_res=10., c_sensitive=[0, 0., 1., 0., 0.])
                    agent = VisualNavigationAgent(mem, eye=eye, nb_visual=nb_white, nb_scans=1, speed=.01, mental_scanning=ms,
                                                  whitening=whitening, zernike=zernike, lateral_inhibition=lateral_inhibition)
                    sim = VisualFamiliarityParallelExplorationSimulation(filename, nb_par=nb_parallel, nb_oris=nb_scans,
                                                                         agent=agent, calibrate=calibrate, name=agent_name,
                                                                         pre_training=pre_training_, order=order)
                    sim.message_intervals = 500
                    # ani = VisualFamiliarityAnimation(sim)
                    # ani(save=save, show=not save, save_type="mp4", save_stats=save)
                    sim(save=save)

                    print("")

                    del mem, eye, agent, sim


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
