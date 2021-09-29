from invertsy.sim._helpers import create_familiarity_map
from invertsy.sim.simulation import __stat_dir__

from invertpy.brain.compass import ring2complex

import matplotlib.pyplot as plt
import numpy as np
import os


def main(*args):
    if len(args) > 1:
        filename = args[1]
    else:
        # filename = "heatmap-perfectmemory-pca-scan8-rows50-cols50-ant1-route1-simpleworld-omm2000"
        # filename = "heatmap-perfectmemory-pca-scan8-rows50-cols50-ant1-route1-seville2009-omm2000"
        # filename = "heatmap-willshawnetwork-pca-scan8-rows50-cols50-ant1-route1-simpleworld-omm2000"
        # filename = "heatmap-willshawnetwork-pca-scan8-rows50-cols50-ant1-route1-seville2009-omm2000"
        filename = "heatmap-willshawnetwork-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"

    data = np.load(os.path.join(__stat_dir__, "%s.npz" % filename))

    heatmap = data["familiarity_map"]
    print(heatmap.shape, heatmap.min(), heatmap.max())
    # heatmap = np.transpose(heatmap, axes=(1, 0, 2))
    heatmap = 1 - (1 - heatmap) / (1 - heatmap).max()
    plt.figure(filename, figsize=(5, 5))
    fam, qui = create_familiarity_map(nb_rows=heatmap.shape[0], nb_cols=heatmap.shape[1])
    fammap = np.max(heatmap, axis=2)
    fam.set_array((fammap - fammap.min()) / (fammap.max() - fammap.min()))
    famdir = ring2complex(heatmap, axis=2)
    qui.set_UVC(famdir.imag, famdir.real)
    # plt.colorbar()

    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
