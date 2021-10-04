from invertsy.sim._helpers import create_familiarity_map, col2x, row2y, ori2yaw
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
        # filename = "heatmap-perfectmemory-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        filename = "heatmap-willshawnetwork-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"

    data = np.load(os.path.join(__stat_dir__, "%s.npz" % filename))

    heatmap = data["familiarity_map"]
    plt.figure(filename, figsize=(5, 5))
    fam, qui = create_familiarity_map(nb_rows=heatmap.shape[0], nb_cols=heatmap.shape[1])
    fammap = np.max(np.transpose(heatmap, axes=(1, 0, 2)), axis=2)
    # fam.set_array((fammap - fammap.min()) / (fammap.max() - fammap.min()))

    fam.set_array(fammap)
    # famdir = ring2complex(np.transpose(heatmap, axes=(1, 0, 2)), axis=2)
    # qui.set_UVC(famdir.imag, famdir.real)
    # plt.colorbar()

    plt.show()

    score = get_score(heatmap, data["position_out"], order=8) * 100.
    print("Score: %.4f %%" % score)


def get_score(familiarity_map, training_route, order=8):
    nb_rows, nb_cols, nb_oris = familiarity_map.shape
    row, col, ori = np.array([index for index in np.ndindex(familiarity_map.shape)]).T
    x = col2x(col, nb_cols=nb_cols, max_meters=10.)
    y = row2y(row, nb_rows=nb_rows, max_meters=10.)

    p = x + 1j * y
    # yaw = ori2yaw(ori, nb_oris=nb_oris, degrees=True)

    x_r, y_r, _, yaw_r = training_route.T
    p_r = x_r + 1j * y_r

    d = np.absolute(p[:, np.newaxis] - p_r[np.newaxis, :]).min(axis=1) / 10.
    d_map = d.reshape(familiarity_map.shape).min(axis=-1)
    f_map = familiarity_map.max(axis=-1)
    p_map = 1 - np.power(1 - d_map, order)

    return np.sum(p_map * f_map) / p_map.sum()  # - np.sum((1 - p_map) * f_map) / (1 - p_map).sum()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
