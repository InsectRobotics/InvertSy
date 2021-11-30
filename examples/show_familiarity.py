from invertsy.sim._helpers import create_familiarity_map, col2x, row2y
from invertsy.sim.simulation import __stat_dir__

from invertpy.brain.compass import ring2complex

from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np
import os


def main(*args):
    if len(args) > 1:
        filename = args[1]
    else:
        # filename = "heatmap-perfectmemory-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-scan16-rows100-cols100-ant1-route1-seville2009-omm1000x16"
        # filename = "heatmap-willshawnetwork-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-pca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000x16"
        filename = "heatmap-perfectmemory-zernike-zca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-zernike-zca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-zernike-zca-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-perfectmemory-zernike-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-zernike-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-zernike-scan16-rows100-cols100-ant1-route1-seville2009-omm1000"

    data = np.load(os.path.join(__stat_dir__, "%s.npz" % filename), allow_pickle=True)
    print([k for k in data.keys()])

    route = data["position_out"]
    # PM: R=0.2876, p=4.78e-24
    # WN: R=0.0403, p=1.65e-01
    # IM: R=0.0678, p=1.95e-02

    # PM-zz: R=0.6473, p=6.47e-142
    # WN-zz: R=0.0814, p=5.04e-03
    # IM-zz: R=-0.01468, p=6.15e-01

    # PM-z: R=0.6148, p=2.51e-124
    # WN-z: R=0.0561, p=5.34e-02
    # IM-z: R=0.0508, p=8.04e-02

    # print([key for key in data.keys()])
    novmap = 1 - data["familiarity_map"]
    # z = 5
    # z = np.maximum(np.sum(data["hidden_layer"][-novmap.size:], axis=1), 1).reshape(novmap.shape)
    # out_layer = data["output_layer"]
    # novmap[:] = out_layer[-novmap.size:].reshape(novmap.shape) / z
    # print(out_layer.shape, novmap.shape)
    # novmap = 1 - np.power(1 - novmap, 4096)
    # # novmap = np.power(novmap, .2)  # WN
    # novmap = np.power(novmap, 32)  # IM
    # print(novmap.min(), novmap.max())
    # novmap = 1 / (1 + np.exp(-50 * (novmap - .34)))  # IM
    # print(novmap.min(), novmap.max())
    # novmap = 1 - np.power(1 - novmap, 4096)
    # print(novmap.min(), novmap.max())
    # novmap = 1 - np.power(1 - novmap, 1/2)
    # print(novmap.min(), novmap.max())
    # novmap = 1 / (1 + np.exp(-10 * (novmap - .05)))
    print(novmap.min(), novmap.max())

    # perfect memory
    # heatmap = data["familiarity_map"]
    heatmap = 1 - novmap

    fammap = np.transpose(heatmap, axes=(1, 0, 2))
    fammap = compose_fammap(fammap, method="angles")

    # score, p = get_score(fammap, data["position_out"], sigma=.01, fill_nans=True)
    # print("r=%.4f, p=%.2e" % (score, p))

    plt.figure(filename, figsize=(5, 5))

    x = np.linspace(0, 10, fammap.shape[0], endpoint=True)
    y = np.linspace(0, 10, fammap.shape[1], endpoint=True)
    x, y = np.meshgrid(x, y)

    plt.contourf(x, y, fammap, cmap="Greys")
    plt.plot(route[:, 1], route[:, 0], 'r:')

    plt.show()


def get_score(familiarity_map, training_route, sigma=.1, fill_nans=False, fill_maps=False):
    nb_rows, nb_cols = familiarity_map.shape
    row, col = np.array([index for index in np.ndindex(familiarity_map.shape)]).T
    x = col2x(col, nb_cols=nb_cols, max_meters=10.)
    y = row2y(row, nb_rows=nb_rows, max_meters=10.)

    p = x + 1j * y
    # yaw = ori2yaw(ori, nb_oris=nb_oris, degrees=True)

    x_r, y_r, _, yaw_r = training_route.T
    p_r = x_r + 1j * y_r

    d = np.absolute(p[:, np.newaxis] - p_r[np.newaxis, :]).min(axis=1) / 10.
    d_map = d.reshape(familiarity_map.shape).T
    f_map = familiarity_map
    # p_map = np.exp(-np.square(d_map) / (2 * np.square(sigma)))  # gaussian
    p_map = np.exp(-np.absolute(d_map) / (np.sqrt(2) * sigma))

    i_map = p_map > 1e-02

    r, p = pearsonr(p_map[i_map], f_map[i_map])

    if fill_maps:
        familiarity_map[:] = p_map[:]
    if fill_nans:
        # familiarity_map.reshape((-1, familiarity_map.shape[-1]))[~i_map, :] = np.nan
        familiarity_map[~i_map] /= 2

    return r, p


def compose_fammap(familiarity_map, method="angles"):
    if familiarity_map.ndim < 3:
        return familiarity_map
    elif "angle" in method:
        angles = np.linspace(0, 2 * np.pi, familiarity_map.shape[2], endpoint=False)
        z = np.sin(angles[:familiarity_map.shape[2] // 2]).sum()
        familiarity_map = np.tensordot(familiarity_map, np.exp(-1j * angles), axes=(-1, 0))
        # return 1 / (1 + np.exp(-5 * (np.absolute(familiarity_map) / z - 0.1)))
        print(z)
        return np.absolute(familiarity_map)
    elif method == "mean":
        return familiarity_map.mean(axis=2) * 10
    elif method == "max":
        return familiarity_map.max(axis=2) * 5
    elif method == "min":
        return familiarity_map.min(axis=2)
    elif method == "sum":
        return familiarity_map.sum(axis=2)
    elif isinstance(method, int) and method < familiarity_map.shape[2]:
        return familiarity_map[..., method]
    else:
        return familiarity_map


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
