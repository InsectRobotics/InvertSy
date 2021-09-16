from invertsy.sim._helpers import create_familiarity_map
from invertsy.sim.simulation import __stat_dir__

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import os


def main(*args):
    if len(args) > 1:
        filename = args[1]
    else:
        # filename = "heatmap-perfectmemory-pca-scan8-rows50-cols50-ant1-route1-simpleworld-omm2000"
        filename = "heatmap-perfectmemory-pca-scan8-rows50-cols50-ant1-route1-seville2009-omm2000"

    data = np.load(os.path.join(__stat_dir__, "%s.npz" % filename))

    # for key in data:
    #     print(key, data[key].shape)

    route_length = data["familiarity_out"].shape[0]

    # omm = np.clip(data["ommatidia"].mean(axis=2), 0, 1)
    # omm_out = omm[:route_length]
    # omm_in = omm[route_length:]

    pn = data["PN"][:, 0]
    pn_out = pn[:route_length]
    pn_in = pn[route_length:]

    mbon_in = np.zeros(pn_in.shape[0], dtype=float)
    for i in range(mbon_in.shape[0]):
        err = mean_squared_error(pn_out.T, np.array([pn_in[i]] * pn_out.shape[0]).T,
                                 multioutput='raw_values', squared=False).min()
        mbon_in[i] = err
        if i % 100 == 0:
            print("% 5d: %.2e" % (i, err))
    print(mbon_in.shape)

    fammap = 1. - mbon_in.reshape(data["familiarity_map"].shape, order="C") / mbon_in.max()
    print(fammap.min(), fammap.max())

    plt.figure(filename, figsize=(5, 5))
    fam = create_familiarity_map(nb_rows=fammap.shape[0], nb_cols=fammap.shape[1])
    fam.set_array(np.max(fammap, axis=2))

    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
