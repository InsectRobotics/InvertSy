from invertpy.brain.synapses import sparse_synapses

from scipy.special import comb

import numpy as np


if __name__ == '__main__':
    nb_in = 100
    nb_out = 4000
    nb_samples = 1000
    max_samples = 10000

    nb_in_min = int(np.maximum(nb_in / nb_out, 1))
    nb_in_max = nb_in_min
    while comb(nb_in, nb_in_max - 1) < comb(nb_in, nb_in_max) < nb_in * nb_out:
        nb_in_max += 1

    spr = sparse_synapses(nb_in, nb_out, nb_in_min=nb_in_min, nb_in_max=nb_in_max, max_samples=max_samples, verbose=True)

    lam = np.linalg.eigvals(spr.dot(spr.T))
    dim_x = np.square(np.sum(lam)) / np.sum(np.square(lam))
    dim_h_min = nb_in / (1 + nb_in / nb_out + np.square(nb_in_max - 1) / nb_in)
    dim_h_max = nb_in / (1 + nb_in / nb_out + np.square(nb_in_min - 1) / nb_in)

    print(f"dim(x) = {dim_x:.2f}, dim(h) = [{dim_h_min:.2f}, {dim_h_max:.2f}]")
    print(f"average correlation: {nb_in_min / nb_in:.2f}, {nb_in_max / nb_in:.2f}")

    print(f"#synapses - min: {(spr > 0).sum(axis=0).min()}, max: {(spr > 0).sum(axis=0).max()}")

    import matplotlib.pyplot as plt

    plt.figure("w_p2k", figsize=(10, 5))
    plt.imshow(spr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.tight_layout()
    plt.show()
