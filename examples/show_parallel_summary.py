from show_parallel import compose_fammap

from invertsy.sim._helpers import create_familiarity_map, col2x, row2y
from invertsy.sim.simulation import __stat_dir__

from invertpy.brain.compass import ring2complex

from scipy.stats.stats import pearsonr
from scipy.special import expit

import matplotlib.pyplot as plt
import numpy as np
import os


def main(*args):
    if len(args) > 1:
        filename = args[1]
    else:
        # filename = "heatmap-perfectmemory-pca-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-perfectmemory-pca01-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-perfectmemory-pca01-li-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-pca-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-pca01-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-infomax-pca01-li-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-pca005-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-pca005-scan180-par1-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-pca050-li-scan180-par1-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-pca-li-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-zernike-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-willshawnetwork-zernike-li-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-incentivecircuitmemory-pca080-scan16-par21-ant1-route1-seville2009-omm1000"
        filename = "heatmap-visualincentivecircuit-pca080-scan16-par21-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-incentivecircuitmemory-pca010-scan16-par21-rand-ant1-route1-seville2009-omm1000"
        # filename = "heatmap-incentivecircuitmemory-pca010-li-scan16-par21-rand-ant1-route1-seville2009-omm1000"

    data = np.load(os.path.join(__stat_dir__, "%s.npz" % filename), allow_pickle=True)
    print([k for k in data.keys()])

    show_i = 2
    show_j = 0

    fammap = data["familiarity_par"]
    fammap = np.hstack([fammap[:, -2::-2], fammap[:, ::2]])
    print(fammap.min(), fammap.max(), fammap.shape)
    # lenmap, angmap = compose_fammap(fammap, method="angles")
    # fammap = expit(40 * (fammap - 0.85))
    # fammap = expit(10 * (fammap - .7))
    fammap = np.power(fammap, 8)
    print(fammap.min(), fammap.max(), fammap.shape)

    figsize = (4, 4)

    fig = plt.figure(filename, figsize=figsize)
    if fammap.shape[2] > 1 or fammap.shape[1] == 1 and fammap.shape[2] == 1:
        for i in range(fammap.shape[1]):
            if i != fammap.shape[1] // show_i:
                continue
            plt.subplot(2, 2, 1)
            if fammap.shape[1] > 1:
                offset = np.linspace(-20, 20, fammap.shape[1], endpoint=True)[i]
            else:
                offset = 0
            plt.title(f"{offset:.0f}cm", fontsize=8)
            plt.imshow(np.roll(fammap[:, i], fammap.shape[2] // 2, axis=1), cmap="RdPu", vmin=0, vmax=1, aspect="auto")
            plt.xticks([.25 * fammap.shape[2], .5 * fammap.shape[2], .75 * fammap.shape[2]],
                       [r"$-90^\circ$", "", r"$+90^\circ$"], fontsize=8)
            plt.yticks([0, fammap.shape[0] / 2, fammap.shape[0] - 1],
                       ["0.0", f"{0.01 * (fammap.shape[0] - 1) / 2:.1f}", f"{0.01 * (fammap.shape[0] - 1):.1f}"],
                       fontsize=8)
            plt.ylabel("position on route (m)", fontsize=8)


    if fammap.shape[1] > 1:
        for i in range(fammap.shape[2]):
            if i != show_j:
                continue

            j = (i + fammap.shape[2] // 2) % fammap.shape[2]
            plt.subplot(2, 2, 2)
            plt.title(f"{np.linspace(-180, 180, fammap.shape[2], endpoint=False)[j]:.1f}$^\circ$", fontsize=8)
            plt.imshow(fammap[..., i], cmap="RdPu", vmin=0, vmax=1, aspect="auto")
            plt.xticks([.25 * fammap.shape[1], .5 * fammap.shape[1], .75 * fammap.shape[1]],
                       [-.1, "", .1], fontsize=8)
            plt.yticks([0, fammap.shape[0] / 2, fammap.shape[0] - 1],
                       ["0.0", f"{0.01 * (fammap.shape[0] - 1) / 2:.1f}", f"{0.01 * (fammap.shape[0] - 1):.1f}"], fontsize=8)

            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=8)

    if fammap.shape[2] > 1 or fammap.shape[1] == 1 and fammap.shape[2] == 1:
        for i in range(fammap.shape[1]):
            if i != fammap.shape[1] // show_i:
                continue

            fam = np.roll(fammap[:, i], fammap.shape[2] // 2, axis=1)
            plt.subplot(2, 2, 3)
            if fammap.shape[1] > 1:
                offset = np.linspace(-20, 20, fammap.shape[1], endpoint=True)[i]
            else:
                offset = 0
            plt.title(f"{offset:.0f}cm", fontsize=8)
            plt.plot(fam.T, 'grey', lw=.5, alpha=.2)
            plt.plot(np.mean(fam, axis=0), 'k', lw=1)
            plt.xticks([.25 * fammap.shape[2], .5 * fammap.shape[2], .75 * fammap.shape[2]],
                       [r"$-90^\circ$", "", r"$+90^\circ$"], fontsize=8)
            plt.yticks([0, 1], fontsize=8)
            plt.xlim([0, fammap.shape[2]-1])
            plt.ylim([0, 1])
            plt.ylabel("familiarity", fontsize=8)
            plt.xlabel("rotation on the spot", fontsize=8)

    if fammap.shape[1] > 1:
        for i in range(fammap.shape[2]):

            if i != show_j:
                continue

            j = (i + fammap.shape[2] // 2) % fammap.shape[2]
            fam = fammap[..., i]
            plt.subplot(2, 2, 4)
            plt.title(f"{np.linspace(-180, 180, fammap.shape[2], endpoint=False)[j]:.1f}$^\circ$", fontsize=8)
            plt.plot(fam.T, 'grey', lw=.5, alpha=.2)
            plt.plot(np.mean(fam, axis=0), 'k', lw=1)
            plt.xticks([.25 * fammap.shape[1], .5 * fammap.shape[1], .75 * fammap.shape[1]],
                       [-.1, "", .1], fontsize=8)
            plt.yticks([0, 1], fontsize=8)
            plt.xlim([0, fammap.shape[1] - 1])
            plt.ylim([0, 1])
            plt.xlabel("parallel disposition (m)", fontsize=8)
    fig.tight_layout()

    # plt.imshow(lenmap.T, cmap="RdPu", vmin=0, vmax=1, aspect="auto")
    # x, y = np.arange(812), np.arange(21)
    # x, y = np.meshgrid(x, y)
    # v, u = np.cos(angmap), np.sin(angmap)
    # plt.quiver(x, y, v, u, pivot='mid', color='k', scale=400)
    # plt.yticks([0, 5, 10, 15, 20], [-0.2, -0.1, 0, 0.1, 0.2])

    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
