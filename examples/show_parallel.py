from invertsy.sim._helpers import create_familiarity_map, col2x, row2y
from invertsy.sim.simulation import get_statsdir

from invertpy.brain.compass import ring2complex

from matplotlib import tri
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

pattern_par = r"heatmap-([a-z]+)-pca([0-9]+)-scan(1)-par([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)-?([0-9]*)"
pattern_rot = r"heatmap-([a-z]+)-pca([0-9]+)-scan([0-9]+)-par(1)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)-?([0-9]*)"


def main(*args):
    if len(args) > 1:
        filenames = [args[1]]
    else:
        filenames = glob(os.path.join(get_statsdir(), "heatmap-*.npz"))

    ant_no, route_no, pca, world, ommatidia, pre_training = 1, 1, 100, "seville2009", 1000, 2

    if pre_training == 0:
        pre_training = 1
    fig = plt.figure(f"sample-ant{ant_no}-route{route_no}", figsize=(10, 3))

    tt, gt, pm, wn, ic = False, False, False, False, False
    for filename in filenames:
        details = re.match(pattern_par, filename.split("\\")[-1].replace(".npz", ""))
        if details is None:
            details = re.match(pattern_rot, filename.split("\\")[-1].replace(".npz", ""))
        if details is None:
            continue
        data = np.load(filename, allow_pickle=True)

        model = details.group(1)
        pca_ = int(details.group(2)) * 10
        scans = int(details.group(3))
        dispositions = int(details.group(4))
        ant = int(details.group(5))
        route = int(details.group(6))
        world_ = details.group(7)
        ommatidia_ = int(details.group(8))
        pre_training_ = int(details.group(9) if details.group(9) != '' else 1)

        if route != route_no:
            continue
        elif ant != ant_no:
            continue
        elif pca != pca_:
            continue
        elif world != world_:
            continue
        elif ommatidia != ommatidia_:
            continue
        elif model == "visualincentivecircuit" and pre_training != pre_training_:
            continue
        elif model == "perfectmemory" and pm:
            continue
        elif model == "willshawnetwork" and wn:
            continue
        elif model == "visualincentivecircuit" and ic:
            continue

        xyz_out = data["xyz_out"]
        xyz = np.roll(data["xyz"], shift=-1, axis=0)
        fam = np.roll(data["familiarity"], shift=-1, axis=0)
        fam_map = np.squeeze(data["familiarity_par"])
        route_length = xyz_out.shape[0]
        # fam = np.power(fam, 8)
        # print(fam.min(), fam.max(), fam.shape)

        print(f"File: {filename}")
        xi, yi = np.meshgrid(np.linspace(-20, 20, 21), 0.1 * np.linspace(0, route_length, route_length, endpoint=False))
        if not tt:
            plt.subplot(151, aspect="auto")
            plt.plot(xyz[:, 1].reshape(-1, route_length).T[:-1, ::3],
                     xyz[:, 0].reshape(-1, route_length).T[:-1, ::3], "grey", lw=.5)
            plt.plot(xyz_out[:, 1], xyz_out[:, 0], "k", lw=2)
            plt.xlim(4.5, 7)
            plt.ylim(0, 10)
            tt = True

        if not gt:

            plt.subplot(152, aspect="auto")
            z = -np.min(np.linalg.norm(xyz[:, None, :2] - xyz_out[None, :, :2], axis=2), axis=1)
            z = (z - z.min()) / (z.max() - z.min())
            sns.kdeplot(xyz[:, 1], xyz[:, 0], weights=z, cmap="Greys", shade=True, bw_adjust=.5)
            # plt.contourf(xi, yi, z, cmap="inferno", levels=11)
            plt.xlim(4.5, 7)
            plt.ylim(0, 10)
            gt = True

        if model == "perfectmemory" and not ic:

            plt.subplot(153, aspect="auto")

            z = fam
            sns.kdeplot(xyz[:, 1], xyz[:, 0], weights=z, cmap="Greys", shade=True, bw_adjust=.5)
            plt.xlim(4.5, 7)
            plt.ylim(0, 10)

            # z = np.hstack([fam_map[:, -2::-2], fam_map[:, 0::2]])
            # # plt.contourf(xi, yi, z, cmap="YlOrBr")
            # plt.contourf(xi, yi, np.power(z, 1), cmap="inferno", levels=11)
            # plt.yticks([0, 20, 40, 60, 80], [""] * 5)
            # plt.xlim(-20, 20)
            # plt.ylim((route_length-1)*.1, 0)
            # plt.xlim(4.5, 7)
            # plt.ylim(0, 10)
            pm = True

        if model == "willshawnetwork" and not wn:

            plt.subplot(154, aspect="auto")
            z = fam
            sns.kdeplot(xyz[:, 1], xyz[:, 0], weights=z, cmap="Greys", shade=True, bw_adjust=.5)
            plt.xlim(4.5, 7)
            plt.ylim(0, 10)

            # z = np.hstack([fam_map[:, 1::2][:, ::-1], fam_map[:, 0::2]])
            # # plt.contourf(xi, yi, z, cmap="YlOrBr")
            # plt.contourf(xi, yi, np.power(z, 1), cmap="inferno", levels=11)
            # plt.yticks([0, 20, 40, 60, 80], [""] * 5)
            # plt.xlim(-20, 20)
            # plt.ylim((route_length-1)*.1, 0)
            wn = True

        if model == "visualincentivecircuit" and not ic:

            plt.subplot(155, aspect="auto")
            z = fam
            sns.kdeplot(xyz[:, 1], xyz[:, 0], weights=z, cmap="Greys", shade=True, bw_adjust=.5)
            plt.xlim(4.5, 7)
            plt.ylim(0, 10)

            # z = np.hstack([fam_map[:, 1::2][:, ::-1], fam_map[:, 0::2]])
            # # plt.contourf(xi, yi, z, cmap="YlOrBr")
            # plt.contourf(xi, yi, np.power(z, 1), cmap="inferno", levels=11)
            # plt.yticks([0, 20, 40, 60, 80], [""] * 5)
            # plt.xlim(-20, 20)
            # plt.ylim((route_length-1)*.1, 0)
            ic = True

    plt.tight_layout()
    plt.show()

    # if fammap.shape[2] > 1 or fammap.shape[1] == 1 and fammap.shape[2] == 1:
    #     nb_rows = 1 + int(fammap.shape[1] > 1)
    #     for i in range(fammap.shape[1]):
    #         plt.subplot(nb_rows, fammap.shape[1], i + 1)
    #         if fammap.shape[1] > 1:
    #             offset = np.linspace(-20, 20, fammap.shape[1], endpoint=True)[i]
    #         else:
    #             offset = 0
    #         plt.title(f"{offset:.0f}cm", fontsize=8)
    #         plt.imshow(np.roll(fammap[:, i], fammap.shape[2] // 2, axis=1), cmap="RdPu", vmin=0, vmax=1, aspect="auto")
    #         plt.xticks([.25 * fammap.shape[2], .5 * fammap.shape[2], .75 * fammap.shape[2]],
    #                    [r"$-90^\circ$", "", r"$+90^\circ$"], fontsize=8)
    #         plt.yticks([0, fammap.shape[0] / 2, fammap.shape[0] - 1],
    #                    ["0.0", f"{0.01 * (fammap.shape[0] - 1) / 2:.1f}", f"{0.01 * (fammap.shape[0] - 1):.1f}"],
    #                    fontsize=8)
    #         if i > 0:
    #             plt.yticks([])
    #         else:
    #             plt.ylabel("position on route (m)", fontsize=8)
    #         if i == fammap.shape[1] // 2:
    #             plt.xlabel("parallel disposition (m)", fontsize=8)
    #     cbar = plt.colorbar()
    #     cbar.ax.tick_params(labelsize=8)
    #
    # if fammap.shape[1] > 1:
    #     nb_rows = 1 + int(fammap.shape[2] > 1)
    #     for i in range(fammap.shape[2]):
    #         j = (i + fammap.shape[2] // 2) % fammap.shape[2]
    #         plt.subplot(nb_rows, fammap.shape[2], j + fammap.shape[2] * (nb_rows - 1) + 1)
    #         plt.title(f"{np.linspace(-180, 180, fammap.shape[2], endpoint=False)[j]:.1f}$^\circ$", fontsize=8)
    #         plt.imshow(fammap[..., i], cmap="RdPu", vmin=0, vmax=1, aspect="auto")
    #         plt.xticks([.25 * fammap.shape[1], .5 * fammap.shape[1], .75 * fammap.shape[1]],
    #                    [-.1, "", .1], fontsize=8)
    #         plt.yticks([0, fammap.shape[0] / 2, fammap.shape[0] - 1],
    #                    ["0.0", f"{0.01 * (fammap.shape[0] - 1) / 2:.1f}", f"{0.01 * (fammap.shape[0] - 1):.1f}"], fontsize=8)
    #         if j > 0:
    #             plt.yticks([])
    #         else:
    #             plt.ylabel("position on route (m)", fontsize=8)
    #         if j == fammap.shape[2] - 1:
    #             cbar = plt.colorbar()
    #             cbar.ax.tick_params(labelsize=8)
    #         if i == 0:
    #             plt.xlabel("rotation on the spot", fontsize=8)
    # fig.tight_layout()
    #
    # fig = plt.figure(filename + "-valley", figsize=figsize)
    # if fammap.shape[2] > 1 or fammap.shape[1] == 1 and fammap.shape[2] == 1:
    #     nb_rows = 1 + int(fammap.shape[1] > 1)
    #     for i in range(fammap.shape[1]):
    #         fam = np.roll(fammap[:, i], fammap.shape[2] // 2, axis=1)
    #         plt.subplot(nb_rows, fammap.shape[1], i + 1)
    #         if fammap.shape[1] > 1:
    #             offset = np.linspace(-20, 20, fammap.shape[1], endpoint=True)[i]
    #         else:
    #             offset = 0
    #         plt.title(f"{offset:.0f}cm", fontsize=8)
    #         plt.plot(fam.T, 'grey', lw=.5, alpha=.2)
    #         plt.plot(np.mean(fam, axis=0), 'k', lw=1)
    #         plt.xticks([.25 * fammap.shape[2], .5 * fammap.shape[2], .75 * fammap.shape[2]],
    #                    [r"$-90^\circ$", "", r"$+90^\circ$"], fontsize=8)
    #         plt.yticks([0, 1], fontsize=8)
    #         plt.xlim([0, fammap.shape[2]-1])
    #         plt.ylim([0, 1])
    #         if i > 0:
    #             plt.yticks([])
    #         else:
    #             plt.ylabel("familiarity", fontsize=8)
    #         if i == fammap.shape[1] // 2:
    #             plt.xlabel("parallel disposition (m)", fontsize=8)
    #
    # if fammap.shape[1] > 1:
    #     nb_rows = 1 + int(fammap.shape[2] > 1)
    #     for i in range(fammap.shape[2]):
    #         j = (i + fammap.shape[2] // 2) % fammap.shape[2]
    #         fam = fammap[..., i]
    #         plt.subplot(nb_rows, fammap.shape[2], j + fammap.shape[2] * (nb_rows - 1) + 1)
    #         plt.title(f"{np.linspace(-180, 180, fammap.shape[2], endpoint=False)[j]:.1f}$^\circ$", fontsize=8)
    #         plt.plot(fam.T, 'grey', lw=.5, alpha=.2)
    #         plt.plot(np.mean(fam, axis=0), 'k', lw=1)
    #         plt.xticks([.25 * fammap.shape[1], .5 * fammap.shape[1], .75 * fammap.shape[1]],
    #                    [-.1, "", .1], fontsize=8)
    #         plt.yticks([0, 1], fontsize=8)
    #         plt.xlim([0, fammap.shape[1] - 1])
    #         plt.ylim([0, 1])
    #         if j > 0:
    #             plt.yticks([])
    #         else:
    #             plt.ylabel("familiarity", fontsize=8)
    #         if i == 0:
    #             plt.xlabel("rotation on the spot", fontsize=8)
    # fig.tight_layout()
    #
    # # plt.imshow(lenmap.T, cmap="RdPu", vmin=0, vmax=1, aspect="auto")
    # # x, y = np.arange(812), np.arange(21)
    # # x, y = np.meshgrid(x, y)
    # # v, u = np.cos(angmap), np.sin(angmap)
    # # plt.quiver(x, y, v, u, pivot='mid', color='k', scale=400)
    # # plt.yticks([0, 5, 10, 15, 20], [-0.2, -0.1, 0, 0.1, 0.2])
    #
    # plt.show()


def compose_fammap(familiarity_map, method="angles"):
    if familiarity_map.ndim < 3:
        return familiarity_map
    elif "angle" in method:
        angles = np.linspace(0, 2 * np.pi, familiarity_map.shape[2], endpoint=False)
        z = np.sin(angles[:familiarity_map.shape[2] // 2]).sum()
        familiarity_map = np.tensordot(familiarity_map, np.exp(-1j * angles), axes=(-1, 0))
        # return 1 / (1 + np.exp(-5 * (np.absolute(familiarity_map) / z - 0.1)))
        print(z)
        return np.absolute(familiarity_map), np.angle(familiarity_map)
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
