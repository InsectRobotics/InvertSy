import matplotlib.pyplot as plt

from invertsy.sim.simulation import get_statsdir

from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns

import re
import os

pattern_par = r"heatmap-([a-z]+)-pca([0-9]+)-scan(1)-par([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)"
pattern_rot = r"heatmap-([a-z]+)-pca([0-9]+)-scan([0-9]+)-par(1)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)"


def main(*args):
    dataset = {
        "ant": [],
        "route": [],
        "world": [],
        "model": [],
        "ommatidia": [],
        "#PCPNs": [],
        "scans": [],
        "dispositions": [],
        "layer": [],
        "correlation": []
    }

    nb_entries = 100

    filenames = glob(os.path.join(get_statsdir(), "heatmap-*.npz"))
    new_entries = False

    for filename in filenames:
        new_filename = filename.replace("heatmap", "correlation")

        details = re.match(pattern_par, filename.split("\\")[-1].replace(".npz", ""))

        if details is None:
            details = re.match(pattern_rot, filename.split("\\")[-1].replace(".npz", ""))
        if details is None:
            continue

        model = details.group(1)
        nb_pcpn = int(details.group(2)) * 10
        nb_scans = int(details.group(3))
        nb_dispositions = int(details.group(4))
        ant_id = int(details.group(5))
        route_id = int(details.group(6))
        world = details.group(7)
        nb_ommatidia = int(details.group(8))

        model_rand = model + "rand"

        if "visualincentivecircuit" not in model or nb_scans != 1:
            continue

        if os.path.exists(new_filename):
            data = np.load(new_filename)
            print(f"\nCorrelations file loaded: {new_filename}")

            c_omm = data["ommatidia"]
            c_pca = data["pcpn"]
            c_kcs = data["kcs"]
            c_rand = data["kc_rand"]
            indexes = np.random.permutation(np.arange(len(c_omm)))[:nb_entries]

            dataset["layer"].extend(["ommatidia"] * nb_entries)
            dataset["layer"].extend(["PCPNs"] * nb_entries)
            dataset["layer"].extend(["KCs"] * nb_entries)
            dataset["layer"].extend(["KCs (rand)"] * nb_entries)
            dataset["correlation"].extend(c_omm[indexes])
            dataset["correlation"].extend(c_pca[indexes])
            dataset["correlation"].extend(c_kcs[indexes])
            dataset["correlation"].extend(c_rand[indexes])
        else:

            filename_rand = filename.replace(model, model_rand)
            if not os.path.exists(filename_rand):
                continue

            print(f"\nFile loading: {filename}")
            data = np.load(filename, allow_pickle=True)
            data_rand = np.load(filename_rand, allow_pickle=True)

            nb_out = data["xyz_out"].shape[0]
            r_omm = np.clip(data["ommatidia"][:nb_out].mean(axis=-1), 0, 1).squeeze()
            r_pca = data["input_layer"][:nb_out].squeeze()
            r_kcs = data["hidden_layer"][:nb_out].squeeze()
            r_kcs_rand = data_rand["hidden_layer"][:nb_out].squeeze()

            # print(pd.DataFrame(r_omm).T.describe())
            c_omm = pd.DataFrame(r_omm).T.corr().to_numpy()
            # c_omm = pd.DataFrame(r_omm).T.corr(method='spearman')
            # c_omm = pd.DataFrame(r_omm).T.corr(method='kendall')
            c_omm[np.tril_indices(c_omm.shape[0])] = np.nan
            print(f"c_omm = {np.nanmean(c_omm)}, ", end="")
            c_pca = pd.DataFrame(r_pca).T.corr().to_numpy()
            c_pca[np.tril_indices(c_pca.shape[0])] = np.nan
            print(f"c_pca = {np.nanmean(c_pca)}, ", end="")
            c_kcs = pd.DataFrame(r_kcs).T.corr().to_numpy()
            c_kcs[np.tril_indices(c_kcs.shape[0])] = np.nan
            print(f"c_kcs = {np.nanmean(c_kcs)}, ", end="")
            c_rand = pd.DataFrame(r_kcs_rand).T.corr().to_numpy()
            c_rand[np.tril_indices(c_rand.shape[0])] = np.nan
            print(f"c_kcs (rand) = {np.nanmean(c_rand)}, ", end="")
            c_dif = c_kcs - c_omm
            c_dif_rand = c_rand - c_omm
            print(f"c_dif = {np.nanmean(c_dif)}, rand = {np.nanmean(c_dif_rand)}")

            c_omm = c_omm[~np.isnan(c_dif)]
            c_pca = c_pca[~np.isnan(c_dif)]
            c_kcs = c_kcs[~np.isnan(c_dif)]
            c_rand = c_rand[~np.isnan(c_dif)]

            indexes = np.random.permutation(np.arange(len(c_omm)))[:nb_entries]

            dataset["layer"].extend(["ommatidia"] * nb_entries)
            dataset["layer"].extend(["PCPNs"] * nb_entries)
            dataset["layer"].extend(["KCs"] * nb_entries)
            dataset["layer"].extend(["KCs (rand)"] * nb_entries)
            dataset["correlation"].extend(c_omm[indexes])
            dataset["correlation"].extend(c_pca[indexes])
            dataset["correlation"].extend(c_kcs[indexes])
            dataset["correlation"].extend(c_rand[indexes])

            np.savez(new_filename, ommatidia=c_omm, pcpn=c_pca, kcs=c_kcs, kc_rand=c_rand)
            new_entries = True

        dataset["model"].extend([model] * nb_entries * 4)
        dataset["#PCPNs"].extend([nb_pcpn] * nb_entries * 4)
        dataset["scans"].extend([nb_scans] * nb_entries * 4)
        dataset["dispositions"].extend([nb_dispositions] * nb_entries * 4)
        dataset["ant"].extend([ant_id] * nb_entries * 4)
        dataset["route"].extend([route_id] * nb_entries * 4)
        dataset["world"].extend([world] * nb_entries * 4)
        dataset["ommatidia"].extend([nb_ommatidia] * nb_entries * 4)

    df = pd.DataFrame(dataset)
    # print(df)
    if new_entries:
        df.to_excel(os.path.join(get_statsdir(), "pca-whitening-results.xlsx"))

    df_stats = df.groupby(["#PCPNs", "layer"]).describe()
    medians = df_stats[("correlation", "50%")].unstack(level=-1)
    quantile1 = df_stats[("correlation", "25%")].unstack(level=-1)
    quantile3 = df_stats[("correlation", "75%")].unstack(level=-1)
    print(medians)

    # print(sns.color_palette("Set3"))
    palette = np.array(sns.color_palette("colorblind"))[[0, 1, 6, 2]]
    hue_order = ["ommatidia", "PCPNs", "KCs (rand)", "KCs"]
    plt.figure("decorrelation", figsize=(5, 2))
    for c, level in zip(palette[:4], hue_order):
        plt.fill_between(medians.index, quantile1[level], quantile3[level], alpha=0.2, facecolor=c)
        # sns.lineplot(medians.index, med, color=c)
    sns.lineplot(x="#PCPNs", y="correlation", hue="layer", style="layer", palette=palette, hue_order=hue_order,
                 data=df, estimator=np.median, markers=True, dashes=False, ci=None)
    # plt.plot([-1, 10], [0, 0], "grey", alpha=.5, lw=2, zorder=0)
    # sns.boxplot(x="#PCPNs", y="correlation", hue="layer", palette=palette, hue_order=hue_order, data=df)

    plt.yticks([0, .5])
    plt.ylim([-0.1, 0.6])
    plt.xticks([10, 50, 100, 200, 300, 400, 500])
    plt.xlim([10, 500])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
