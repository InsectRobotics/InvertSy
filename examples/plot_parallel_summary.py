from invertsy.sim.simulation import get_statsdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 800)


def main(*args):
    plot = sns.boxplot

    par = 21  # 1
    rot = 1  # 36
    pca = 50  # 500, 900

    df = pd.read_excel(os.path.join(get_statsdir(), "parallel-summary.xlsx"), index_col=0)
    print(df)
    df_mean = df.melt(id_vars=["ant", "route", "world", "model", "ommatidia", "pca", "scans", "dispositions", "r_d", "p_d"],
                      value_vars=["mean_d", "mean_dc"], value_name="mean", var_name="control")
    df_mean.loc[df_mean["control"] == "mean_d", "control"] = False
    df_mean.loc[df_mean["control"] == "mean_dc", "control"] = True
    df_var = df.melt(id_vars=["ant", "route", "world", "model", "ommatidia", "pca", "scans", "dispositions", "r_d", "p_d"],
                     value_vars=["variance_d", "variance_dc"], value_name="variance", var_name="control")
    df_var.loc[df_var["control"] == "variance_d", "control"] = False
    df_var.loc[df_var["control"] == "variance_dc", "control"] = True
    df_skew = df.melt(id_vars=["ant", "route", "world", "model", "ommatidia", "pca", "scans", "dispositions", "r_d", "p_d"],
                      value_vars=["skewness_d", "skewness_dc"], value_name="skewness", var_name="control")
    df_skew.loc[df_skew["control"] == "skewness_d", "control"] = False
    df_skew.loc[df_skew["control"] == "skewness_dc", "control"] = True
    df_kurt = df.melt(id_vars=["ant", "route", "world", "model", "ommatidia", "pca", "scans", "dispositions", "r_d", "p_d"],
                      value_vars=["kurtosis_d", "kurtosis_dc"], value_name="kurtosis", var_name="control")
    df_kurt.loc[df_kurt["control"] == "kurtosis_d", "control"] = False
    df_kurt.loc[df_kurt["control"] == "kurtosis_dc", "control"] = True

    df = pd.concat([df_mean, df_var, df_skew, df_kurt], axis=0)
    print(df)
    # df.loc[:, "mean_d"] = df["mean_d"] - df["mean_dc"]
    # df.loc[:, "variance_d"] = df["variance_d"] - df["variance_dc"]
    # df.loc[:, "skewness_d"] = df["skewness_d"] - df["skewness_dc"]
    # df.loc[:, "kurtosis_d"] = df["kurtosis_d"] - df["kurtosis_dc"]
    # df.loc[:, "mean_r"] = df["mean_r"] - df["mean_rc"]
    # df.loc[:, "variance_r"] = df["variance_r"] - df["variance_rc"]
    # df.loc[:, "skewness_r"] = df["skewness_r"] - df["skewness_rc"]
    # df.loc[:, "kurtosis_r"] = df["kurtosis_r"] - df["kurtosis_rc"]
    df.loc[df["model"] == "perfectmemory", "model"] = "PM"
    df.loc[df["model"] == "visualincentivecircuit", "model"] = "IC"
    df.loc[df["model"] == "willshawnetwork", "model"] = "WN"

    palette = np.array(sns.color_palette("Set3"))[3:]
    hue_order = ["PM", "WN", "IC"]
    plt.figure(f"dispositions-par{par}-rot{rot}-pca{pca}", figsize=(13, 3))
    plt.subplot(1, 5, 1)
    plot(x="model", y="r_d", palette=palette, order=hue_order, data=df[
        (df["dispositions"] == par) & (df["scans"] == rot) & (df["pca"] == pca) & ~df["control"]])
    plt.ylim(-.01, 1.01)

    plt.subplot()

    # plt.subplot(1, 5, 2)
    # plot(x="control", y="mean", hue="model", palette=palette, hue_order=hue_order, data=df[
    #     (df["dispositions"] == par) & (df["scans"] == rot) & (df["pca"] == pca)])
    # plt.ylim(-.01, .01)
    # plt.subplot(1, 5, 3)
    # plot(x="control", y="variance", hue="model", palette=palette, hue_order=hue_order, data=df[
    #     (df["dispositions"] == par) & (df["scans"] == rot) & (df["pca"] == pca)])
    # plt.ylim(-.01, None)
    # plt.subplot(1, 5, 4)
    # plot(x="control", y="skewness", hue="model", palette=palette, hue_order=hue_order, data=df[
    #     (df["dispositions"] == par) & (df["scans"] == rot) & (df["pca"] == pca)])
    # plt.ylim(-.1, .1)
    # plt.subplot(1, 5, 5)
    # plot(x="control", y="kurtosis", hue="model", palette=palette, hue_order=hue_order, data=df[
    #     (df["dispositions"] == par) & (df["scans"] == rot) & (df["pca"] == pca)])
    # plt.ylim(0, None)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
