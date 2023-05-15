from invertsy.sim.simulation import get_statsdir

from scipy.optimize import curve_fit
from scipy.stats import circmean, pearsonr, linregress, ttest_ind, page_trend_test, kendalltau, spearmanr
from glob import glob
from copy import copy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re

pattern_par = r"heatmap-([a-z]+)-pca([0-9]+)-?(li)?-scan(1)-par([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)-?([0-9]*)"
pattern_rot = r"heatmap-([a-z]+)-pca([0-9]+)-?(li)?-scan([0-9]+)-par(1)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)-?([0-9]*)"
pattern_zer = r"heatmap-([a-z]+)-zernike-scan([0-9]+)-par([0-9]+)-ant([0-9]+)-route([0-9]+)-([a-zA-Z0-9]+)-omm([0-9]+)-?([0-9]*)"


def main(*args):

    show_glance = True
    show_infomax = True
    dataset = {
        "ant": [],
        "route": [],
        "world": [],
        "model": [],
        "ommatidia": [],
        "pca": [],
        "li": [],
        "zernike": [],
        "scans": [],
        "dispositions": [],
        "pre_training": [],
        "r_r": [],
        "r_d": [],
        "p_r": [],
        "p_d": [],
        "mean_d": [],
        "variance_d": [],
        "skewness_d": [],
        "kurtosis_d": [],
        "mean_dc": [],
        "variance_dc": [],
        "skewness_dc": [],
        "kurtosis_dc": [],
        "mean_r": [],
        "variance_r": [],
        "skewness_r": [],
        "kurtosis_r": [],
        "mean_rc": [],
        "variance_rc": [],
        "skewness_rc": [],
        "kurtosis_rc": []
    }

    page_bins = np.linspace(0, 8, 16)
    filenames = glob(os.path.join(get_statsdir(), "heatmap-*.npz"))

    cc_dis, cc_rot = {"model": [], "correlation": []}, {"model": [], "correlation": []}
    dis, rot = {"PM": [], "WN": [], "IC": [], "IM": []}, {"PM": [], "WN": [], "IC": [], "IM": []}
    cert = {"PM": [], "WN": [], "IC": [], "IM": []}

    model_name = {
        "perfectmemory": "PM",
        "willshawnetwork": "WN",
        "infomax": "IM",
        "visualincentivecircuit": "IC"
    }

    show_config = {
        "pca": 50,
        "world": "seville2009",
        "li": False,
        "zernike": False
    }
    pre_training = 5

    for filename in filenames:

        zernike = False
        details = re.match(pattern_par, filename.split("\\")[-1].replace(".npz", ""))
        if details is None:
            details = re.match(pattern_rot, filename.split("\\")[-1].replace(".npz", ""))
        if details is None:
            details = re.match(pattern_zer, filename.split("\\")[-1].replace(".npz", ""))
            zernike = True
        if details is None:
            continue
        if details.group(1) not in model_name:
            continue
        if int(details.group(7 - int(zernike) * 2)) > 1:
            continue

        data = np.load(filename, allow_pickle=True)

        # print(f"\nFile: {filename}")
        # print([k for k in data.keys()])

        dataset["model"].append(model_name[details.group(1)])
        dataset["pca"].append(int(details.group(2)) * 10 if not zernike else 0)
        dataset["li"].append(details.group(3) == "li" if not zernike else False)
        dataset["zernike"].append(zernike)
        dataset["scans"].append(int(details.group(4 - int(zernike) * 2)))
        dataset["dispositions"].append(int(details.group(5 - int(zernike) * 2)))
        dataset["ant"].append(int(details.group(6 - int(zernike) * 2)))
        dataset["route"].append(int(details.group(7 - int(zernike) * 2)))
        dataset["world"].append(details.group(8 - int(zernike) * 2))
        dataset["ommatidia"].append(int(details.group(9 - int(zernike) * 2)))
        dataset["pre_training"].append(int(details.group(10 - int(zernike) * 2)
                                           if details.group(10 - int(zernike) * 2) != '' else 1))

        show_this = np.all([dataset[key][-1] == value for key, value in show_config.items()])
        show_this = show_this and ((pre_training == dataset["pre_training"][-1] and
                                    dataset["model"][-1] == "IC" and
                                    dataset["scans"][-1] == 1) or
                                   dataset["scans"][-1] > 1 or
                                   dataset["model"][-1] != "IC")

        fammap = data["familiarity_par"]
        fammap = np.hstack([fammap[:, -2::-2], fammap[:, ::2]])
        z = np.maximum(fammap.max() - fammap.min(), np.finfo(float).eps)
        fammap = (fammap - fammap.min()) / z
        # fammap = fammap ** 8
        # lenmap, angmap = compose_fammap(fammap, method="angles")
        # fammap = expit(40 * (fammap - 0.85))
        # fammap = expit(10 * (fammap - .7))

        show_i = 2
        show_j = 0

        # the reference distribution for the parallel disposition
        parmap = np.zeros((fammap.shape[0], fammap.shape[1]), dtype=float)
        parmap += 1 - np.abs(np.linspace(-1, 1, parmap.shape[1]))

        # the distribution of the parallel disposition
        parpre = fammap[..., show_j]
        # z = np.maximum(parpre.max(axis=1) - parpre.min(axis=1), np.finfo(float).eps)
        # parpre = (parpre - parpre.min(axis=1)[..., np.newaxis]) / z[..., np.newaxis]
        parpre[~np.isfinite(parpre)] = 0
        d_route = np.array([np.linspace(-0.2, 0.2, parmap.shape[1])] * parmap.shape[0])

        r_par, p_par = pearsonr(parmap.flatten(), parpre.flatten())
        dataset["r_d"].append(r_par)
        dataset["p_d"].append(p_par)

        if show_this and dataset["dispositions"][-1] > 1:
            model = dataset["model"][-1]
            dis[model].append(np.mean(parpre, axis=0))
            cc_dis["model"].append(model)
            cc_dis["correlation"].append(r_par)

        # the reference distribution for the rotation on the spot
        rotmap = np.zeros((fammap.shape[0], fammap.shape[2] + 1), dtype=float)
        rotmap += np.abs(np.linspace(-1, 1, rotmap.shape[1], endpoint=True))

        # the distribution of the rotation on the spot
        rotpre = fammap[:, fammap.shape[1] // show_i]
        # z = np.maximum(rotpre.max(axis=1) - rotpre.min(axis=1), np.finfo(float).eps)
        # rotpre = (rotpre - rotpre.min(axis=1)[..., np.newaxis]) / z[..., np.newaxis]
        rotpre = np.c_[rotpre, rotpre[:, :1]]
        r_route = np.array([np.linspace(0, 360, rotmap.shape[1], endpoint=True)] * rotmap.shape[0])

        r_rot, p_rot = pearsonr(rotmap.flatten(), rotpre.flatten())
        dataset["r_r"].append(r_rot)
        dataset["p_r"].append(p_rot)

        if show_this and dataset["scans"][-1] > 1:
            model = dataset["model"][-1]
            rot[model].append(circmean(rotpre, axis=0))
            cc_rot["model"].append(model)
            cc_rot["correlation"].append(r_rot)

        # print(f"Parallel: r={r_par:.4f}, p={p_par:02.0e}")
        # print(f"Rotation: r={r_rot:.4f}, p={p_rot:02.0e}")

        # calculate weighted mean, variance and kurtosis
        z_par = np.sum(parpre)
        mu_par = np.sum(d_route * parpre) / z_par
        sigma_par = np.sqrt(np.sum(parpre * np.square(d_route - mu_par)) / z_par)
        gamma_par = np.sum(parpre * np.power((d_route - mu_par) / sigma_par, 3)) / z_par
        kapa_par = np.sum(parpre * np.power((d_route - mu_par) / sigma_par, 4)) / z_par
        dataset["mean_d"].append(mu_par)
        dataset["variance_d"].append(sigma_par)
        dataset["skewness_d"].append(gamma_par)
        dataset["kurtosis_d"].append(kapa_par)

        if show_this and dataset["dispositions"][-1] > 1:
            d_parpre = np.diff(parpre, axis=0)
            z_par_0 = np.sum(parpre, axis=1)
            mu_par_0 = np.sum(d_route * parpre, axis=1) / z_par_0
            sigma_par_0 = np.sqrt(np.sum(parpre * np.square(d_route - mu_par_0[:, None]), axis=1) / z_par_0)
            gamma_par_0 = np.sum(parpre * np.power((d_route - mu_par_0[:, None]) / sigma_par_0[:, None], 3), axis=1) / z_par_0
            kapa_par_0 = np.sum(parpre * np.power((d_route - mu_par_0[:, None]) / sigma_par_0[:, None], 4), axis=1) / z_par_0
            model = dataset["model"][-1]
            cert[model].append(sigma_par_0)

        # calculate the weighted mean, variance, skewness and kurtosis
        z_rot = np.sum(rotpre)
        mu_rot = np.sum(r_route * rotpre) / z_rot
        sigma_rot = np.sqrt(np.sum(rotpre * np.square(r_route - mu_rot)) / z_rot)
        gamma_rot = np.sum(rotpre * np.power((r_route - mu_rot) / sigma_rot, 3)) / z_rot
        kapa_rot = np.sum(rotpre * np.power((r_route - mu_rot) / sigma_rot, 4)) / z_rot
        dataset["mean_r"].append(mu_rot)
        dataset["variance_r"].append(sigma_rot)
        dataset["skewness_r"].append(gamma_rot)
        dataset["kurtosis_r"].append(kapa_rot)

        # calculate the weighted mean, variance, skewness and kurtosis
        mu_parref = np.sum(d_route * parmap) / z_par
        sigma_parref = np.sqrt(np.sum(parmap * np.square(d_route - mu_parref)) / z_par)
        gamma_parref = np.sum(parmap * np.power((d_route - mu_parref) / sigma_parref, 3)) / z_par
        kapa_parref = np.sum(parmap * np.power((d_route - mu_parref) / sigma_parref, 4)) / z_par
        dataset["mean_dc"].append(mu_parref)
        dataset["variance_dc"].append(sigma_parref)
        dataset["skewness_dc"].append(gamma_parref)
        dataset["kurtosis_dc"].append(kapa_parref)

        # calculate the weighted mean, variance, skewness and kurtosis
        mu_rotref = np.sum(r_route * rotmap) / z_rot
        sigma_rotref = np.sqrt(np.sum(rotmap * np.square(r_route - mu_rotref)) / z_rot)
        gamma_rotref = np.sum(rotmap * np.power((r_route - mu_rotref) / sigma_rotref, 3)) / z_rot
        kapa_rotref = np.sum(rotmap * np.power((r_route - mu_rotref) / sigma_rotref, 4)) / z_rot
        dataset["mean_rc"].append(mu_rotref)
        dataset["variance_rc"].append(sigma_rotref)
        dataset["skewness_rc"].append(gamma_rotref)
        dataset["kurtosis_rc"].append(kapa_rotref)

    df = pd.DataFrame(dataset)
    df.to_excel(os.path.join(get_statsdir(), "parallel-summary.xlsx"))

    palette = np.array(sns.color_palette("Set3"))[3:]
    hue_order = ["PM", "WN", "IC", "IM"]

    dis_show = pd.DataFrame(cc_dis)
    rot_show = pd.DataFrame(cc_rot)

    plt.figure(f"summary-{'-'.join([f'{key}{value}' for key, value in show_config.items()])}",
               figsize=(10, 4))

    plt.subplot(251)
    sns.boxplot(x="model", y="correlation", palette=palette, order=hue_order, data=dis_show)
    plt.ylim(-.01, 1.01)
    plt.xlabel("")
    plt.title("Pearson CC")

    plt.subplot(256)
    sns.boxplot(x="model", y="correlation", palette=palette, order=hue_order, data=rot_show)
    plt.ylim(-.01, 1.01)

    x_dis = np.linspace(-0.2, 0.2, 21)
    x_rot = np.linspace(0, 2 * np.pi, 37)

    plt.subplot(252)
    dis_pm = np.array(dis["PM"]).T
    dis_pm = (dis_pm - dis_pm.min()) / (dis_pm.max() - dis_pm.min())
    plt.plot(x_dis, dis_pm, 'grey', lw=0.5)
    plt.plot(x_dis, np.median(dis_pm, axis=1), 'k', lw=2)
    plt.ylim(-.01, 1.01)
    plt.xlim(-.2, 0.2)
    plt.title("PM")

    if show_glance:
        ax = plt.subplot(257, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        rot_pm = np.array(rot["PM"]).T
        rot_pm = (rot_pm - rot_pm.min()) / (rot_pm.max() - rot_pm.min())
        plt.plot(x_rot, rot_pm, 'grey', lw=0.5)
        plt.plot(x_rot, np.median(rot_pm, axis=1), 'k', lw=2)
        plt.ylim(-.01, 1.01)
        plt.yticks([])
        plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [""] * 8)

    plt.subplot(253)
    dis_wn = np.array(dis["WN"]).T
    dis_wn = (dis_wn - dis_wn.min()) / (dis_wn.max() - dis_wn.min())
    plt.plot(x_dis, dis_wn, 'grey', lw=0.5)
    plt.plot(x_dis, np.median(dis_wn, axis=1), 'k', lw=2)
    plt.ylim(-.01, 1.01)
    plt.xlim(-.2, 0.2)
    plt.title("WN")

    if show_glance:
        ax = plt.subplot(258, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        rot_wn = np.array(rot["WN"]).T
        rot_wn = (rot_wn - rot_wn.min()) / (rot_wn.max() - rot_wn.min())
        plt.plot(x_rot, rot_wn, 'grey', lw=0.5)
        plt.plot(x_rot, np.median(rot_wn, axis=1), 'k', lw=2)
        plt.ylim(-.01, 1.01)
        plt.yticks([])
        plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [""] * 8)

    plt.subplot(254)
    dis_ic = np.array(dis["IC"]).T
    dis_ic = (dis_ic - dis_ic.min()) / (dis_ic.max() - dis_ic.min())
    print("Displacements", dis_ic.shape)
    plt.plot(x_dis, dis_ic, 'grey', lw=0.5)
    plt.plot(x_dis, np.median(dis_ic, axis=1), 'k', lw=2)
    plt.ylim(-.01, 1.01)
    plt.xlim(-.2, 0.2)
    plt.title("IC")

    if show_glance:
        ax = plt.subplot(259, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        rot_ic = np.array(rot["IC"]).T
        rot_ic = (rot_ic - rot_ic.min()) / (rot_ic.max() - rot_ic.min())
        print("Glancing", rot_ic.shape)
        plt.plot(x_rot, rot_ic, 'grey', lw=0.5)
        plt.plot(x_rot, np.median(rot_ic, axis=1), 'k', lw=2)
        plt.ylim(-.01, 1.01)
        plt.yticks([])
        plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [""] * 8)

    if show_infomax:
        plt.subplot(255)
        dis_im = np.array(dis["IM"]).T
        dis_im = (dis_im - dis_im.min()) / (dis_im.max() - dis_im.min())
        plt.plot(x_dis, dis_im, 'grey', lw=0.5)
        plt.plot(x_dis, np.median(dis_im, axis=1), 'k', lw=2)
        plt.ylim(-.01, 1.01)
        plt.xlim(-.2, 0.2)
        plt.title("IM")

        if show_glance:
            ax = plt.subplot(2, 5, 10, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            rot_im = np.array(rot["IM"]).T
            rot_im = (rot_im - rot_im.min()) / (rot_im.max() - rot_im.min())
            plt.plot(x_rot, rot_im, 'grey', lw=0.5)
            plt.plot(x_rot, np.median(rot_im, axis=1), 'k', lw=2)
            plt.ylim(-.01, 1.01)
            plt.yticks([])
            plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [""] * 8)

    plt.tight_layout()

    plt.figure(f"certainty-{'-'.join([f'{key}{value}' for key, value in show_config.items()])}",
               figsize=(8, 4))

    dx = 1e-02
    plt.subplot(131)
    max_x = 8.
    max_y = 1.0  # 2.5
    min_y = 0.0  # 1.5
    power = 1.
    intervals = 10

    def kurt2cert(k):
        # return 5 * (.4 / k - 3 - 0.25)
        return .4 / k - 3

    plt.subplot(241)
    x_cert = []
    bins_pm = []
    for i, cert_pm in enumerate(cert["PM"]):
        nb_samples = len(cert_pm)
        s_needed = intervals - nb_samples % intervals
        cert["PM"][i] = np.r_[np.full(s_needed, np.nan), cert_pm]
        cert_x = np.linspace(0, len(cert["PM"][i]) * dx, len(cert["PM"][i]))
        x_cert.append(cert_x)

        x_digit = np.digitize(cert_x[-cert_pm.shape[0]:], np.linspace(0, 8, 16))
        bin_median = [np.nanmedian(cert_pm[x_digit == j]) for j in range(1, len(page_bins))]
        bins_pm.append(bin_median)
    bins_pm = kurt2cert(np.array(bins_pm))
    cert_pm = kurt2cert(np.concatenate(cert["PM"]))
    cert_x = np.concatenate(x_cert)
    mask_pm = np.isfinite(cert_x) & np.isfinite(cert_pm)
    result_pm = linregress(cert_x[mask_pm], cert_pm[mask_pm])
    print("PM", result_pm, len(cert_x[mask_pm]))
    plt.plot(np.nanmedian(cert_x.reshape((-1, intervals)), axis=1),
             np.nanmedian(cert_pm.reshape((-1, intervals)), axis=1), 'grey', ls='', marker='.', markersize=.5)
    plt.plot(cert_x, result_pm.intercept + result_pm.slope * cert_x, 'k', lw=2)
    plt.ylim(min_y, max_y)
    plt.xlim(0, max_x)
    plt.title("PM")

    plt.subplot(245)
    sns.kdeplot(cert_x, cert_pm, cmap="Reds", shade=True, bw_adjust=.5)
    # plt.scatter(cert_x, cert_ic, facecolor='k', marker='.', s=8, alpha=.1)
    plt.plot(cert_x, result_pm.intercept + result_pm.slope * cert_x, 'k', lw=2)
    plt.ylim(.25, .45)
    # plt.ylim(0, 1)
    plt.xlim(0, max_x)

    page_pm = page_trend_test(bins_pm)
    print("PM", page_pm)

    tau_pm, p_kend_pm = kendalltau(cert_x, cert_pm, nan_policy='omit', alternative='greater')
    print(f"PM Kendall tau = {tau_pm:.4f}, p-value = {p_kend_pm:.2e}")

    spr_pm, p_spr_pm = spearmanr(cert_x, cert_pm, nan_policy='omit', alternative='greater')
    print(f"PM Spearman r = {spr_pm:.4f}, p-value = {p_spr_pm:.2e}")

    print()

    plt.subplot(242)
    x_cert = []
    bins_wn = []
    for i, cert_wn in enumerate(cert["WN"]):
        nb_samples = len(cert_wn)
        s_needed = intervals - nb_samples % intervals
        cert["WN"][i] = np.r_[np.full(s_needed, np.nan), cert_wn]
        cert_x = np.linspace(0, len(cert["WN"][i]) * dx, len(cert["WN"][i]))
        x_cert.append(cert_x)

        x_digit = np.digitize(cert_x[-cert_wn.shape[0]:], np.linspace(0, 8, 16))
        bin_median = [np.nanmedian(cert_wn[x_digit == j]) for j in range(1, len(page_bins))]
        bins_wn.append(bin_median)
    bins_wn = kurt2cert(np.array(bins_wn))
    cert_wn = kurt2cert(np.concatenate(cert["WN"]))
    cert_x = np.concatenate(x_cert)
    mask_wn = np.isfinite(cert_x) & np.isfinite(cert_wn)
    result_wn = linregress(cert_x[mask_wn], cert_wn[mask_wn])
    print("WN", result_wn, len(cert_x[mask_wn]))
    plt.plot(np.nanmedian(cert_x.reshape((-1, intervals)), axis=1),
             np.nanmedian(cert_wn.reshape((-1, intervals)), axis=1), 'grey', ls='', marker='.', markersize=.5)
    plt.plot(cert_x, result_wn.intercept + result_wn.slope * cert_x, 'k', lw=2)
    plt.ylim(min_y, max_y)
    plt.xlim(0, max_x)
    plt.title("WN")

    plt.subplot(246)
    sns.kdeplot(cert_x, cert_wn, cmap="Reds", shade=True, bw_adjust=.5)
    # plt.scatter(cert_x, cert_ic, facecolor='k', marker='.', s=8, alpha=.1)
    plt.plot(cert_x, result_wn.intercept + result_wn.slope * cert_x, 'k', lw=2)
    plt.ylim(.25, .45)
    # plt.ylim(0, 1)
    plt.xlim(0, max_x)

    page_wn = page_trend_test(bins_wn)
    print("WN", page_wn)

    tau_wn, p_kend_wn = kendalltau(cert_x, cert_wn, nan_policy='omit', alternative='greater')
    print(f"WN Kendall tau = {tau_wn:.4f}, p-value = {p_kend_wn:.2e}")

    spr_wn, p_spr_wn = spearmanr(cert_x, cert_wn, nan_policy='omit', alternative='greater')
    print(f"WN Spearman r = {spr_wn:.4f}, p-value = {p_spr_wn:.2e}")

    t_wn, p_wn = ttest_ind(cert_pm[mask_pm], cert_wn[mask_wn])
    print(f"WN t-test: t-statistic = {t_wn:.4f}, p-value = {p_wn:.2e}")
    print()

    plt.subplot(243)
    x_cert = []
    bins_ic = []
    for i, cert_ic in enumerate(cert["IC"]):
        nb_samples = len(cert_ic)
        s_needed = intervals - nb_samples % intervals
        cert["IC"][i] = np.r_[np.full(s_needed, np.nan), cert_ic]
        cert_x = np.linspace(0, len(cert["IC"][i]) * dx, len(cert["IC"][i]))
        x_cert.append(cert_x)

        x_digit = np.digitize(cert_x[-cert_ic.shape[0]:], np.linspace(0, 8, 16))
        bin_median = [np.nanmedian(cert_ic[x_digit == j]) for j in range(1, len(page_bins))]
        bins_ic.append(bin_median)
    bins_ic = kurt2cert(np.array(bins_ic))
    cert_ic = kurt2cert(np.concatenate(cert["IC"]))
    cert_x = np.concatenate(x_cert)
    mask_ic = np.isfinite(cert_x) & np.isfinite(cert_ic)
    result_ic = linregress(cert_x[mask_ic], cert_ic[mask_ic])
    print("IC", result_ic, len(cert_x[mask_ic]))
    plt.plot(np.nanmedian(cert_x.reshape((-1, intervals)), axis=1),
             np.nanmedian(cert_ic.reshape((-1, intervals)), axis=1), 'grey', ls='', marker='.', markersize=.5)
    plt.plot(cert_x, result_ic.intercept + result_ic.slope * cert_x, 'k', lw=2)
    plt.ylim(min_y, max_y)
    plt.xlim(0, max_x)
    plt.title("IC")

    plt.subplot(247)
    sns.kdeplot(cert_x, cert_ic, cmap="Reds", shade=True, bw_adjust=.5)
    # plt.scatter(cert_x, cert_ic, facecolor='k', marker='.', s=8, alpha=.1)
    plt.plot(cert_x, result_ic.intercept + result_ic.slope * cert_x, 'k', lw=2)
    plt.ylim(.25, .45)
    # plt.ylim(0, 1)
    plt.xlim(0, max_x)

    page_ic = page_trend_test(bins_ic)
    print("IC", page_ic)

    tau_ic, p_kend_ic = kendalltau(cert_x, cert_ic, nan_policy='omit', alternative='greater')
    print(f"IC Kendall tau = {tau_ic:.4f}, p-value = {p_kend_ic:.2e}")

    spr_ic, p_spr_ic = spearmanr(cert_x, cert_ic, nan_policy='omit', alternative='greater')
    print(f"IC Spearman r = {spr_ic:.4f}, p-value = {p_spr_ic:.2e}")

    t_ic, p_ic = ttest_ind(cert_pm[mask_pm], cert_ic[mask_ic])
    print(f"IC t-test: t-statistic = {t_ic:.4f}, p-value = {p_ic:.2e}")
    print()

    if show_infomax:
        plt.subplot(244)
        x_cert = []
        bins_im = []
        for i, cert_im in enumerate(cert["IM"]):
            nb_samples = len(cert_im)
            s_needed = intervals - nb_samples % intervals
            cert["IM"][i] = np.r_[np.full(s_needed, np.nan), cert_im]
            cert_x = np.linspace(0, len(cert["IM"][i]) * dx, len(cert["IM"][i]))
            x_cert.append(cert_x)

            x_digit = np.digitize(cert_x[-cert_im.shape[0]:], np.linspace(0, 8, 16))
            bin_median = [np.nanmedian(cert_im[x_digit == j]) for j in range(1, len(page_bins))]
            bins_im.append(bin_median)
        bins_im = kurt2cert(np.array(bins_im))
        cert_im = kurt2cert(np.concatenate(cert["IM"]))
        cert_x = np.concatenate(x_cert)
        mask_im = np.isfinite(cert_x) & np.isfinite(cert_im)
        result_im = linregress(cert_x[mask_im], cert_im[mask_im])
        print("IM", result_im, len(cert_x[mask_im]))
        plt.plot(np.nanmedian(cert_x.reshape((-1, intervals)), axis=1),
                 np.nanmedian(cert_im.reshape((-1, intervals)), axis=1), 'grey', ls='', marker='.', markersize=.5)
        plt.plot(cert_x, result_im.intercept + result_im.slope * cert_x, 'k', lw=2)
        plt.ylim(min_y, max_y)
        plt.xlim(0, max_x)
        plt.title("IM")

        plt.subplot(248)
        sns.kdeplot(cert_x, cert_im, cmap="Reds", shade=True, bw_adjust=.5)
        # plt.scatter(cert_x, cert_ic, facecolor='k', marker='.', s=8, alpha=.1)
        plt.plot(cert_x, result_im.intercept + result_im.slope * cert_x, 'k', lw=2)
        plt.ylim(.25, .45)
        # plt.ylim(0, 1)
        plt.xlim(0, max_x)

        page_im = page_trend_test(bins_im)
        print("IM", page_im)

        tau_im, p_kend_im = kendalltau(cert_x, cert_im, nan_policy='omit', alternative='greater')
        print(f"IM Kendall tau = {tau_im:.4f}, p-value = {p_kend_im:.2e}")

        spr_im, p_spr_im = spearmanr(cert_x, cert_im, nan_policy='omit', alternative='greater')
        print(f"IM Spearman r = {spr_im:.4f}, p-value = {p_spr_im:.2e}")

        t_im, p_im = ttest_ind(cert_pm[mask_pm], cert_im[mask_im])
        print(f"IM t-test: t-statistic = {t_im:.4f}, p-value = {p_im:.2e}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
