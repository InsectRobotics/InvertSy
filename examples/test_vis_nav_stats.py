import numpy as np
import matplotlib.pyplot as plt

import os
import re

__data__ = os.path.realpath(os.path.join(os.path.curdir, '..', 'data', 'animation', 'stats'))


if __name__ == '__main__':

    df = {
        'model': [],
        'pca': [],
        'scans': [],
        'ant': [],
        'route': [],
        'route_length': [],
        'replaces': [],
        'ommatidia': [],
    }

    for _, _, filenames in os.walk(__data__):
        for filename in filenames:
            match = re.match(r"vn-([a-z]+\-?[A-Z]?)(-pca)?-scan([0-9]+)-ant([0-9]+)-route([0-9]+)(-replace)?-omm([0-9]+).npz",
                             filename)
            if match is None:
                continue
            print(filename)
            model = match.group(1)
            pca = match.group(2) is not None
            nb_scans = int(match.group(3))
            ant_no = int(match.group(4))
            route_no = int(match.group(5))
            replace = match.group(6) is not None
            nb_omm = int(match.group(7))

            try:
                data = np.load(os.path.join(__data__, filename))
            except Exception:
                print(filename)
                continue

            if replace and 'replace' in data:
                df['model'].append(model)
                df['pca'].append(pca)
                df['scans'].append(nb_scans)
                df['ant'].append(ant_no)
                df['route'].append(route_no)
                df['ommatidia'].append(nb_omm)
                df['replaces'].append(data['replace'].sum())
                df['route_length'].append(len(data['outbound']))

    plt.figure('visual-navigation-stats', figsize=(10, 4))

    plt.subplot(121)
    i = np.argsort(df['ommatidia'])
    for key in df:
        df[key] = np.array(df[key])[i]

    nb_models = len(np.unique(df['model']))
    for j, model in enumerate(np.unique(df['model'])):
        i = (df['model'] == model) & (df['scans'] == 121) & df['pca']
        omm = np.sort(np.unique(df['ommatidia']))
        omm_range = omm.max() - omm.min()
        rep = 100 * df['replaces'][i] / df['route_length'][i]
        rep_25 = np.array([np.nanquantile(rep[df['ommatidia'][i] == o], q=0.25) for o in omm])
        rep_50 = np.array([np.nanmedian(rep[df['ommatidia'][i] == o]) for o in omm])
        rep_75 = np.array([np.nanquantile(rep[df['ommatidia'][i] == o], q=0.75) for o in omm])
        plt.fill_between(omm[~np.isnan(rep_50)], rep_25[~np.isnan(rep_50)], rep_75[~np.isnan(rep_50)],
                         'C%d' % j, alpha=.2)
        plt.plot(omm[~np.isnan(rep_50)], rep_50[~np.isnan(rep_50)], 'C%d' % j, linestyle='-', label=model)
        plt.plot(df['ommatidia'][i] + (j / nb_models - .25) * (omm_range / 25), rep,
                 'C%d' % j, linestyle='', marker='.')

    plt.legend()
    plt.xticks([500, 1000, 2000, 3000, 4000, 5000])
    plt.xlim(300, 5200)
    plt.xlabel('number of ommatidia (121 scans)')
    plt.ylabel('mean replacements / meter')
    plt.ylim([0, 5])

    plt.subplot(122)
    i = np.argsort(df['scans'])
    for key in df:
        df[key] = np.array(df[key])[i]

    for j, model in enumerate(np.unique(df['model'])):
        i = (df['model'] == model) & (df['ommatidia'] == 1000) & df['pca']
        sca = np.sort(np.unique(df['scans']))
        sca_range = sca.max() - sca.min()
        rep = 100 * df['replaces'][i] / df['route_length'][i]
        # rep_12 = np.array([np.nanquantile(rep[df['scans'][i] == o], q=0.125) for o in sca])
        rep_25 = np.array([np.nanquantile(rep[df['scans'][i] == o], q=0.25) for o in sca])
        # rep_37 = np.array([np.nanquantile(rep[df['scans'][i] == o], q=0.375) for o in sca])
        rep_50 = np.array([np.nanmedian(rep[df['scans'][i] == o]) for o in sca])
        # rep_63 = np.array([np.nanquantile(rep[df['scans'][i] == o], q=0.625) for o in sca])
        rep_75 = np.array([np.nanquantile(rep[df['scans'][i] == o], q=0.75) for o in sca])
        # rep_88 = np.array([np.nanquantile(rep[df['scans'][i] == o], q=0.875) for o in sca])
        # plt.fill_between(sca, rep_12, rep_88, 'C%d' % j, alpha=.2)
        plt.fill_between(sca[~np.isnan(rep_50)], rep_25[~np.isnan(rep_50)], rep_75[~np.isnan(rep_50)],
                         'C%d' % j, alpha=.2)
        # plt.fill_between(sca, rep_37, rep_63, 'C%d' % j, alpha=.2)
        plt.plot(sca[~np.isnan(rep_50)], rep_50[~np.isnan(rep_50)], 'C%d' % j, linestyle='-', label=model)
        plt.plot(df['scans'][i] + (j / nb_models - .25) * (sca_range / 25), rep,
                 'C%d' % j, linestyle='', marker='.')

    plt.legend()
    plt.xticks([7, 31, 61, 91, 121])
    plt.xlabel('number of scans (1000 ommatidia)')
    plt.ylabel('mean replacements / meter')
    plt.xlim(2, 126)
    plt.ylim([0, 5])

    plt.tight_layout()
    plt.show()
