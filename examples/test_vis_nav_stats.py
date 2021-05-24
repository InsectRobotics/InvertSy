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

    for j, model in enumerate(np.unique(df['model'])):
        i = (df['model'] == model) & (df['scans'] == 121) & df['pca']
        omm = np.sort(np.unique(df['ommatidia']))
        rep = 100 * df['replaces'][i] / df['route_length'][i]
        rep_25 = [np.nanquantile(rep[df['ommatidia'][i] == o], q=0.25) for o in omm]
        rep_50 = [np.nanmedian(rep[df['ommatidia'][i] == o]) for o in omm]
        rep_75 = [np.nanquantile(rep[df['ommatidia'][i] == o], q=0.75) for o in omm]
        plt.fill_between(omm, rep_25, rep_75, 'C%d' % j, alpha=.2)
        plt.plot(omm, rep_50, 'C%d' % j, linestyle='-', label=model)
        plt.plot(df['ommatidia'][i], rep, 'C%d' % j, linestyle='', marker='.')

    plt.legend()
    plt.xticks(np.sort(np.unique(df['ommatidia'])))
    plt.xlabel('number of ommatidia')
    plt.ylabel('percentage of replaces (%)')
    plt.ylim([0, 5])

    plt.subplot(122)
    i = np.argsort(df['scans'])
    for key in df:
        df[key] = np.array(df[key])[i]

    for j, model in enumerate(np.unique(df['model'])):
        i = (df['model'] == model) & (df['ommatidia'] == 5000) & df['pca']
        sca = np.sort(np.unique(df['scans']))
        rep = 100 * df['replaces'][i] / df['route_length'][i]
        rep_25 = [np.nanquantile(rep[df['scans'][i] == o], q=0.25) for o in sca]
        rep_50 = [np.nanmedian(rep[df['scans'][i] == o]) for o in sca]
        rep_75 = [np.nanquantile(rep[df['scans'][i] == o], q=0.75) for o in sca]
        plt.fill_between(sca, rep_25, rep_75, 'C%d' % j, alpha=.2)
        plt.plot(sca, rep_50, 'C%d' % j, linestyle='-', label=model)
        plt.plot(df['scans'][i], rep, 'C%d' % j, linestyle='', marker='.')

    plt.legend()
    plt.xticks(np.sort(np.unique(df['scans'])))
    plt.xlabel('number of scans')
    plt.ylabel('number of replaces')
    plt.ylim([0, 10])

    plt.tight_layout()
    plt.show()
