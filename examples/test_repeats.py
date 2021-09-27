import numpy as np
import matplotlib.pyplot as plt

import os
import re

__data__ = os.path.realpath(os.path.join(os.path.curdir, '..', 'data', 'animation', 'stats'))


if __name__ == '__main__':

    df = {
        'model': [],
        'pca': [],
        'ant': [],
        'route': [],
        'route_length': [],
        'replaces': [],
        'ommatidia': [],
        'repeat': []
    }

    for _, _, filenames in os.walk(__data__):
        for filename in filenames:
            match = re.match(
                r"vnopp-([a-z]+\-?[A-Z]?)(-pca)?-ant([0-9]+)-route([0-9]+)(-replace)?-omm([0-9]+)-repeat([0-9]+).npz",
                filename)
            if match is None:
                continue
            print(filename)
            model = match.group(1)
            pca = match.group(2) is not None
            ant_no = int(match.group(3))
            route_no = int(match.group(4))
            replace = match.group(5) is not None
            nb_omm = int(match.group(6))
            repeat = int(match.group(7))

            try:
                data = np.load(os.path.join(__data__, filename))
            except Exception:
                print(filename)
                continue

            if replace and 'replace' in data:
                df['model'].append(model)
                df['pca'].append(pca)
                df['ant'].append(ant_no)
                df['route'].append(route_no)
                df['ommatidia'].append(nb_omm)
                df['replaces'].append(data['replace'].sum())
                df['route_length'].append(len(data['outbound']))
                df['repeat'].append(repeat)

    plt.figure('repeats-analysis', figsize=(5, 4))

    i = np.argsort(df['ommatidia'])
    for key in df:
        df[key] = np.array(df[key])[i]

    nb_models = len(np.unique(df['model']))
    for j, model in enumerate(np.unique(df['model'])):
        i = df['model'] == model
        repeats = np.sort(np.unique(df['repeat']))
        repeats_range = repeats.max() - repeats.min()
        rep = df['replaces'][i]
        # rep = 100 * df['replaces'][i] / df['route_length'][i]
        rep_25 = np.array([np.nanquantile(rep[df['repeat'][i] == o], q=0.25) for o in repeats])
        rep_50 = np.array([np.nanmedian(rep[df['repeat'][i] == o]) for o in repeats])
        rep_75 = np.array([np.nanquantile(rep[df['repeat'][i] == o], q=0.75) for o in repeats])
        plt.fill_between(repeats[~np.isnan(rep_50)], rep_25[~np.isnan(rep_50)], rep_75[~np.isnan(rep_50)],
                         'C%d' % j, alpha=.2)
        plt.plot(repeats[~np.isnan(rep_50)], rep_50[~np.isnan(rep_50)], 'C%d' % j, linestyle='-', label=model)
        plt.plot(df['repeat'][i] + (j / nb_models - .25) * (repeats_range / 25), rep,
                 'C%d' % j, linestyle='', marker='.')

    plt.legend()
    plt.xticks(np.sort(np.unique(df['repeat'])))
    plt.xlim(0, 20)
    plt.xlabel('number of repeats')
    plt.ylabel('number of replacements')
    # plt.ylabel('mean replacements / meter')
    plt.ylim([0, 15])

    plt.tight_layout()
    plt.savefig("repeats-analysis.png")
    plt.show()
