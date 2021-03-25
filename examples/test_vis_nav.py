from env.seville2009 import load_routes, Seville2009
from simplot.animation import VisualNavigationAnimation


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

    nb_scans = 31
    agent_name = "vn-whillshaw-pca-%d-ant%d-route%d" % (nb_scans, ant_no, rt_no)
    print(" - Agent: %s" % agent_name)

    ani = VisualNavigationAnimation(rt, world=Seville2009(), show_history=True, calibrate=True,
                                    nb_scans=nb_scans, name=agent_name)
    ani(save=True, show=False, save_type="mp4")


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
