from env.seville2009 import load_routes, Seville2009
from simplot.animation import VisualNavigationAnimation


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    ani = VisualNavigationAnimation(rt, world=Seville2009(),
                                    name="vn-ant%d-route%d" % (ant_no, rt_no))
    ani(save=True, show=False, save_type="mp4")


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
