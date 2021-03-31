from env.seville2009 import load_routes
from sim.simulation import RouteSimulation
from simplot.animation import RouteAnimation


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    sim = RouteSimulation(rt, name="seville-ant%d-route%d" % (ant_no, rt_no))
    ani = RouteAnimation(sim)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
