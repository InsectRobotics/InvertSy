from env.seville2009 import load_routes, Seville2009
from sim.simulation import VisualNavigationSimulation
from simplot.animation import VisualNavigationAnimation


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]), end='')

    save, show = False, True
    nb_scans = 11
    nb_ommatidia = 5000
    replace = True
    agent_name = "vn-whillshaw-pca-scan%d-ant%d-route%d%s" % (nb_scans, ant_no, rt_no, "-replace" if replace else "")
    agent_name += ("-omm%d" % nb_ommatidia) if nb_ommatidia is not None else ""
    print(" - Agent: %s" % agent_name, ("- save " if save else "") + ("- show" if show else ""))

    sim = VisualNavigationSimulation(rt, world=Seville2009(), calibrate=True, nb_scans=nb_scans,
                                     nb_ommatidia=nb_ommatidia, name=agent_name, free_motion=not replace)
    ani = VisualNavigationAnimation(sim, show_history=True, name=agent_name)
    ani(save=save, show=show, save_type="mp4")


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
