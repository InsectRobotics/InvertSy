from invertsy.agent import RouteFollowingAgent
from invertsy.env.world import Seville2009
from invertsy.sim.simulation import PathIntegrationSimulation
from invertsy.sim.animation import MapResponsesAnimation

from invertpy.brain.mushroombody import VisualIncentiveCircuit
from invertpy.brain.preprocessing import pca


def main(*args):
    nb_white = 300
    whitening = pca
    lateral_inhibition = False

    routes = Seville2009.load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    rt = rt[::-1]
    rt[:, 3] = (rt[:, 3] - 0) % 360 - 180
    # rt[:, 3] = (rt[:, 3] - 5) % 360 - 180
    agent = RouteFollowingAgent(mb_class=VisualIncentiveCircuit,
                                speed=.01, noise=0.,
                                nb_visual=nb_white, nb_scans=1, mental_scanning=0,
                                whitening=whitening, lateral_inhibition=lateral_inhibition)
    sim = PathIntegrationSimulation(rt, agent=agent, world=Seville2009(), zero_vector=True,
                                    name="vispi-ant%d-route%d" % (ant_no, rt_no))
    ani = MapResponsesAnimation(sim, width=15, height=8)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
