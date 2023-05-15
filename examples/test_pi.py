from invertsy.env.world import Seville2009
from invertsy.agent import PathIntegrationAgent
from invertsy.sim.simulation import PathIntegrationSimulation
from invertsy.sim.animation import PathIntegrationAnimation


def main(*args):
    routes = Seville2009.load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    rt = rt[::-1]
    rt[:, 3] = (rt[:, 3] - 0) % 360 - 180
    # rt[:, 3] = (rt[:, 3] - 5) % 360 - 180
    agent = PathIntegrationAgent()
    agent.step_size = .01
    sim = PathIntegrationSimulation(rt, agent=agent, noise=0., name="pi-ant%d-route%d" % (ant_no, rt_no))
    ani = PathIntegrationAnimation(sim, show_history=True)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
