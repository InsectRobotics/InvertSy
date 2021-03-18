from agent.pathintegration import PathIntegrationAgent
from env.seville2009 import load_routes
from simplot.routes import anim_path_integration

import sys


def main(*args):
    routes = load_routes(degrees=True)
    ant_no, rt_no, rt = routes['ant_no'][0], routes['route_no'][0], routes['path'][0]
    print("Ant#: %d, Route#: %d, steps#: %d" % (ant_no, rt_no, rt.shape[0]))

    agent = PathIntegrationAgent()

    anim_path_integration(agent, rt, save=True, show=False, fps=15,
                          title="pi-ant%d-route%d" % (ant_no, rt_no))


if __name__ == '__main__':
    main(*sys.argv)
