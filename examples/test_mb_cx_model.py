from invertsy.env.world import create_route_from_points, RNG
from invertsy.sim.simulation import NavigationSimulation
from invertsy.sim.animation import NavigationAnimation

import numpy as np


def main(*args):
    nb_routes = 2
    nb_points = 10
    route_max_distance = 4.  # meters
    route_range = np.pi / 12

    print("Mushroom Body and Central Complex simulation for vector memory.")

    default_directions = np.linspace(0, 2 * np.pi, nb_routes, endpoint=False) + np.pi/4
    routes = []
    for i in range(nb_routes):
        distances = route_max_distance * RNG.rand(nb_points)
        angles = 2 * route_range * RNG.rand(nb_points) - route_range + default_directions[i]
        j = np.argsort(distances)
        route_complex = 5 + 5j + (distances * np.exp(1j * angles))[j]
        route_complex = np.insert(route_complex, 0, 5 + 5j)
        x = route_complex.real
        y = route_complex.imag
        z = np.full_like(x, 0.01)
        points = np.array([x, y, z]).T
        route = create_route_from_points(*points, step_size=0.01)
        routes.append(route)

        print(f"Route#: {i+1:d}, steps#: {route.shape[0]:d}")

    sim = NavigationSimulation(routes, name=f"vector-memory-routes{nb_routes:02d}")
    ani = NavigationAnimation(sim)
    ani(save=False, show=True, save_type="mp4", save_stats=False)


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
