from invertsy.agent import Agent
from invertsy.sim.animation import GradientVectorAnimation
from invertsy.sim.simulation import Gradient, GradientVectorSimulation

import matplotlib.pyplot as plt
from matplotlib import animation
from celluloid import Camera
from scipy.spatial.transform import Rotation as R

import numpy as np


def main(*args):
    print("Mushroom body and fan-shaped body integration for route following.")

    ani_name = "animation-08"
    build_animation = False
    save = build_animation
    max_time = 1000
    agent = Agent(speed=0.04)
    init_xyz = 0, -5, 0.01
    init_ori = R.from_euler("Z", 0)
    # init_ori = R.from_euler("Z", np.pi)
    # init_xyz = -5, -2, 0.01

    if not build_animation:
        plt.ion()

    x = 4 * np.sin(np.linspace(0, 2 * np.pi, 1000)) * 1.
    y = np.linspace(-4, 4, 1000)
    z = np.full(1000, 0.01)
    phi = np.rad2deg(np.arctan2(y, x))
    route = np.vstack([[x], [y], [z], [phi]]).T
    print(f"Animation: '{ani_name}'; Route.shape: {route.shape}")

    grad = Gradient(route, sigma=1, grad_type="gaussian")
    #
    # plt.figure("familiarity", figsize=(2, 2))
    # ax = plt.subplot(111, polar=True)
    # ax.set_theta_zero_location("N")
    # theta = np.linspace(0, 2 * np.pi, 181)
    # ax.plot(theta, grad(4, -2, theta), 'k-', lw=2)
    # ax.set_ylim(0, 1)
    # plt.show()
    #
    # sys.exit()

    sim = GradientVectorSimulation(agent=agent, gradient=grad, name=f"mb-fb-run")
    sim.agent.xyz = init_xyz
    sim.agent.ori = init_ori
    sim()

    # ani = GradientAnimation(grad)
    ani = GradientVectorAnimation(grad, fps=15, height=6, width=9, max_time=max_time, mosaic="""
    AAO
    AAG
    AAI
    AAL
    AAF
    AAJ
    NBM""", name=ani_name)
    sim.callback = ani

    if build_animation:
        ani.reset(sim.agent)

        def init():
            sim.agent.xyz = init_xyz
            sim.agent.ori = init_ori
            return ani.reset(sim.agent)

        anim = animation.FuncAnimation(ani.figure, sim, init_func=init,
                                       frames=max_time, interval=10, blit=True)
        if save:
            writer = animation.FFMpegWriter(fps=ani.frames_per_second)
            anim.save(rf"C:\Users\Odin\OneDrive - University of Edinburgh\Projects\2022-InsectNeuroNano\{ani_name}.mp4",
                      writer=writer)
        else:
            plt.show()
    else:
        for _ in range(max_time-1):
            sim()
            plt.waitforbuttonpress(.5 / ani.frames_per_second)
        plt.waitforbuttonpress()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
