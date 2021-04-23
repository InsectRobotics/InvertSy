from invertsy.agent import VisualNavigationAgent, PathIntegrationAgent

from invertpy.brain import CelestialCompass, MushroomBody
from invertpy.sense import CompoundEye

import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.lines
import matplotlib.image
import numpy as np


def create_map_axis(world=None, nest=None, feeder=None, subplot=111, ax=None):
    """
    Draws a map with all the vegetation from the world (if given), the nest and feeder positions (if given) and returns
    the ongoing and previous paths of the agent, the agent's current position, the marker (arror) of the agents facing
    direction, the calibration points and the points where the agent is taken back on the route after replacing.

    Parameters
    ----------
    world: Seville2009, optional
        the world containing the vegetation. Default is None
    nest: np.ndarray[float], optional
        the position of the nest. Default is None
    feeder: np.ndarray[float], optional
        the position of the feeder. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    line_c: matplotlib.lines.Line2D
        the ongoing path of the agent
    line_b: matplotlib.lines.Line2D
        the previous paths of the agent
    pos: matplotlib.collections.PathCollection
        the current position of the agent
    marker: tuple[np.ndarray[float], np.ndarray[float]]
        the marker parameters
    cal: matplotlib.collections.PathCollection
        the points on the map where the calibration took place
    poi: matplotlib.collections.PathCollection
        the points on the map where the agent was put back on the route
    """

    if ax is None:
        ax = plt.subplot(subplot)

    line_b, = ax.plot([], [], 'grey', lw=2)

    if world is not None:
        for polygon, colour in zip(world.polygons, world.colours):
            x = polygon[[0, 1, 2, 0], 0]
            y = polygon[[0, 1, 2, 0], 1]
            ax.plot(y, x, c=colour)

    if nest is not None:
        ax.scatter([nest[1]], [nest[0]], marker='o', s=50, c='black')
        ax.text(nest[1] - 1, nest[0] - .5, "Nest")

    if feeder is not None:
        ax.scatter([feeder[1]], [feeder[0]], marker='o', s=50, c='black')
        ax.text(feeder[1] + .2, feeder[0] + .2, "Feeder")

    ax.set_ylim(0, 10)
    ax.set_xlim(0, 10)
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', labelsize=8)

    cal = ax.scatter([], [], marker='.', s=50, c='orange')

    poi = ax.scatter([], [], marker='.', s=100, c='blue')

    line_c, = ax.plot([], [], 'r', lw=2)
    pos = ax.scatter([], [], marker=(3, 2, 0), s=100, c='red')

    points = [0, 2, 3, 4, 6]
    vert = np.array(pos.get_paths()[0].vertices)[points]
    vert[0] *= 2
    codes = pos.get_paths()[0].codes[points]
    vert = np.hstack([vert, np.zeros((vert.shape[0], 1))])

    return line_c, line_b, pos, (vert, codes), cal, poi


def create_eye_axis(eye, cmap="Greys_r", subplot=111, ax=None):
    """

    Parameters
    ----------
    eye: CompoundEye
    cmap: str, optional
    subplot: int, tuple
    ax: plt.Axes, optional

    Returns
    -------
    matplotlib.collections.PathCollection
    """
    if ax is None:
        ax = plt.subplot(subplot)
    ax.set_yticks(np.sin([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]))
    ax.set_yticklabels([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-180, 180)
    ax.tick_params(axis='both', labelsize=8)

    yaw, pitch, roll = eye.omm_ori.as_euler('ZYX', degrees=True).T
    eye_size = 5000. / eye.nb_ommatidia * eye.omm_area * 80
    omm = ax.scatter(yaw.tolist(), (np.sin(np.deg2rad(-pitch))).tolist(), s=eye_size,
                     c=np.zeros(yaw.shape[0], dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    return omm


def create_mem_axis(agent: VisualNavigationAgent, cmap="Greys", subplot=111, ax=None):
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, 5])
    ax.set_xlim([0, 13])
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    mem = agent.brain[0]  # type: MushrooBody

    size = 400.
    ax.text(.1, 4.8, "PN", fontsize=10)
    pn = ax.scatter(np.linspace(.3, 12.7, mem.nb_cs), np.full(mem.nb_cs, 4.5), s=size / mem.nb_cs,
                    c=np.zeros(mem.nb_cs, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 3.8, "KC", fontsize=10)
    nb_rows = 50
    nb_cols = int(mem.nb_kc / nb_rows) + 1
    x = np.array([np.linspace(.3, 12.7, nb_cols)] * nb_rows).flatten()[:mem.nb_kc]
    y = np.array([np.linspace(1.3, 3.5, nb_rows)] * nb_cols).T.flatten()[:mem.nb_kc]
    kc = ax.scatter(x, y, s=size / mem.nb_kc, c=np.zeros(mem.nb_kc, dtype='float32'), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 0.8, "familiarity", fontsize=10)
    nb_fam = len(agent.pref_angles)
    fam = ax.scatter(np.linspace(.3, 12.7, nb_fam), np.full(nb_fam, 0.5), s=size / nb_fam,
                     c=np.zeros(nb_fam, dtype='float32'), cmap=cmap, vmin=0, vmax=mem.nb_cs * mem.sparseness)

    return pn, kc, fam


def create_pn_history(agent: VisualNavigationAgent, nb_frames: int, sep: float = None, cmap="Greys",
                      subplot=111, ax=None):
    nb_pn = agent.brain[0].nb_cs
    return create_image_history(nb_pn, nb_frames, sep=sep, title="PN",  cmap=cmap, subplot=subplot, ax=ax)


def create_kc_history(agent: VisualNavigationAgent, nb_frames: int, sep: float = None, cmap="Greys",
                      subplot=111, ax=None):
    nb_kc = agent.brain[0].nb_kc
    return create_image_history(nb_kc, nb_frames, sep=sep, title="KC",  cmap=cmap, subplot=subplot, ax=ax)


def create_familiarity_response_history(agent: VisualNavigationAgent, nb_frames: int, sep: float = None, cmap="Greys",
                                        subplot=111, ax=None):
    nb_scans = agent.nb_scans

    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, nb_scans-1])
    ax.set_xlim([0, nb_frames-1])
    ax.set_yticks([0, nb_scans//2, nb_scans-1])
    ax.set_yticklabels([int(agent.pref_angles[0]), int(agent.pref_angles[nb_scans//2]), int(agent.pref_angles[-1])])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("familiarity", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    fam = ax.imshow(np.zeros((nb_scans, nb_frames), dtype='float32'), cmap=cmap, vmin=0, vmax=.2,
                    interpolation="none", aspect="auto")

    fam_line, = ax.plot([], [], 'red', lw=.5, alpha=.5)

    if sep is not None:
        ax.plot([sep, sep], [0, nb_scans-1], 'grey', lw=1)

    return fam, fam_line


def create_familiarity_history(nb_frames: int, sep: float = None, subplot=111, ax=None):
    return create_single_line_history(nb_frames, sep=sep, title="familiarity (%)", ylim=20, subplot=subplot, ax=ax)


def create_capacity_history(nb_frames: int, sep: float = None, subplot=111, ax=None):
    return create_single_line_history(nb_frames, sep=sep, title="capacity (%)", ylim=100, subplot=subplot, ax=ax)


def create_dra_axis(sensor: CelestialCompass, cmap="coolwarm", centre=None, scale=1., draw_axis=True,
                    subplot=111, ax=None):
    omm_x, omm_y, omm_z = sensor.omm_xyz.T

    if ax is None:
        ax = plt.subplot(subplot)

    if centre is None:
        centre = [.5, .5]

    if draw_axis:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.set_aspect('equal', 'box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    size = 20. * scale
    ax.text(centre[0] - .7, centre[1] + .3, "POL", fontsize=10)
    omm = ax.scatter((omm_y * scale + centre[0]).tolist(), (omm_x * scale + centre[1]).tolist(), s=size,
                     c=np.zeros(omm_y.shape[0], dtype='float32'), cmap=cmap, vmin=-.5, vmax=.5)

    return omm


def create_tb1_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                       subplot=111, ax=None):
    nb_tb1 = agent.brain[1].nb_tb1
    return create_image_history(nb_tb1, nb_frames, sep=sep, title="TB1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cl1_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                       subplot=111, ax=None):
    nb_cl1 = agent.brain[1].nb_cl1
    return create_image_history(nb_cl1, nb_frames, sep=sep, title="CL1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu1_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                        subplot=111, ax=None):
    nb_cpu1 = agent.brain[1].nb_cpu1
    return create_image_history(nb_cpu1, nb_frames, sep=sep, title="CPU1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                        subplot=111, ax=None):
    nb_cpu4 = agent.brain[1].nb_cpu4
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_mem_history(agent: PathIntegrationAgent, nb_frames: int, sep: float = None, cmap="coolwarm",
                            subplot=111, ax=None):
    nb_cpu4 = agent.brain[1].nb_cpu4
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4 (mem)", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cx_axis(agent: PathIntegrationAgent, cmap="coolwarm", subplot=111, ax=None):
    if ax is None:
        ax = plt.subplot(subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, 5])
    ax.set_xlim([0, 5])
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    size = 20.
    omm = create_dra_axis(agent.sensors[0], cmap=cmap, centre=[.8, 4.5], draw_axis=False, ax=ax)

    ax.text(1.5, 4.8, "TB1", fontsize=10)
    tb1 = ax.scatter(np.linspace(2, 4.5, 8), np.full(8, 4.5), s=2 * size,
                     c=np.zeros_like(agent.brain[1].r_tb1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 3.8, "CL1", fontsize=10)
    cl1 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 3.5), s=2 * size,
                     c=np.zeros_like(agent.brain[1].r_cl1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 2.8, "CPU1", fontsize=10)
    cpu1 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 2.5), s=2 * size,
                      c=np.zeros_like(agent.brain[1].r_cpu1), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, 1.8, "CPU4", fontsize=10)
    cpu4 = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, 1.5), s=2 * size,
                      c=np.zeros_like(agent.brain[1].r_cpu4), cmap=cmap, vmin=0, vmax=1)

    ax.text(.1, .8, "CPU4 (mem)", fontsize=10)
    cpu4mem = ax.scatter(np.linspace(.5, 4.5, 16), np.full(16, .5), s=2 * size,
                         c=np.zeros_like(agent.brain[1].r_cpu4), cmap=cmap, vmin=0, vmax=1)

    return omm, tb1, cl1, cpu1, cpu4, cpu4mem


def create_image_history(nb_values: int, nb_frames: int, sep: float = None, title: str = None, cmap="Greys",
                         subplot=111, vmin=0, vmax=1, ax=None):
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, nb_values-1])
    ax.set_xlim([0, nb_frames-1])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if title is not None:
        ax.set_ylabel(title)

    im = ax.imshow(np.zeros((nb_values, nb_frames), dtype='float32'), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="none", aspect="auto")

    if sep is not None:
        ax.plot([sep, sep], [0, nb_values-1], 'grey', lw=1)

    return im


def create_single_line_history(nb_frames: int, sep: float = None, title: str = None, ylim: float = 1., subplot=111, ax=None):
    return create_multi_line_history(nb_frames, 1, sep=sep, title=title, ylim=ylim, subplot=subplot, ax=ax)


def create_multi_line_history(nb_frames: int, nb_lines: int, sep: float = None, title: str = None, ylim: float = 1.,
                              subplot=111, ax=None):
    ax = get_axis(ax, subplot)

    ax.set_ylim([0, ylim])
    ax.set_xlim([0, nb_frames])
    ax.tick_params(axis='both', labelsize=8)
    ax.set_aspect('auto', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if sep is not None:
        ax.plot([sep, sep], [0, ylim], 'grey', lw=3)

    lines, = ax.plot(np.full((nb_frames, nb_lines), np.nan), 'k-', lw=2)
    if title is not None:
        ax.text(120, ylim * 1.05, title, fontsize=10)

    return lines


def get_axis(ax=None, subplot=111):
    if ax is None:
        if isinstance(subplot, int):
            ax = plt.subplot(subplot)
        else:
            ax = plt.subplot(*subplot)
    return ax
