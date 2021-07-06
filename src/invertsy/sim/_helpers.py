from invertsy.agent.agent import VisualNavigationAgent, PathIntegrationAgent, LandmarkIntegrationAgent

from invertpy.brain import MushroomBody
from invertpy.sense import CompoundEye, PolarisationSensor

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
            ax.fill_between(y, x[0], x, facecolor=colour, edgecolor=colour, alpha=.7, lw=.5)

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
    Draws a map of the positions of the ommatidia coloured using their photo-receptor responses.

    Parameters
    ----------
    eye: CompoundEye
        the eye to take the ommatidia positions from
    cmap: str, optional
        the colour map of the responses. Default is 'Greys_r'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.collections.PathCollection
        the ommatidia as a path collection
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


def create_mem_axis(agent, cmap="Greys", subplot=111, ax=None):
    """
    Draws the responses of the PNs, KCs and the familiarity current value in neuron-like arrays.

    Parameters
    ----------
    agent: VisualNavigationAgent
        The agent to get the data from
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    pn: matplotlib.collections.PathCollection
        collection of the PN responses
    kc: matplotlib.collections.PathCollection
        collection of the KC responses
    fam: matplotlib.collections.PathCollection
        collection of the familiarity value per scan
    """
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 13)
    ax.set_aspect('equal', 'box')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    mem = agent.brain[0]  # type: MushroomBody

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


def create_pn_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the PN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: VisualNavigationAgent | LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the PN history responses
    """
    nb_pn = agent.brain[int(isinstance(agent, LandmarkIntegrationAgent))].nb_cs
    return create_image_history(nb_pn, nb_frames, sep=sep, title="PN",  cmap=cmap, subplot=subplot, ax=ax)


def create_kc_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the KC history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: VisualNavigationAgent | LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the KC history responses
    """
    nb_kc = agent.brain[int(isinstance(agent, LandmarkIntegrationAgent))].nb_kc
    return create_image_history(nb_kc, nb_frames, sep=sep, title="KC",  cmap=cmap, subplot=subplot, ax=ax)


def create_familiarity_response_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the familiarity history for every scan as an image, where each pixel is a scan in time and its colour reflects
    the familiarity in this scan. Also the lowest value is marked using a red line.

    Parameters
    ----------
    agent: VisualNavigationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the familiarity history
    matplotlib.lines.Line2D
        the line showing the lowest familiarity value
    """
    nb_scans = agent.nb_scans
    if nb_scans <= 1:
        nb_scans = agent.nb_mental_rotations

    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, nb_scans-1)
    ax.set_xlim(0, nb_frames-1)
    ax.set_yticks([0, nb_scans//2, nb_scans-1])
    angles = np.roll(agent.pref_angles, len(agent.pref_angles) // 2)
    ax.set_yticklabels([angles[0], angles[len(angles) // 2], angles[-1]])
    ax.set_aspect('auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("familiarity", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    fam = ax.imshow(np.zeros((nb_scans, nb_frames), dtype='float32'), cmap=cmap,
                    vmin=0, vmax=1,
                    # vmin=0.40, vmax=0.60,
                    interpolation="none", aspect="auto")

    fam_line, = ax.plot([], [], 'red', lw=.5, alpha=.5)

    if sep is not None:
        ax.plot([sep, sep], [0, nb_scans-1], 'grey', lw=1)

    return fam, fam_line


def create_familiarity_history(nb_frames, sep=None, subplot=111, ax=None):
    """
    Draws a line of the lowest familiarity per iteration.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the line of the lowest familiarity per iteration
    """
    return create_single_line_history(nb_frames, sep=sep, title="familiarity (%)", ylim=100, subplot=subplot, ax=ax)


def create_capacity_history(nb_frames, sep=None, subplot=111, ax=None):
    """
    Draws a line of the available capacity per iteration.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the line of the available capacity per iteration.
    """
    return create_single_line_history(nb_frames, sep=sep, title="capacity (%)", ylim=100, subplot=subplot, ax=ax)


def create_dra_axis(sensor, cmap="coolwarm", centre=None, scale=1., draw_axis=True, subplot=111, ax=None):
    """
    Draws the DRA and the responses of its ommatidia.

    Parameters
    ----------
    sensor: PolarisationSensor
        the compass sensor to get the data and parameters from
    centre: list[float], optional
        the centre of the DRA map. Default is [.5, .5]
    scale: float, optional
        a factor that scales the position of the ommatidia on the figure. Default is 1
    draw_axis: bool, optional
        if True, it draws the axis for the DRA, otherwise it draws on the existing axis. Default is True
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.collections.PathCollection
        the ommatidia of the DRA as a path collection
    """
    omm_x, omm_y, omm_z = sensor.omm_ori.apply(np.array([1, 0, 0])).T

    if ax is None:
        ax = plt.subplot(subplot)

    if centre is None:
        centre = [.5, .5]

    if draw_axis:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
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


def create_cmp_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the compass history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the TB1 history responses
    """
    nb_cmp = agent.brain[2].nb_cmp
    return create_image_history(nb_cmp, nb_frames, sep=sep, title="CMP", cmap=cmap, vmin=0, vmax=1, subplot=subplot,
                                ax=ax)


def create_tb1_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the TB1 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the TB1 history responses
    """
    nb_tb1 = agent.brain[1].nb_tb1
    return create_image_history(nb_tb1, nb_frames, sep=sep, title="TB1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cl1_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CL1 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CL1 history responses
    """
    nb_cl1 = agent.brain[1].nb_cl1
    return create_image_history(nb_cl1, nb_frames, sep=sep, title="CL1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu1_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CPU1 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU1 history responses
    """
    nb_cpu1 = agent.brain[1].nb_cpu1
    return create_image_history(nb_cpu1, nb_frames, sep=sep, title="CPU1", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CPU4 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU4 history responses
    """
    nb_cpu4 = agent.brain[1].nb_cpu4
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_cpu4_mem_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the CPU4 history of memories as an image, where each pixel is a neuron in time and its colour reflects the
    memory of the neuron.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the CPU4 history memories
    """
    nb_cpu4 = agent.brain[1].nb_cpu4
    return create_image_history(nb_cpu4, nb_frames, sep=sep, title="CPU4 (mem)", cmap=cmap, vmin=-1, vmax=1, subplot=subplot, ax=ax)


def create_compass_history(agent, nb_frames, sep=None, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws the compass history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the compass history responses
    """
    nb_cmp = agent.brain[1].nb_cmp
    return create_image_history(nb_cmp, nb_frames, sep=sep, title="Compass", cmap=cmap, vmin=-1, vmax=1,
                                subplot=subplot, ax=ax)


def create_epg_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the E-PG history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the E-PG history responses
    """
    nb_epg = agent.brain[2].nb_epg
    return create_image_history(nb_epg, nb_frames, sep=sep, title="E-PG", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_peg_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the P-EG history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the P-EG history responses
    """
    nb_peg = agent.brain[2].nb_peg
    return create_image_history(nb_peg, nb_frames, sep=sep, title="P-EG", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_pen_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the P-EN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the P-EN history responses
    """
    nb_pen = agent.brain[2].nb_pen
    return create_image_history(nb_pen, nb_frames, sep=sep, title="P-EN", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_pfl_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the PFL3 history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the PFL3 history responses
    """
    nb_pfl = agent.brain[2].nb_pfl3
    return create_image_history(nb_pfl, nb_frames, sep=sep, title="PFL3", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_fbn_history(agent, nb_frames, sep=None, cmap="Greys", subplot=111, ax=None):
    """
    Draws the FsBN history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    agent: LandmarkIntegrationAgent
        the agent to get the data and properties from
    nb_frames: int
        the total number of frames for the animation
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the FsBN history responses
    """
    nb_fbn = agent.brain[2].nb_fbn
    return create_image_history(nb_fbn, nb_frames, sep=sep, title="FsBN", cmap=cmap, vmin=0, vmax=1,
                                subplot=subplot, ax=ax)


def create_bcx_axis(agent, cmap="coolwarm", subplot=111, ax=None):
    """
    Draws all the neurons and ommatidia of the given agent in a single axis, representing a snapshot of their current
    values.

    Parameters
    ----------
    agent: PathIntegrationAgent
        the agent to get the data and properties from
    cmap: str, optional
        the colour map of the responses. Default is 'coolwarm'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    omm: matplotlib.collections.PathCollection
        the ommatidia of the DRA as a path collection
    tb1: matplotlib.collections.PathCollection
        the TB1 responses as a path collection
    cl1: matplotlib.collections.PathCollection
        the CL1 responses as a path collection
    cpu1: matplotlib.collections.PathCollection
        the CPU1 responses as a path collection
    cpu4: matplotlib.collections.PathCollection
        the CPU4 responses as a path collection
    cpu4mem: matplotlib.collections.PathCollection
        the CPU4 memories as a path collection
    """
    if ax is None:
        ax = plt.subplot(subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 5)
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


def create_image_history(nb_values, nb_frames, sep=None, title=None, cmap="Greys", subplot=111, vmin=0, vmax=1,
                         ax=None):
    """
    Draws the history of responses as an image, where each pixel is a neuron in time and its colour reflects the
    response rate of the neuron.

    Parameters
    ----------
    nb_values: int
    nb_frames: int
        the total number of frames for the animation
    title: str, optional
    vmin: float, optional
    vmax: float, optional
    sep: float, optional
        the iteration where the phase changes. Default is None
    cmap: str, optional
        the colour map of the responses. Default is 'Greys'
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.image.AxesImage
        the image of the history responses
    """
    ax = get_axis(ax, subplot)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, nb_values-1)
    ax.set_xlim(0, nb_frames-1)
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


def create_single_line_history(nb_frames, sep=None, title=None, ylim=1., subplot=111, ax=None):
    """
    Draws a single line representing the history of a value.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    title: str, optional
        draw the title for the plot. Default is None
    ylim: float, optional
        the maximum value for the Y axis. Default is 1
    sep: float, optional
        the iteration where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the drawn line
    """
    return create_multi_line_history(nb_frames, 1, sep=sep, title=title, ylim=ylim, subplot=subplot, ax=ax)


def create_multi_line_history(nb_frames, nb_lines, sep=None, title=None, ylim=1., subplot=111, ax=None):
    """
    Draws multiple lines representing the history of many values.

    Parameters
    ----------
    nb_frames: int
        the total number of frames for the animation
    nb_lines: int
    title: str, optional
        draw the title for the plot. Default is None
    ylim: float, optional
        the maximum value for the Y axis. Default is 1
    sep: float, optional
        the iteration where the phase changes. Default is None
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    matplotlib.lines.Line2D
        the drawn lines
    """
    ax = get_axis(ax, subplot)

    ax.set_ylim(0, ylim)
    ax.set_xlim(0, nb_frames)
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
    """
    If the axis is None it creates a new axis in the 'subplot' slot, otherwise it returns the given axis.

    Parameters
    ----------
    subplot: int, tuple
        the subplot ID. Default is 111
    ax: plt.Axes, optional
        the axis to draw the subplot on. Default is None

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        if isinstance(subplot, int):
            ax = plt.subplot(subplot)
        else:
            ax = plt.subplot(*subplot)
    return ax
