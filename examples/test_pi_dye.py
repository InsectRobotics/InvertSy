from invertpy.brain.centralcomplex.dyememory import DyeMemoryCX
from invertpy.brain.centralcomplex import StoneCX
from invertsy.env.world import Seville2009
from invertsy.agent import PathIntegrationAgent
from invertsy.sim.simulation import PathIntegrationSimulation
from invertsy.sim.animation import PathIntegrationAnimation

from datetime import datetime

import loguru as lg
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.signal as ss
import scipy.interpolate as si
import numpy as np
import multiprocessing as mp
import glob
import os
import sys


DEFAULT_ACC = 0.15  # a good value because keeps speed under 1
DEFAULT_DRAG = 0.15

lg.logger.remove()
LOGGER = lg.logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <5}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>    "
           "<level>{message}</level>",
    filter=lambda record: "experiment" not in record["extra"]
)


def generate_random_route(T=1000, step_size=0.01, mean_acc=DEFAULT_ACC, drag=DEFAULT_DRAG, kappa=100.0,
                          max_acc=DEFAULT_ACC, min_acc=0.0, vary_speed=False, min_homing_distance=4, rng=None):
    """
    Generate a random outbound route using bee_simulator physics.
    The rotations are drawn randomly from a von mises distribution and smoothed
    to ensure the agent makes more natural turns.

    Parameters
    ----------
    T
    mean_acc
    drag
    kappa
    max_acc
    min_acc
    vary_speed
    min_homing_distance

    Returns
    -------

    """

    if rng is None:
        rng = np.random.RandomState()

    # Generate random turns
    mu = 0.0
    vm = rng.vonmises(mu, kappa, T)
    rotation = ss.lfilter([1.0], [1, -0.4], vm)
    rotation[0] = 0.0

    route = np.zeros((T, 4))
    route[:, 2] = 0.001  # z-axis is 1 mm above the ground

    # Randomly sample some points within acceptable acceleration and interpolate to create smoothly varying speed.
    if vary_speed:
        if T > 200:
            num_key_speeds = T // 50
        else:
            num_key_speeds = 4

        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = si.interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T, endpoint=True)
        acceleration = f(xnew)
    else:
        acceleration = mean_acc * np.ones(T)

    # Get headings and velocity for each step
    headings = np.zeros(T)
    velocity = np.zeros((T, 2))

    for t in range(1, T):
        headings[t], velocity[t, :] = get_next_state(
            dt=1.0, heading=headings[t-1], velocity=velocity[t-1, :],
            rotation=rotation[t], acceleration=acceleration[t], drag=drag)
        route[t, :2] = route[t-1, :2] + velocity[t, :]

    route[:, 3] = np.angle(np.diff(route[:, 0] + route[:, 1] * 1j, append=True), deg=True)
    route[-1, 3] = route[-2, 3]

    xy = route[:, 0] + route[:, 1] * 1j
    d = np.diff(xy, prepend=0.)
    d *= step_size / np.maximum(abs(d), np.finfo(float).eps)
    xy = np.cumsum(d)

    route[:, 0] = np.real(xy - xy.mean()) + 5
    route[:, 1] = np.imag(xy - xy.mean()) + 5

    if abs(xy[0] - xy[-1]) < min_homing_distance:
        return generate_random_route(T, step_size, mean_acc, drag, kappa, max_acc, min_acc, vary_speed,
                                     min_homing_distance, rng)

    return route


def get_next_state(dt, heading, velocity, rotation, acceleration, drag=0.5):
    """
    Get new heading and velocity, based on relative rotation and acceleration and linear drag.
    Parameters
    ----------
    dt
    heading
    velocity
    rotation
    acceleration
    drag

    Returns
    -------

    """
    theta = rotate(dt, heading, rotation)
    v = velocity + thrust(dt, theta, acceleration)
    v *= (1.0 - drag) ** dt

    return theta, v


def rotate(dt, theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r * dt + np.pi) % (2.0 * np.pi) - np.pi


def thrust(dt, theta, acceleration):
    """
    Thrust vector from current heading and acceleration

    Parameters
    ----------
    dt: float
        delta time
    theta: float
        clockwise radians around z-axis, where 0 is forward
    acceleration: float
        float where max speed is ....?!?

    Returns
    -------

    """
    return np.array([np.sin(theta), np.cos(theta)]) * acceleration * dt


def plot_results(data, name='results'):
    x_in, y_in, z_in, yaw_in = data['xyz'].T
    x_out, y_out, z_out, yaw_out = data['xyz_out'].T
    l_in, l_out = data['L'], data['L_out']
    c_in, c_out = data['C'], data['C_out']

    # tortuosity
    tau_out = l_out / l_in[0]
    tau_in = l_in / l_in[0]
    l_in_ideal = np.maximum(l_in[0] - c_in, 0)
    tau_in_ideal = l_in_ideal / l_in[0]

    epg = data['CL1']
    pfn = data['CPU4']
    fc2 = data['CPU4mem']
    pfl3 = data['CPU1']

    mosaic = """
    AB
    AC
    AD
    AE
    AF
    """
    fig = plt.figure(num=name, figsize=(10, 5))
    ax = fig.subplot_mosaic(mosaic,
                            per_subplot_kw={
                                'A': {'aspect': 'equal'}
                            })
    ax['A'].plot(x_out, y_out, color='grey', label='outbound')
    ax['A'].plot(x_in, y_in, color='orange', label='inbound')
    ax['A'].plot([x_out[0]], [y_out[0]], color='grey', marker=(3, 1, np.rad2deg(yaw_out[0]) - 90), ls='', ms=10)
    ax['A'].plot([x_in[0]], [y_in[0]], color='orange', marker=(3, 1, np.rad2deg(yaw_in[0]) - 90), ls='', ms=10)

    ax['B'].imshow(np.vstack([epg[:, ::2].T, epg[:, 1::2].T]), vmin=0, vmax=1, aspect='auto')
    ax['B'].set_ylabel('EPG')
    ax['B'].set_yticks([0, 15])
    ax['B'].set_xticks([])
    ax['B'].set_ylim([epg.shape[1] - 0.5, -0.5])
    ax['B'].set_xlim([-0.5, epg.shape[0] - 0.5])
    plot_twin_angles(ax['B'], epg, len(l_out))

    ax['C'].imshow(pfn.T, vmin=0, vmax=1, aspect='auto')
    ax['C'].set_ylabel('PFN')
    ax['C'].set_yticks([0, 15])
    ax['C'].set_xticks([])
    ax['C'].set_ylim([pfn.shape[1] - 0.5, -0.5])
    ax['C'].set_xlim([-0.5, pfn.shape[0] - 0.5])
    plot_twin_angles(ax['C'], pfn, len(l_out), angle_max=4 * np.pi)

    ax['D'].imshow(fc2.T, vmin=0, vmax=1, aspect='auto')
    ax['D'].set_ylabel('FC2')
    ax['D'].set_yticks([0, 15])
    ax['D'].set_xticks([])
    ax['D'].set_ylim([fc2.shape[1] - 0.5, -0.5])
    ax['D'].set_xlim([-0.5, fc2.shape[0] - 0.5])
    plot_twin_angles(ax['D'], fc2, len(l_out), angle_max=4 * np.pi)

    ax['E'].imshow(pfl3.T, vmin=0, vmax=1, aspect='auto')
    ax['E'].set_ylabel('PFL3')
    ax['E'].set_yticks([0, 15])
    ax['E'].set_xticks([])
    ax['E'].set_ylim([pfl3.shape[1] - 0.5, -0.5])
    ax['E'].set_xlim([-0.5, pfl3.shape[0] - 0.5])
    plot_twin_angles(ax['E'], pfl3, len(l_out), angle_max=4 * np.pi)

    ax['F'].plot(len(tau_out) + np.arange(len(tau_in)), tau_in_ideal * 100, color='red', ls='--')
    ax['F'].plot(np.arange(len(tau_out)), tau_out * 100, color='grey')
    ax['F'].plot(len(tau_out) + np.arange(len(tau_in)), tau_in * 100, color='orange')
    ax['F'].set_xlim([0, len(tau_out) + len(tau_in) - 1])
    ax['F'].set_ylim([0, 100])
    ax['F'].set_ylabel('distance')

    fig.tight_layout()
    plt.show()


def plot_summarised_results(data, name='summarised_results'):

    x_perc = np.linspace(-150, 200, 1051)
    dataset = {"xy": [], "yaw": [], "perc": [], "turn": [], "tau": [], "tau_opt": [], "memory": []}
    for datum in data:
        x_in, y_in, z_in, yaw_in = datum['xyz'].T
        x_out, y_out, z_out, yaw_out = datum['xyz_out'].T
        xy = np.r_[x_out, x_in] + 1j * np.r_[y_out, y_in]
        yaw = np.r_[yaw_out, yaw_in]
        l_in, c_in = datum['L'], datum['C']
        l = np.r_[datum['L_out'], l_in]
        c = np.r_[datum['C_out'], c_in]

        turn_point = len(x_out)

        home_distance = np.argmin(abs(l_in[0] - c_in))
        t = np.arange(turn_point + len(x_in))
        t_per = (t - turn_point) / home_distance * 100
        dataset["perc"].append(x_perc)
        dataset["turn"].append(np.argmin(abs(x_perc)))

        dataset["xy"].append(np.interp(x_perc, t_per, xy))
        dataset["yaw"].append(np.interp(x_perc, t_per, yaw))

        tau = l / l_in[0]
        tau_opt = np.r_[np.zeros(turn_point), np.maximum(l_in[0] - c_in, 0) / l_in[0] ]
        dataset["tau"].append(np.interp(x_perc, t_per, tau))
        dataset["tau_opt"].append(np.interp(x_perc, t_per, tau_opt))

        r_memory = datum['CPU4mem']
        pref_angles = np.linspace(0, 4 * np.pi, r_memory.shape[1], endpoint=False) + np.pi
        memory = np.sum(r_memory * np.exp(1j * pref_angles[None, :]), axis=1)
        dataset["memory"].append(np.interp(x_perc, t_per, memory))

    plt.figure(name, figsize=(2.5, 5))

    turn = int(np.mean(dataset["turn"]))

    # distance from home
    plt.subplot(311)
    x_mean = np.mean(dataset["perc"], axis=0)
    x_min, x_max = x_mean.min(), x_mean.max()

    tau_25 = np.quantile(dataset["tau"], 0.25, axis=0)
    tau_50 = np.quantile(dataset["tau"], 0.50, axis=0)
    tau_75 = np.quantile(dataset["tau"], 0.75, axis=0)

    tau_opt_50 = np.median(dataset["tau_opt"], axis=0)

    plt.plot([x_mean[turn]] * 2, [0, 100], 'k:')

    plt.fill_between(x_mean[:turn], tau_25[:turn] * 100, tau_75[:turn] * 100,
                     color='grey', edgecolor=None, alpha=0.2)
    plt.fill_between(x_mean[turn-1:], tau_25[turn-1:] * 100, tau_75[turn-1:] * 100,
                     color='orange', edgecolor=None, alpha=0.2)

    plt.plot(x_mean[turn:], tau_opt_50[turn:] * 100, color='red')
    plt.plot(x_mean[:turn], tau_50[:turn] * 100, color='grey')
    plt.plot(x_mean[turn-1:], tau_50[turn-1:] * 100, color='orange')

    plt.xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500], [""] * 11)
    plt.ylim(0, 100)
    plt.xlim(x_min, x_max)
    plt.ylabel('distance from home [%]', fontsize=8)
    # plt.xlabel('distance travelled\nrelative to turning point [%]', fontsize=8)

    # heading error
    plt.subplot(312)
    locs = np.array(dataset['xy'])
    yaws = np.rad2deg(np.array(dataset['yaw']))
    home_headings = np.angle(locs[:, :1] - locs, deg=True)

    heading_error = abs((home_headings - yaws + 180) % 360 - 180)
    heading_error_25 = np.quantile(heading_error, 0.25, axis=0)
    heading_error_50 = np.quantile(heading_error, 0.50, axis=0)
    heading_error_75 = np.quantile(heading_error, 0.75, axis=0)

    plt.plot([x_mean[turn]] * 2, [0, 180], 'k:')

    plt.fill_between(x_mean[:turn], heading_error_25[:turn], heading_error_75[:turn],
                     color='grey', edgecolor=None, alpha=0.2)
    plt.fill_between(x_mean[turn-1:], heading_error_25[turn-1:], heading_error_75[turn-1:],
                     color='orange', edgecolor=None, alpha=0.2)

    plt.plot(x_mean[:turn], heading_error_50[:turn], color='grey')
    plt.plot(x_mean[turn-1:], heading_error_50[turn-1:], color='orange')

    plt.xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500], [""] * 11)
    plt.yticks([0, 45, 90, 135, 180])
    plt.ylim(0, 180)
    plt.xlim(x_min, x_max)
    plt.ylabel(r'heading error [$^o$]', fontsize=8)
    # plt.xlabel('distance travelled relative to turning point [%]', fontsize=8)

    # memory error
    plt.subplot(313)
    v_memory = np.array(dataset['memory'])
    a_memory = np.angle(v_memory, deg=True)

    memory_error = abs((home_headings - a_memory + 180) % 360 - 180)
    memory_error_25 = np.quantile(memory_error, 0.25, axis=0)
    memory_error_50 = np.quantile(memory_error, 0.50, axis=0)
    memory_error_75 = np.quantile(memory_error, 0.75, axis=0)

    plt.plot([x_mean[turn]] * 2, [0, 90], 'k:')

    plt.fill_between(x_mean[:turn], memory_error_25[:turn], memory_error_75[:turn],
                     color='grey', edgecolor=None, alpha=0.2)
    plt.fill_between(x_mean[turn-1:], memory_error_25[turn-1:], memory_error_75[turn-1:],
                     color='orange', edgecolor=None, alpha=0.2)

    plt.plot(x_mean[:turn], memory_error_50[:turn], color='grey')
    plt.plot(x_mean[turn-1:], memory_error_50[turn-1:], color='orange')

    plt.xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500],
               [-500, "", -300, "", -100, "", 100, "", 300, "", 500])
    plt.yticks([0, 30, 60, 90])
    plt.ylim(0, 90)
    plt.xlim(x_min, x_max)
    plt.ylabel(r'memory error [$^o$]', fontsize=8)
    plt.xlabel('distance travelled\nrelative to turning point [%]', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_twin_angles(axis, responses, nb_out, angle_min=0, angle_max=2*np.pi):

    bins = responses.shape[1]
    nb_in = responses.shape[0] - nb_out
    ax_twin = axis.twinx()
    pref_angles = np.linspace(angle_min, angle_max, bins, endpoint=False)
    ang = np.angle(np.mean(responses * np.exp(1j * pref_angles)[None, :], axis=1), deg=True)
    ang = (ang + 0.5 / bins * 360) % 360 - 0.5 / bins * 360
    ts_out = np.arange(nb_out)
    ang_out = ang[:nb_out]
    ts_in = nb_out + np.arange(nb_in)
    ang_in = ang[nb_out:]
    for ts_i, ang_i, c in zip([ts_out, ts_in], [ang_out, ang_in], ["grey", "orange"]):

        split_i = np.where(abs(np.diff(ang_i)) > 90)[0] + 1
        ang_s = np.split(ang_i, split_i)
        ts_s = np.split(ts_i, split_i)
        for ts, ang in zip(ts_s, ang_s):
            ax_twin.plot(ts, ang, ls='-', color=c)
            ax_twin.plot(ts, ang + 360, ls='-', color=c)

    ax_twin.set_yticks([0, 180, 360, 15 / 16 * 720])
    ax_twin.set_ylim([15.5 / 16 * 720, -0.5 / 16 * 720])

    return ax_twin


def run_simulation(task):
    simulation_name, route, ts_outbound, ts_inbound, step_size, seed, noise, animation, cx_class, cx_params = task

    lg.logger.info(f"Seed: {seed}")
    rng = np.random.RandomState(seed)
    agent = PathIntegrationAgent(cx_class=eval(cx_class), cx_params=cx_params, noise=noise, rng=rng)
    agent.step_size = step_size

    if route == "random":
        rt = generate_random_route(step_size=agent.step_size, T=ts_outbound, rng=rng)
    elif route == "default":
        rt = Seville2009.load_routes(Seville2009.ROUTES_FILENAME, degrees=True)["path"][0]
    else:
        rt = Seville2009.load_routes(route, degrees=True)["path"][0]

    sim = PathIntegrationSimulation(rt, agent=agent, nb_iterations=ts_outbound+ts_inbound,
                                    name=simulation_name, rng=rng)

    if animation is not None:
        lg.logger.info(f"Running animation: {simulation_name}")
        ani_kwargs = {
            "save": animation == 'save',
            "show": animation == 'show',
            "save_stats": True,
            "save_type": 'mp4'
        }
        ani = PathIntegrationAnimation(sim, show_history=True)
        ani(**ani_kwargs)
    else:
        lg.logger.info(f"Running simulation: {simulation_name}")
        sim(save=True)

    return simulation_name


def main(simulation_name='pi', route='random', ts_outbound=1000, ts_inbound=1500,
         step_size=0.01, seed=2023, noise=0.1, animation=None,
         cx_class='StoneCX', cx_params=None, threads=None):

    if not isinstance(seed, list):
        seed = [seed]

    if len(seed) == 2 and seed[0] < seed[1]:
        seed_end = seed[1]
        seed[1] = seed[0] + 1
        for i in range(seed[1] + 1, seed_end + 1):
            seed.append(i)

    if threads is None:
        threads = np.minimum(len(seed), 8)

    tasks = []
    for sd in seed:
        simulation_name_i = os.path.join(simulation_name, f"{datetime.now().strftime('%Y%m%d-%H%M')}_{sd}")
        tasks.append(
            (simulation_name_i, route, ts_outbound, ts_inbound, step_size, sd, noise, animation, cx_class, cx_params))

    mp.set_start_method('forkserver')  # fixes problem with multiprocessing in mac
    results = mp.Pool(threads).imap_unordered(run_simulation, tasks)

    for result in results:
        lg.logger.info(f"Finished: {result}", flush=True)


if __name__ == '__main__':
    import warnings
    import argparse
    import yaml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        parser = argparse.ArgumentParser(
            description="Run a path integration test."
        )

        parser.add_argument("-c", dest='config_file', type=str, required=False, default=None,
                            help="File with the configuration for the experiment.")
        parser.add_argument('-r', dest='results_file', type=str, required=False, default=None,
                            help='Load the results instead of running them.')
        parser.add_argument('-a', dest='results_directory', type=str, required=False, default=None,
                            help='Load the results in the directory to plot summary')

        p_args = parser.parse_args()

        kwargs = {}
        if p_args.config_file is not None:
            lg.logger.info(f"Reading configuration file: {p_args.config_file}")
            with open(p_args.config_file, 'r') as f:
                kwargs = yaml.safe_load(f)
            main(**kwargs)

        if p_args.results_file is not None:
            dat = {}
            if p_args.results_file in ["current", "last"]:
                list_of_files = glob.glob(os.path.join('..', 'data', 'animation', 'stats', '**', '*.npz'), recursive=True)
                latest_file = os.path.abspath(max(list_of_files, key=os.path.getctime))
                lg.logger.info(f"Reading simulation data file: {latest_file}")
                dat[latest_file] = np.load(latest_file)
            elif os.path.exists(p_args.results_file) and os.path.isdir(p_args.results_file):
                list_of_files = glob.glob(os.path.join(p_args.results_file, '**', '*.npz'), recursive=True)

                for f in list_of_files:
                    dat[f] = np.load(f)
            else:
                lg.logger.info(f"Reading simulation data file: {p_args.results_file}")
                dat[p_args.results_file] = np.load(p_args.results_file)

            for name_, dat_ in dat.items():
                dir_name, file_name = os.path.split(name_)
                _, dir_name = os.path.split(dir_name)
                file_name = file_name[:-4]
                plot_results(dat_, name=f"{dir_name}_{file_name}")

        if p_args.results_directory is not None:
            if p_args.results_directory in ["current", "last"]:
                list_of_dirs = glob.glob(os.path.join('..', 'data', 'animation', 'stats', '*'))
                latest_dir = os.path.abspath(max(list_of_dirs, key=os.path.getmtime))
            else:
                latest_dir = os.path.abspath(p_args.results_directory)

            lg.logger.info(f"Reading simulation data from directory: {latest_dir}")
            list_of_files = glob.glob(os.path.join(latest_dir, '*.npz'))

            dat = []
            for file_i in list_of_files:
                lg.logger.info(f"Reading data from: {file_i}")
                dat.append(np.load(file_i))

            plot_summarised_results(dat, name=os.path.split(latest_dir)[-1])

