from sky import Sky
from ephemeris import Sun
from observer import Observer
from sensor import Compass, decode_sph
from conditions import Hybrid
from antworld import load_routes
from centralcomplex import CX
from sphere import tilt
from plots import plot_route
from utils import eps

from datetime import datetime, timedelta
from pytz import timezone

import numpy as np
import matplotlib.pyplot as plt
import os

__root__ = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
__data__ = os.path.join(__root__, "data", "ephemeris_routes")


def encode(theta, phi, Y, P, A, theta_t=0., phi_t=0., d_phi=0., nb_tcl=8, sigma=np.deg2rad(13),
           shift=np.deg2rad(40)):
    n = theta.shape[0]
    alpha = (phi + np.pi / 2) % (2 * np.pi) - np.pi
    phi_tcl = np.linspace(0., 2 * np.pi, nb_tcl, endpoint=False)  # TB1 preference angles
    phi_tcl = (phi_tcl + d_phi) % (2 * np.pi)

    # Input (POL) layer -- Photo-receptors
    s_1 = Y * (np.square(np.sin(A - alpha)) + np.square(np.cos(A - alpha)) * np.square(1. - P))
    s_2 = Y * (np.square(np.cos(A - alpha)) + np.square(np.sin(A - alpha)) * np.square(1. - P))
    r_1, r_2 = np.sqrt(s_1), np.sqrt(s_2)
    r_pol = (r_1 - r_2) / (r_1 + r_2 + eps)

    # Tilting (CL1) layer
    d_cl1 = (np.sin(shift - theta) * np.cos(theta_t) +
             np.cos(shift - theta) * np.sin(theta_t) *
             np.cos(phi - phi_t))
    gate = np.power(np.exp(-np.square(d_cl1) / (2. * np.square(sigma))), 1)
    w = -float(nb_tcl) / (2. * float(n)) * np.sin(phi_tcl[np.newaxis] - alpha[:, np.newaxis]) * gate[:, np.newaxis]
    r_tcl = r_pol.dot(w)

    R = r_tcl.dot(np.exp(-np.arange(nb_tcl) * (0. + 1.j) * 2. * np.pi / float(nb_tcl)))
    res = np.clip(3.5 * (np.absolute(R) - .53), 0, 2)  # certainty of prediction
    ele_pred = 26 * (1 - 2 * np.arcsin(1 - res) / np.pi) + 15
    d_phi += np.deg2rad(9 + np.exp(.1 * (54 - ele_pred))) / (60. / float(dt))

    return r_tcl, d_phi


if __name__ == '__main__':
    build_stats = False
    load_stats = True
    show_routes = True

    stats = {
        "max_alt": [],
        "noise": [],
        "opath": [],
        "ipath": [],
        "d_x": [],
        "d_c": [],
        "tau": []
    }

    if build_stats:
        noise = 0.
        dx = 1e-02  # meters
        dt = 1 / 30  # minutes
        delta = timedelta(minutes=dt)
        routes = load_routes()
        flow = dx * np.ones(2) / np.sqrt(2)
        theta_t, phi_t = 0., 0.

        compass = Compass()
        sky = Sky(phi_s=np.pi, theta_s=np.pi / 3)
        sun = Sun()

        avg_time = timedelta(0.)

        # sun position
        p1 = Observer(lon=-84.628, lat=45.868, inrad=False,
                      date=datetime(2020, 8, 20, 12, tzinfo=timezone("US/Eastern")),
                      city="Mackinac Island (MI)")  # start
        p2 = Observer(lon=-89.963, lat=39.569, inrad=False,
                      date=datetime(2020, 9, 8, 12, tzinfo=timezone("US/Central")),
                      city="Illinois (IL)")  # stop 1
        p3 = Observer(lon=-96.314445, lat=30.601389, inrad=False,
                      date=datetime(2020, 10, 1, 12, tzinfo=timezone("US/Central")),
                      city="College Station (TX)")  # stop 2
        p4 = Observer(lon=-101.649200, lat=19.646402, inrad=False,
                      date=datetime(2020, 11, 1, 12, tzinfo=timezone("US/Central")),
                      city="Michoacan (MX)")  # end

        for enable_ephemeris in [True]:
            if enable_ephemeris:
                print("Foraging with a circadian mechanism.")
            else:
                print("Foraging without a circadian mechanism.")

            # stats
            d_x = []  # logarithmic distance
            d_c = []
            tau = []  # tortuosity
            ri = 0

            print("Routes: ", end="")
            obs = p1

            for route in routes[::2]:
                net = CX(noise=0., pontin=False)
                net.update = True

                # sun position
                cur = datetime(2018, 6, 21, 10, 0, 0)
                obs.date = cur
                sun.compute(obs)
                theta_s = np.array([np.pi / 2 - sun.alt])
                phi_s = np.array([(sun.az + np.pi) % (2 * np.pi) - np.pi])

                sun_azi = []
                sun_ele = []
                time = []

                # outward route
                route.condition = Hybrid(tau_x=dx)
                oroute = route.reverse()
                x, y, yaw = [(x0, y0, yaw0) for x0, y0, _, yaw0 in oroute][0]
                opath = [[x, y, yaw]]

                v = np.zeros(2)
                tb1 = []
                d_phi = 0.

                theta, phi = compass.dra.theta, compass.dra.phi
                for _, _, _, yaw in oroute:
                    theta_n, phi_n = tilt(theta_t, phi_t, theta, phi + yaw)

                    sun_ele.append(theta_s[0])
                    sun_azi.append(phi_s[0])
                    time.append(cur)
                    sky.theta_s, sky.phi_s = theta_s, phi_s
                    Y, P, A = sky(theta_n, phi_n, noise=noise)

                    if enable_ephemeris:
                        r_tb1, d_phi = encode(theta, phi, Y, P, A, d_phi=d_phi)
                    else:
                        r_tb1, d_phi = encode(theta, phi, Y, P, A, d_phi=0.)
                    yaw0 = yaw
                    _, yaw = np.pi - decode_sph(r_tb1) + phi_s

                    net(yaw, flow)
                    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                    v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx
                    opath.append([opath[-1][0] + v[0], opath[-1][1] + v[1], yaw])
                    tb1.append(net.tb1)

                    cur += delta
                    obs.date = cur
                    sun.compute(obs)
                    theta_s = np.array([np.pi / 2 - sun.alt])
                    phi_s = np.array([(sun.az + np.pi) % (2 * np.pi) - np.pi])
                opath = np.array(opath)

                yaw -= phi_s

                # inward route
                ipath = [[opath[-1][0], opath[-1][1], opath[-1][2]]]
                L = 0.  # straight distance to the nest
                C = 0.  # distance towards the nest that the agent has covered
                SL = 0.
                TC = 0.
                tb1 = []
                tau.append([])
                d_x.append([])
                d_c.append([])

                while C < 15:
                    theta_n, phi_n = tilt(theta_t, phi_t, theta, phi + yaw)

                    sun_ele.append(theta_s[0])
                    sun_azi.append(phi_s[0])
                    time.append(cur)
                    sky.theta_s, sky.phi_s = theta_s, phi_s
                    Y, P, A = sky(theta_n, phi_n, noise=noise)

                    if enable_ephemeris:
                        r_tb1, d_phi = encode(theta, phi, Y, P, A, d_phi=d_phi)
                    else:
                        r_tb1, d_phi = encode(theta, phi, Y, P, A, d_phi=0.)
                    _, yaw = np.pi - decode_sph(r_tb1) + phi_s
                    motor = net(yaw, flow)
                    yaw = (ipath[-1][2] + motor + np.pi) % (2 * np.pi) - np.pi
                    v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx
                    ipath.append([ipath[-1][0] + v[0], ipath[-1][1] + v[1], yaw])
                    tb1.append(net.tb1)
                    L = np.sqrt(np.square(opath[0][0] - ipath[-1][0]) + np.square(opath[0][1] - ipath[-1][1]))
                    C += route.dx
                    d_x[-1].append(L)
                    d_c[-1].append(C)
                    tau[-1].append(L / C)
                    if C <= route.dx:
                        SL = L
                    if TC == 0. and len(d_x[-1]) > 50 and d_x[-1][-1] > d_x[-1][-2]:
                        TC = C

                    cur += delta
                    obs.date = cur
                    sun.compute(obs)
                    theta_s = np.array([np.pi / 2 - sun.alt])
                    phi_s = np.array([(sun.az + np.pi) % (2 * np.pi) - np.pi])

                ipath = np.array(ipath)
                d_x[-1] = np.array(d_x[-1]) / SL * 100
                d_c[-1] = np.array(d_c[-1]) / TC * 100
                tau[-1] = np.array(tau[-1])

                ri += 1

                avg_time += cur - p1.date

                stats["max_alt"].append(0.)
                stats["noise"].append(noise)
                stats["opath"].append(opath)
                stats["ipath"].append(ipath)
                stats["d_x"].append(d_x[-1])
                stats["d_c"].append(d_c[-1])
                stats["tau"].append(tau[-1])
                print(".", end=" ")
            print()
            print("average time:", avg_time / ri)  # 1:16:40

        np.savez_compressed(os.path.join(__data__, "pi-stats-ephem.npz"), **stats)

    if load_stats:
        stats = np.load(os.path.join(__data__, "pi-stats-ephem.npz"), allow_pickle=True)

    if show_routes:
        opaths = stats["opath"]
        ipaths = stats["ipath"]
        for opath, ipath in zip(opaths, ipaths):
            plot_route(opath, ipath)
        plt.show()
