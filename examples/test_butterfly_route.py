from ephemeris import Sun
from observer import Observer, get_distance
from sphere import sphdist_angle, sphdist_meters
from utils import eps

from datetime import datetime, timedelta
from pytz import timezone

import quaternion as qt

import numpy as np
import matplotlib.pyplot as plt
import os

__root__ = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
__data__ = os.path.join(__root__, "data", "ephemeris_routes")

_rel_drift = 0.


def no_circmech(observer, yaw_sun, delta_x, delta_t):
    return get_new_location(observer, delta_x, delta_t, yaw_sun=yaw_sun, correction=np.pi)


def constant_circmech(observer, yaw_sun, delta_x, delta_t):
    h = (observer.date.hour + sun.hour_angle) + observer.date.minute / 60 + observer.date.second / 3600
    return get_new_location(observer, delta_x, delta_t, yaw_sun=yaw_sun, correction=np.deg2rad(15 * (h - 12)) - np.pi)


def relative_circmech(observer, yaw_sun, delta_x, delta_t):
    global _rel_drift

    sun = Sun(observer)
    if observer.date.hour < sun.sunrise.hour + 3:
        _rel_drift = sun.az
    else:
        _rel_drift += np.deg2rad(np.exp((np.rad2deg(sun.alt) - 36) / 10) + 9) * (delta_t.total_seconds() / 3600)
    return get_new_location(observer, delta_x, delta_t, yaw_sun=yaw_sun, correction=_rel_drift)


def hour_angle_circmech(observer, yaw_sun, delta_x, delta_t):
    sun = Sun(observer)
    return get_new_location(observer, delta_x, delta_t, yaw_sun=yaw_sun, correction=sun.hour_angle - np.pi)


def perfect_circmech(observer, yaw_sun, delta_x, delta_t):
    sun = Sun(observer)
    return get_new_location(observer, delta_x, delta_t, yaw_sun=yaw_sun, correction=sun.az)


def get_new_location(observer, delta_x, delta_t, yaw_sun, correction=None):
    o_lat, o_lon = np.array([observer.lat, observer.lon])
    sun = Sun(observer)

    if correction is None:
        correction = 0.

    if not isinstance(yaw_sun, np.ndarray):
        o_lat += delta_x * np.cos(yaw_sun + sun.az - correction)
        o_lon += delta_x * np.sin(yaw_sun + sun.az - correction)
    else:
        dq = qt.invert(yaw_sun)
        # l_lat, l_lon = np.array([np.pi / 2, 0]) + np.array([-1, 1]) * qt.as_spherical_coords(dq)[0]
        # yaw_sun = np.arctan2(l_lat, l_lon)
        # delta_x = np.sqrt(np.sum(np.square([l_lat, l_lon])))
        # l_lat = delta_x * np.cos(yaw_sun + sun.az - correction)
        # l_lon = delta_x * np.sin(yaw_sun + sun.az - correction)
        # nq = qt.from_spherical_coords(np.pi/2 - np.deg2rad(l_lat), np.deg2rad(l_lon))[0]
        oq = qt.from_spherical_coords(np.deg2rad(o_lat), np.deg2rad(o_lon))[0]
        noq = qt.rotate_quaternion(oq, dq)
        o_lat, o_lon = np.rad2deg(qt.as_spherical_coords(noq))

    return Observer(lat=o_lat, lon=o_lon, inrad=False, date=observer.date + delta_t)


if __name__ == '__main__':
    build_stats = True
    load_stats = False
    show_routes = False

    if build_stats:

        hour_travel_per_day = 9  # hours
        breaks_every = 3  # hours
        breaks_duration = 30  # min

        tol = 5e-01  # meters
        dt = 30  # minutes

        ps = Observer(lon=-84.628, lat=45.868, inrad=False,
                      date=datetime(2020, 8, 20, 6, tzinfo=timezone("US/Eastern")),
                      city="Mackinac Island (MI)")  # start
        p2 = Observer(lon=-89.963, lat=39.569, inrad=False,
                      date=datetime(2020, 9, 8, 12, tzinfo=timezone("US/Central")),
                      city="Illinois (IL)")  # stop 1
        p3 = Observer(lon=-96.314445, lat=30.601389, inrad=False,
                      date=datetime(2020, 10, 1, 12, tzinfo=timezone("US/Central")),
                      city="College Station (TX)")  # stop 2
        pe = Observer(lon=-101.649200, lat=19.646402, inrad=False,
                      date=datetime(2020, 11, 1, 18, tzinfo=timezone("US/Central")),
                      city="Michoacan (MX)")  # end

        noise = 0.
        rho = 6371e+03  # meters
        total_distance_ang = sphdist_angle(np.deg2rad(ps.lat), np.deg2rad(ps.lon),
                                           np.deg2rad(pe.lat), np.deg2rad(pe.lon), zenith=False)  # in rad
        total_distance_met = sphdist_meters(np.deg2rad(ps.lat), np.deg2rad(ps.lon),
                                            np.deg2rad(pe.lat), np.deg2rad(pe.lon), rho=rho, zenith=False)  # in meter
        direction_v = np.array([pe.lon - ps.lon, pe.lat - ps.lat]) / total_distance_ang
        direction = np.arctan2(direction_v[0], direction_v[1])
        days_travel = pe.date.timetuple().tm_yday - ps.date.timetuple().tm_yday
        total_hours_travel = hour_travel_per_day * days_travel
        dx = total_distance_met / total_hours_travel
        dw_ang = total_distance_ang / total_hours_travel

        print("PS: lat=%.8f, lon=%.8f" % (ps.lat, ps.lon))
        print("PE: lat=%.8f, lon=%.8f" % (pe.lat, pe.lon))
        print("")
        print("total distance ang: %.2f deg" % np.rad2deg(total_distance_ang))
        print("total distance met: %.2f km" % (total_distance_met / 1000))
        print("direction: lon=%.4f, lat=%.4f" % tuple(direction_v))
        print("direction: %.4f" % np.rad2deg(direction))
        print("days travel: %d" % days_travel)
        print("total time travel: %d hours" % total_hours_travel)
        print("speed: %.4f km/hour" % (dx / 1000))
        print("angular speed: %.4f deg/hour" % np.rad2deg(dw_ang))
        print("")

        dx = np.rad2deg(dw_ang / (dt / 60))  # deg
        delta = timedelta(minutes=dt)
        sun = Sun()
        avg_time = timedelta(0.)

        # sun position

        paths = {}

        for circadian_mechanism in [no_circmech, constant_circmech, relative_circmech, hour_angle_circmech, perfect_circmech]:
            if circadian_mechanism is not None:
                print("Foraging with a circadian mechanism:", circadian_mechanism.__name__)
            else:
                print("Foraging without a circadian mechanism.")

            obs = ps
            # s3d = qt.from_spherical_coords(np.deg2rad(ps.lat), np.deg2rad(ps.lon))[0]
            # e3d = qt.from_spherical_coords(np.deg2rad(pe.lat), np.deg2rad(pe.lon))[0]
            # print("S3D:", s3d, np.rad2deg(qt.as_spherical_coords(s3d)))
            # print("E3D:", e3d, np.rad2deg(qt.as_spherical_coords(e3d)))
            # m3d = qt.slerp(s3d, e3d, u=(dt / 60) / total_hours_travel)
            # print("M3D:", m3d, np.rad2deg(qt.as_spherical_coords(m3d)))
            #
            # direction_q = qt.mul(qt.invert(s3d), m3d)
            # print("D3D:", direction_q, np.rad2deg(qt.as_spherical_coords(direction_q)))
            #
            # d_temp = s3d
            # for i in range(int(total_hours_travel / (dt/60))):
            #     print("Q:", d_temp, "S:", np.rad2deg(qt.as_spherical_coords(d_temp)))
            #     d_temp = qt.rotate_quaternion(d_temp, direction_q)
            #
            # print("E3D:", e3d, np.rad2deg(qt.as_spherical_coords(e3d)))
            # quit()
            sun.compute(obs)
            sdir = direction - sun.az

            ha = []
            observers = [obs.copy()]
            day = 0
            dtime = timedelta(seconds=0)
            btime = timedelta(seconds=0)

            while np.rad2deg(get_distance(obs, pe)) > tol and obs.date < pe.date:
                day_count = obs.date.timetuple().tm_yday - ps.date.timetuple().tm_yday
                if day > day_count:
                    print("Day changed", day, (obs.date - ps.date).days, sun.obs.date)
                    obs.date += timedelta(hours=12)
                    sun = Sun(obs)
                    obs.date = sun.sunrise + timedelta(hours=2)

                if day > (obs.date - ps.date).days + 1:
                    break

                if circadian_mechanism is not None:
                    obs = circadian_mechanism(obs, direction, dx, delta)
                    # obs = circadian_mechanism(obs, direction_q, dx, delta)

                observers.append(obs.copy())
                ha.append([obs.date, sun.hour_angle])

                dtime += delta
                btime += delta

                if (btime.total_seconds() / 3600) >= breaks_every:
                    obs.date += timedelta(minutes=breaks_duration)
                    dtime += timedelta(minutes=breaks_duration)
                    btime = timedelta(seconds=0)
                    print("BREAK!")

                print(obs)
                if dtime.total_seconds() >= (sun.sunset - sun.sunrise - timedelta(hours=4)).total_seconds():
                    day += 1
                    dtime = timedelta(seconds=0)
                    btime = timedelta(seconds=0)
                    print("DAY++")
            paths[circadian_mechanism] = observers
        print()

        plt.figure("circadian-mechanisms", figsize=(5, 6))
        for i, circadian_mechanism in enumerate(paths):

            lon, lat = [], []
            for obs in paths[circadian_mechanism]:
                lon.append(obs.lon)
                lat.append(obs.lat)
            plt.plot(lon, lat, 'C%d-' % (4 + i), label=circadian_mechanism.__name__)
        plt.plot(ps.lon, ps.lat, 'C0o', label=ps.city)
        plt.plot(p2.lon, p2.lat, 'C1o', label=p2.city)
        plt.plot(p3.lon, p3.lat, 'C2o', label=p3.city)
        plt.plot(pe.lon, pe.lat, 'C3o', label=pe.city)
        plt.xticks([-100, -90, -80])
        plt.yticks([20, 30, 40, 50])
        plt.xlim([-105, -80])
        plt.ylim([10, 50])
        plt.grid()
        plt.legend()
        plt.show()
