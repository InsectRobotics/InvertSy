from sphere import sph2vec, vec2sph, angdist, sphdist_angle, sphdist_meters
from ephemeris import Sun

import quaternion as qt

from datetime import datetime, timedelta
from pytz import timezone
from copy import copy

import matplotlib.pyplot as plt
import numpy as np


class Observer(object):
    def __init__(self, lon=None, lat=None, date=datetime.now(), city=None, inrad=True):
        self._lon = float(lon) if inrad else np.deg2rad(float(lon))
        self._lat = float(lat) if inrad else np.deg2rad(float(lat))
        self._date = date
        self._city = city
        h_loc = date.hour
        h_gmt = date.astimezone(timezone("GMT")).hour
        self._tzinfo = date.tzinfo
        self._tz = h_loc - h_gmt
        self.on_change = None
        self.__inrad = inrad

    @property
    def lon(self) -> float:
        """
        The longitude of the observer.
        """
        return self._lon if self.__inrad else np.rad2deg(self._lon)

    @lon.setter
    def lon(self, value: {float, int, str}):
        self._lon = float(value) if self.__inrad else np.deg2rad(float(value))
        if self.on_change is not None:
            self.on_change()

    @property
    def lat(self) -> float:
        """
        The latitude of the observer.
        """
        return self._lat if self.__inrad else np.rad2deg(self._lat)

    @lat.setter
    def lat(self, value: {float, int, str}):
        self._lat = float(value) if self.__inrad else np.deg2rad(float(value))
        if self.on_change is not None:
            self.on_change()

    @property
    def tzgmt(self) -> int:
        return self._tz

    @property
    def timezone(self):
        return self._tzinfo

    @property
    def date(self) -> datetime:
        """
        The date and time in the current position.
        """
        return self._date

    @date.setter
    def date(self, value: datetime):
        self._date = value
        if self.on_change is not None:
            self.on_change()

    @property
    def city(self) -> str:
        return self._city

    def angdist(self, obs2) -> float:
        return get_distance(self, obs2)

    def copy(self):
        return copy(self)

    def __copy__(self):
        return Observer(lon=copy(self.lon), lat=copy(self.lat), inrad=copy(self.__inrad),
                        date=copy(self.date), city=copy(self.city))

    def __repr__(self):
        return "Observer(lon='%.6f', lat='%.6f', %sdate='%s', timezone='%s')" % (
            self.lon, self.lat, ("city='%s', " % self.city) if self._city is not None else "",
            str(self._date), self.timezone)


def get_seville_observer():
    sev = Observer()
    sev.lat = '37.392509'
    sev.lon = '-5.983877'
    sev._city = "Seville"

    return sev


def get_edinburgh_observer():
    edi = Observer()
    edi.lat = '55.946388'
    edi.lon = '-3.200000'
    edi._city = "Edinburgh"

    return edi


def get_live_observer():
    import requests
    import json

    send_url = "http://api.ipstack.com/check?access_key=9d6917440142feeccd73751e2f2124dc"
    geo_req = requests.get(send_url)
    geo_json = json.loads(geo_req.text)

    obs = Observer(lon=geo_json['longitude'], lat=geo_json['latitude'], date=datetime.now())
    obs._city = geo_json['city']
    return obs


def get_distance(obs1: Observer, obs2: Observer):
    v1 = sph2vec(theta=obs1._lat, phi=obs1._lon, zenith=False)
    v2 = sph2vec(theta=obs2._lat, phi=obs2._lon, zenith=False)
    return angdist(v1, v2, zenith=False)


def interpolate(observers_list, breaks_every=3, breaks_duation=30, dt=timedelta(minutes=30), tol=1e-001):

    obss = observers_list[:-1]
    obse = observers_list[1:]
    observers = []
    for obs1, obs2 in zip(obss, obse):
        v1 = sph2vec(theta=obs1._lat, phi=obs1._lon, zenith=False)
        v2 = sph2vec(theta=obs2._lat, phi=obs2._lon, zenith=False)

        temp = obs1.copy()
        travel_duration = 0
        while temp.date <= obs2.date:
            sun = Sun(temp)
            travel_duration += ((sun.sunset - sun.sunrise).total_seconds() / 3600 - 4)
            temp.date += timedelta(days=1)
        sun = Sun(obs1)
        print("Start:", obs1.city, "sunrise:", sun.sunrise, "sunset:", sun.sunset, "daylight:", sun.sunset-sun.sunrise)
        sun.obs = obs2
        print("End:", obs2.city, "sunrise:", sun.sunrise, "sunset:", sun.sunset, "daylight:", sun.sunset-sun.sunrise)
        du = obs2.date - obs1.date
        travel_distance = obs1.angdist(obs2)
        # travel_duration = du.days * travel_hours_per_day
        distance = np.sqrt(np.square(v2 - v1).sum())
        direction = (v2 - v1) / distance
        ang_speed = travel_distance / travel_duration
        uni_speed = distance / travel_duration
        print("Angular distance: %.4f" % np.rad2deg(travel_distance))
        print("Unit distance: %.4f" % distance)
        print("Travel duration: %d hours" % travel_duration)
        print("Speed: %.4f deg/h or %.4f units/h" % (np.rad2deg(ang_speed), uni_speed))
        print("Iterations: %d" % (travel_distance / ang_speed))  # 171, 207, 279
        velocity = direction * uni_speed * (dt.seconds / 3600)

        obs = obs1.copy()
        sun = Sun(obs)
        obs.date = sun.sunrise + timedelta(minutes=30)
        observers.append(obs.copy())
        v = v1
        day = 0
        obs_init = obs.copy()
        dtime = timedelta(seconds=0)
        btime = timedelta(seconds=0)
        while np.rad2deg(angdist(v, v2, zenith=False)) > tol:

            sun = Sun(obs)

            day_count = obs.date.timetuple().tm_yday - obs_init.date.timetuple().tm_yday
            if day > day_count:
                print("Day changed", day, (obs.date - obs_init.date).days, sun.obs.date)
                obs.date += timedelta(hours=12)
                sun = Sun(obs)
                obs.date = sun.sunrise + timedelta(hours=2)

            if day > (obs.date - obs_init.date).days + 1:
                break

            v = sph2vec(theta=obs._lat, phi=obs._lon, zenith=False)
            v += velocity
            lat, lon, _ = vec2sph(v, zenith=False)
            obs = Observer(lon=np.rad2deg(lon), lat=np.rad2deg(lat), inrad=False,
                           date=obs.date + dt)
            observers.append(obs.copy())
            dtime += dt
            btime += dt

            if (btime.total_seconds() / 3600) >= breaks_every:
                obs.date += timedelta(minutes=breaks_duation)
                dtime += timedelta(minutes=breaks_duation)
                btime = timedelta(seconds=0)
                print("BREAK!")

            print(obs)
            if dtime.total_seconds() >= (sun.sunset - sun.sunrise - timedelta(hours=4)).total_seconds():
                day += 1
                dtime = timedelta(seconds=0)
                btime = timedelta(seconds=0)
                print("DAY++")

        print("\n")
    print(observers[0])
    print(observers[-1])
    print(observers_list[-1])
    return observers


def interpolate2():
    hour_travel_per_day = 9
    ps = Observer(lon=-84.628, lat=45.868, inrad=False,
                  date=datetime(2020, 8, 20, 6, tzinfo=timezone("US/Eastern")),
                  city="Mackinac Island (MI)")  # start
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
    print("SLERP")
    # qs = qt.from_euler(yaw=np.deg2rad(ps.lon), pitch=np.pi/2 - np.deg2rad(ps.lat), roll=-direction)
    qs = qt.from_spherical_coords(phi=np.deg2rad(ps.lon), theta=np.deg2rad(ps.lat))
    # qe = qt.from_euler(yaw=np.deg2rad(pe.lon), pitch=np.pi/2 - np.deg2rad(pe.lat), roll=-direction)
    qe = qt.from_spherical_coords(phi=np.deg2rad(pe.lon), theta=np.deg2rad(pe.lat))
    lat, lon = qt.as_spherical_coords(qs).T

    print("P_start: lon=%.8f, lat=%.8f" % (ps.lon, ps.lat))
    print("P_end:   lon=%.8f, lat=%.8f" % (pe.lon, pe.lat))
    print("")
    dist = 0
    loc = []
    while dist < total_distance_ang:
        u = dist / total_distance_ang
        qc = qt.slerp(qs, qe, u)
        lat, lon = qt.as_spherical_coords(qc).T
        loc.append([lon, lat])
        print("p_%.3f: lon=%.8f, lat=%.8f" % (u, np.rad2deg(lon), np.rad2deg(lat)))
        dist += dw_ang
    loc = np.array(loc)
    print("P_end:   lon=%.8f, lat=%.8f" % (pe.lon, pe.lat))

    plt.figure("butterflies-course", figsize=(7.5, 5))
    plt.plot(loc[:, 0], loc[:, 1], 'k.')
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi / 2, np.pi / 2])
    plt.show()


if __name__ == '__main__':
    print(get_live_observer())


# 1.79m sit, 3.48m pet, 2.48m 3yl, 2.18m ore, 2.98m xry
# 1.96m sit, 1.49m pet, 1.86m 3yl, 2.23m ore, 1.12m xry
# 1.56m sit, 3.04m pet, 2.17m 3yl, 1.91m ore, 1.30m xry
# 5.31m sit, 8.01m pet, 6.51m 3yl, 6.24m ore, 5.40m xry
