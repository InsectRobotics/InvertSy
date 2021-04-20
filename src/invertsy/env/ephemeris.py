__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from datetime import datetime
from pytz import timezone

import numpy as np


class Sun(object):
    def __init__(self, observer=None):
        self._jd = 0.
        self._srv = 0.
        self._sd = 0.
        self._eot = 0.
        self._sn = 0.
        self._srt = 0.
        self._sst = 0.
        self._sld = 0.
        self._sea = 0.
        self._aar = 0.
        self._hra = 0.
        self._tst = 0.

        self._alt = 0.
        self._azi = 0.
        self._is_ready = False

        # set the observer of the sun on Earth
        self._obs = None
        if observer is not None:
            self.compute(observer)

    def compute(self, observer):
        self.obs = observer
        lon, lat = observer._lon, observer._lat

        jd = self._jd = julian_day(observer.date)
        jc = julian_century(jd)

        gmls = geom_mean_long_sun(jc)
        gmas = geom_mean_anom_sun(jc)
        eeo = eccent_earth_orbit(jc)
        seoc = sun_eq_of_ctr(jc, gmas)
        stl = sun_true_long(gmls, seoc)
        sta = sun_true_anom(gmas, seoc)
        self._srv = sun_rad_vector(eeo, sta)

        sal = sun_app_long(jc, stl)
        moe = mean_obliq_ecliptic(jc)
        oc = obliq_corr(jc, moe)
        sra = sun_rt_ascen(sal, oc)
        sd = self._sd = sun_declin(sal, oc)

        vy = var_y(oc)
        eot = self._eot = eq_of_time(gmls, gmas, eeo, vy)

        hasr = ha_sunrise(lat, sd)
        sn = self._sn = solar_noon(lon, eot, tz=self.obs.tzgmt)
        self._srt = sunrise_time(hasr, sn)
        self._sst = sunset_time(hasr, sn)
        self._sld = sunlight_duration(hasr)

        tst = self._tst = true_solar_time(lon, observer.date, eot, tz=self.obs.tzgmt)
        ha = self._hra = hour_angle(tst)
        sza = solar_zenith_angle(lat, sd, ha)
        sea = self._sea = solar_elevation_angle(sza)
        aar = self._aar = approx_atmospheric_refraction(sea)
        self._alt = solar_elevation_corrected_for_atm_refraction(sea, aar)
        self._azi = solar_azimuth_angle(lat, ha, sza, sd)

        self._is_ready = True

    def update(self):
        assert self.obs is not None, (
            "Observer has not been set. Please set the observer before you update the sun position."
        )

        self.compute(self.obs)

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        value.on_change = self.update
        self._obs = value
        self._is_ready = False

    @property
    def alt(self):
        """
        The altitude of the sun (rads). Solar elevation (altitude) corrected for atmospheric refraction.
        """
        return self._alt

    @property
    def az(self):
        """
        The azimuth of the sun (rads). Clockwise from North.
        """
        return self._azi

    @property
    def zenith_angle(self):
        return np.pi/2 - self._alt

    @property
    def equation_of_time(self):
        """
        The Equation of Time (EoT) (in minutes) is an empirical equation that corrects for the eccentricity of the
        Earth's orbit and the Earth's axial tilt
        """
        return self._eot

    @property
    def solar_elevation_angle(self):
        """
        Solar elevation without the correction for atmospheric refraction.
        """
        return self._sea

    @property
    def approximate_atmospheric_refraction(self):
        return self._aar

    @property
    def hour_angle(self):
        """
        The Hour Angle converts the local solar time (LST) into the number of degrees which the sun moves across the
        env. By definition, the HRA is 0° at solar noon. Since the Earth rotates 15° per hour away from solar noon
        corresponds to an angular motion of the sun in the env of 15°. In the morning the hour angle is negative, in
        the afternoon the hour angle is positive.
        """
        return self._hra

    @property
    def declination(self):
        """
        The declination angle.
        """
        return self._sd

    @property
    def sunrise(self):
        return relative_to_absolute_time(self._obs, self._srt)

    @property
    def sunset(self):
        return relative_to_absolute_time(self._obs, self._sst)

    @property
    def is_ready(self):
        return self._is_ready


def julian_day(date: datetime):
    return date.toordinal() + 1721424.5 + (date.hour + (date.minute + date.second / 60) / 60) / 24


def julian_century(jd: float):
    return (jd - 2451545) / 36525


def geom_mean_long_sun(jc: float):
    return np.deg2rad((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)


def geom_mean_anom_sun(jc: float):
    return np.deg2rad(357.52911 + jc * (35999.05029 - 0.0001537 * jc))


def eccent_earth_orbit(jc: float):
    return 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)


def sun_eq_of_ctr(jc: float, gmas: float):
    return np.deg2rad(np.sin(gmas) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
                      np.sin(2 * gmas) * (0.019993 - 0.000101 * jc) +
                      np.sin(3 * gmas) * 0.000289)


def sun_true_long(gmls: float, seoc: float):
    return gmls + seoc


def sun_true_anom(gmas: float, seoc: float):
    return gmas + seoc


def sun_rad_vector(eeo: float, sta: float):
    return (1.000001018 * (1 - np.square(eeo))) / (1 + eeo * np.cos(sta))


def sun_app_long(jc: float, stl: float):
    return stl - np.deg2rad(0.00569 + 0.00478 * np.sin(np.deg2rad(125.04 - 1934.136 * jc)))


def mean_obliq_ecliptic(jc: float):
    return np.deg2rad(23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60)


def obliq_corr(jc: float, moe: float):
    return moe + np.deg2rad(0.00256) * np.cos(np.deg2rad(125.04 - 1934.136 * jc))


def sun_rt_ascen(sal: float, oc: float):
    return np.arctan2(np.cos(oc) * np.sin(sal), np.cos(sal))


def sun_declin(sal: float, oc: float):
    return np.arcsin(np.sin(oc) * np.sin(sal))


def var_y(oc: float):
    return np.square(np.tan(oc / 2))


def eq_of_time(gmls: float, gmas: float, eeo: float, vy: float):
    return 4 * np.rad2deg(
        vy * np.sin(2 * gmls) -
        2 * eeo * np.sin(gmas) +
        4 * eeo * vy * np.sin(gmas) * np.cos(2 * gmls) -
        0.5 * np.square(vy) * np.sin(4 * gmls) - 1.25 * np.square(eeo) * np.sin(2 * gmas))


def ha_sunrise(lat: float, sd: float):
    return np.arccos(np.clip(np.cos(np.deg2rad(90.833)) / (np.cos(lat) * np.cos(sd)) - np.tan(lat) * np.tan(sd), -1, 1))


def solar_noon(lon: float, eot: float, tz: int = 0):
    return (720 - 4 * np.rad2deg(lon) - eot + tz * 60) / 1440


def sunrise_time(hasr: float, sn: float):
    return sn - np.rad2deg(hasr) * 4 / 1440


def sunset_time(hasr: float, sn: float):
    return sn + np.rad2deg(hasr) * 4 / 1440


def sunlight_duration(hasr: float):
    return 8 * np.rad2deg(hasr)


def true_solar_time(lon: float, date: datetime, eot: float, tz: int = 0):
    h = (date.hour + (date.minute + date.second / 60) / 60) / 24
    return (h * 1440 + eot + 4 * np.rad2deg(lon) - 60 * tz) % 1440


def hour_angle(tst: float):
    return np.deg2rad(tst / 4 + 180 if tst < 0 else tst / 4 - 180)
    # return np.deg2rad(tst / 4 + 180) % (2 * np.pi) - np.pi


def solar_zenith_angle(lat: float, sd: float, ha: float):
    return np.arccos(np.sin(lat) * np.sin(sd) + np.cos(lat) * np.cos(sd) * np.cos(ha))


def solar_elevation_angle(sza: float):
    return np.pi/2 - sza


def approx_atmospheric_refraction(sea: float):
    if np.rad2deg(sea) > 85:
        return 0
    elif np.rad2deg(sea) > 5:
        return np.deg2rad((1 / np.tan(sea) - 0.07 / np.power(np.tan(sea), 3) + 0.000086 / np.power(np.tan(sea), 5)) / 3600)
    elif np.rad2deg(sea) > -0.575:
        return np.deg2rad((1735 + sea * (-518.2 - sea * (-518.2 + sea * (103.4 + sea * (-12.79 + sea * 0.711))))) / 3600)
    else:
        return np.deg2rad((-20.772 / np.tan(sea)) / 3600)


def solar_elevation_corrected_for_atm_refraction(sea: float, aar: float):
    return sea + aar


def solar_azimuth_angle(lat: float, ha: float, sza: float, sd: float):
    temp = np.arccos(((np.sin(lat) * np.cos(sza)) - np.sin(sd)) / (np.cos(lat) * np.sin(sza)))
    if ha > 0:
        return (temp + np.pi) % (2 * np.pi)
    else:
        return (np.deg2rad(540) - temp) % (2 * np.pi)


def relative_to_absolute_time(obs, time):
    h = (time % 1) * 24
    m = (h - int(h)) * 60
    s = (m - int(m)) * 60
    return datetime(year=obs.date.year, month=obs.date.month, day=obs.date.day,
                    hour=int(h), minute=int(m), second=int(s), tzinfo=obs.timezone)


# if __name__ == '__main__':
#     from observer import Observer
#
#     import matplotlib.pyplot as plt
#
#     obs = Observer(lon=np.deg2rad(0), lat=np.deg2rad(42), date=datetime.now())
#     sun = Sun(obs)
#
#     plt.figure()
#     for c, lat in [['r', np.deg2rad(0)],
#                    ['y', np.deg2rad(22.5)],
#                    ['g', np.deg2rad(45)],
#                    ['b', np.deg2rad(67.5)],
#                    ['c', np.deg2rad(89)]]:
#         sun.lat = lat
#         e, a = [], []
#         for h in range(24):
#             sun.date = datetime(2020, 9, 21, h, tzinfo=timezone('GMT'))
#             e.append(sun.alt)
#             a.append(sun.az)
#
#         e, a = np.array(e), np.array(a)
#
#         plt.plot(a, e, '%s.-' % c)
#     plt.xlim([0, 2 * np.pi])
#     plt.ylim([0, np.pi/2])
#
#     plt.show()
#
