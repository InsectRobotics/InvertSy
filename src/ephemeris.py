from observer import Observer

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

        self._alt = 0.
        self._azi = 0.

        # set the observer of the sun on Earth
        self._obs = None  # type: Observer
        if observer is not None:
            self.compute(observer)

    def compute(self, observer: Observer):
        self.obs = observer
        lon, lat = observer.lon, observer.lat

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
        sn = self._sn = solar_noon(lon, eot, tz=self.obs.timezone)
        self._srt = sunrise_time(hasr, sn)
        self._sst = sunset_time(hasr, sn)
        self._sld = sunlight_duration(hasr)

        tst = true_solar_time(lon, observer.date, eot, tz=self.obs.timezone)
        ha = self._hra = hour_angle(tst)
        sza = solar_zenith_angle(lat, sd, ha)
        sea = self._sea = solar_elevation_angle(sza)
        aar = self._aar = approx_atmospheric_refraction(sea)
        self._alt = solar_elevation_corrected_for_atm_refraction(sea, aar)
        self._azi = solar_azimuth_angle(lat, ha, sza, sd)

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
        if isinstance(value, Observer):
            value.on_change = self.update
            self._obs = value

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
        sky. By definition, the HRA is 0° at solar noon. Since the Earth rotates 15° per hour away from solar noon
        corresponds to an angular motion of the sun in the sky of 15°. In the morning the hour angle is negative, in
        the afternoon the hour angle is positive.
        """
        return self._hra

    @property
    def declination(self):
        """
        The declination angle.
        """
        return self._sd


def julian_day(date):
    return date.toordinal() + 1721424.5 + (date.hour + (date.minute + date.second / 60) / 60) / 24


def julian_century(jd):
    return (jd - 2451545) / 36525


def geom_mean_long_sun(jc):
    return np.deg2rad((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)


def geom_mean_anom_sun(jc):
    return np.deg2rad(357.52911 + jc * (35999.05029 - 0.0001537 * jc))


def eccent_earth_orbit(jc):
    return 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)


def sun_eq_of_ctr(jc, gmas):
    return np.deg2rad(np.sin(gmas) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
                      np.sin(2 * gmas) * (0.019993 - 0.000101 * jc) +
                      np.sin(3 * gmas) * 0.000289)


def sun_true_long(gmls, seoc):
    return gmls + seoc


def sun_true_anom(gmas, seoc):
    return gmas + seoc


def sun_rad_vector(eeo, sta):
    return (1.000001018 * (1 - np.square(eeo))) / (1 + eeo * np.cos(sta))


def sun_app_long(jc, stl):
    return stl - np.deg2rad(0.00569 + 0.00478 * np.sin(np.deg2rad(125.04 - 1934.136 * jc)))


def mean_obliq_ecliptic(jc):
    return np.deg2rad(23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60)


def obliq_corr(jc, moe):
    return moe + np.deg2rad(0.00256) * np.cos(np.deg2rad(125.04 - 1934.136 * jc))


def sun_rt_ascen(sal, oc):
    return np.arctan2(np.cos(oc) * np.sin(sal), np.cos(sal))


def sun_declin(sal, oc):
    return np.arcsin(np.sin(oc) * np.sin(sal))


def var_y(oc):
    return np.square(np.tan(oc / 2))


def eq_of_time(gmls, gmas, eeo, vy):
    return 4 * np.rad2deg(
        vy * np.sin(2 * gmls) -
        2 * eeo * np.sin(gmas) +
        4 * eeo * vy * np.sin(gmas) * np.cos(2 * gmls) -
        0.5 * np.square(vy) * np.sin(4 * gmls) - 1.25 * np.square(eeo) * np.sin(2 * gmas))


def ha_sunrise(lat, sd):
    return np.arccos(np.cos(np.deg2rad(90.833)) / (np.cos(lat) * np.cos(sd)) - np.tan(lat) * np.tan(sd))


def solar_noon(lon, eot, tz=0):
    return (720 - 4 * np.rad2deg(lon) - eot + tz * 60) / 1440


def sunrise_time(hasr, sn):
    return sn - np.rad2deg(hasr) * 4 / 1440


def sunset_time(hasr, sn):
    return sn + np.rad2deg(hasr) * 4 / 1440


def sunlight_duration(hasr):
    return 8 * np.rad2deg(hasr)


def true_solar_time(lon, date, eot, tz=0):
    h = (date.hour + (date.minute + date.second / 60) / 60) / 24
    return (h * 1440 + eot + 4 * np.rad2deg(lon) - 60 * tz) % 1440


def hour_angle(tst):
    return np.deg2rad(tst / 4 + 180 if tst < 0 else tst / 4 - 180)
    # return np.deg2rad(tst / 4 + 180) % (2 * np.pi) - np.pi


def solar_zenith_angle(lat, sd, ha):
    return np.arccos(np.sin(lat) * np.sin(sd) + np.cos(lat) * np.cos(sd) * np.cos(ha))


def solar_elevation_angle(sza):
    return np.pi/2 - sza


def approx_atmospheric_refraction(sea):
    if np.rad2deg(sea) > 85:
        return 0
    elif np.rad2deg(sea) > 5:
        return np.deg2rad((1 / np.tan(sea) - 0.07 / np.power(np.tan(sea), 3) + 0.000086 / np.power(np.tan(sea), 5)) / 3600)
    elif np.rad2deg(sea) > -0.575:
        return np.deg2rad((1735 + sea * (-518.2 - sea * (-518.2 + sea * (103.4 + sea * (-12.79 + sea * 0.711))))) / 3600)
    else:
        return np.deg2rad((-20.772 / np.tan(sea)) / 3600)


def solar_elevation_corrected_for_atm_refraction(sea, aar):
    return sea + aar


def solar_azimuth_angle(lat, ha, sza, sd):
    temp = np.arccos(((np.sin(lat) * np.cos(sza)) - np.sin(sd)) / (np.cos(lat) * np.sin(sza)))
    if ha > 0:
        return (temp + np.pi) % (2 * np.pi)
    else:
        return (np.deg2rad(540) - temp) % (2 * np.pi)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    obs = Observer(lon=np.deg2rad(0), lat=np.deg2rad(42), date=datetime.now())
    sun = Sun(obs)

    plt.figure()
    for c, lat in [['r', np.deg2rad(0)],
                   ['y', np.deg2rad(22.5)],
                   ['g', np.deg2rad(45)],
                   ['b', np.deg2rad(67.5)],
                   ['c', np.deg2rad(89)]]:
        sun.lat = lat
        e, a = [], []
        for h in range(24):
            sun.date = datetime(2020, 9, 21, h, tzinfo=timezone('GMT'))
            e.append(sun.alt)
            a.append(sun.az)

        e, a = np.array(e), np.array(a)

        plt.plot(a, e, '%s.-' % c)
    plt.xlim([0, 2 * np.pi])
    plt.ylim([0, np.pi/2])

    plt.show()

