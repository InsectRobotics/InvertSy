from ephemeris import *
from observer import get_seville_observer
from pytz import timezone

from datetime import datetime

import ephem


class TestJulianDay:
    def test_julian_day(self, d=2):
        date = datetime(2010, 6, 21, 0, 6, 0, tzinfo=timezone("US/Central")).astimezone(timezone('GMT'))
        jd = julian_day(date)
        assert round(jd, d) == round(2455368.75, d)


class TestJulianCentury:
    def test_julian_century(self, d=4):
        jd = 2455368.747916667
        jc = julian_century(jd)
        assert round(jc, d) == round(0.10468868, d)


class TestGeomMeanLongSun:
    def test_geom_mean_long_sun(self, d=3):
        jc = 0.10468868
        gmls = np.rad2deg(geom_mean_long_sun(jc))
        assert round(gmls, d) == round(89.33966, d)


class TestGeomMeanAnomSun:
    def test_geom_mean_anom_sun(self, d=3):
        jc = 0.10468868
        gmas = np.rad2deg(geom_mean_anom_sun(jc))
        assert round(gmas, d) == round(4126.222, d)


class TestEccentEarthOrbit:
    def test_eccent_earth_orbit(self, d=4):
        jc = 0.10468868
        eeo = eccent_earth_orbit(jc)
        assert round(eeo, d) == round(0.016704, d)


class TestSunEqOfCtr:
    def test_sun_eq_of_ctr(self, d=4):
        jc = 0.10468868
        gmas = 4126.222
        seoc = np.rad2deg(sun_eq_of_ctr(jc, np.deg2rad(gmas)))
        assert round(seoc, d) == round(0.4468, d)


class TestSunTrueLong:
    def test_sun_true_long(self, d=4):
        gmls = 89.33966
        seoc = 0.4468
        stl = np.rad2deg(sun_true_long(np.deg2rad(gmls), np.deg2rad(seoc)))
        assert round(stl, d) == round(89.78646, d)


class TestSunTrueAnom:
    def test_sun_true_anom(self, d=3):
        gmas = 4126.222
        seoc = 0.4468
        sta = np.rad2deg(sun_true_anom(np.deg2rad(gmas), np.deg2rad(seoc)))
        assert round(sta, d) == round(4126.669, d)


class TestSunRadVector:
    def test_sun_rad_vector(self, d=4):
        eeo = 0.016704
        sta = 4126.669
        srv = sun_rad_vector(eeo, np.deg2rad(sta))
        assert round(srv, d) == round(1.01624, d)


class TestSunAppLong:
    def test_sun_app_long(self, d=4):
        jc = 0.10468868
        stl = 89.78646
        sal = np.rad2deg(sun_app_long(jc, np.deg2rad(stl)))
        assert round(sal, d) == round(89.78544, d)


class TestMeanObliqEcliptic:
    def test_mean_obliq_ecliptic(self, d=4):
        jc = 0.10468868
        moe = np.rad2deg(mean_obliq_ecliptic(jc))
        assert round(moe, d) == round(23.43793, d)


class TestObliqCorr:
    def test_obliq_corr(self, d=4):
        jc = 0.10468868
        moe = 23.43793
        oc = np.rad2deg(obliq_corr(jc, np.deg2rad(moe)))
        assert round(oc, d) == round(23.43849, d)


class TestSunRtAscen:
    def test_sun_rt_ascen(self, d=4):
        sal = 89.78544
        oc = 23.43849
        sra = np.rad2deg(sun_rt_ascen(np.deg2rad(sal), np.deg2rad(oc)))
        assert round(sra, d) == round(89.76614, d)


class TestSunDeclin:
    def test_sun_declin(self, d=4):
        sal = 89.78544
        oc = 23.43849
        sd = np.rad2deg(sun_declin(np.deg2rad(sal), np.deg2rad(oc)))
        assert round(sd, d) == round(23.43831, d)


class TestVarY:
    def test_var_y(self, d=4):
        oc = 23.43849
        vy = var_y(np.deg2rad(oc))
        assert round(vy, d) == round(0.043031, d)


class TestEqOfTime:
    def test_eq_of_time(self, d=4):
        gmls = 89.33966
        gmas = 4126.222
        eeo = 0.016704
        vy = 0.043031
        eot = eq_of_time(np.deg2rad(gmls), np.deg2rad(gmas), eeo, vy)
        assert round(eot, d) == round(-1.7063078, d)


class TestHaSunrise:
    def test_ha_sunrise(self, d=4):
        lat = 40
        sd = 23.43831
        hasr = np.rad2deg(ha_sunrise(np.deg2rad(lat), np.deg2rad(sd)))
        assert round(hasr, d) == round(112.6103, d)


class TestSolarNoon:
    def test_solar_noon(self, d=2):
        lon = -105
        eot = -1.7063078
        sn = solar_noon(np.deg2rad(lon), eot, tz=-6)
        assert round(sn, d) == round(0.54, d)


class TestSunriseTime:
    def test_sunrise_time(self, d=4):
        sn = 0.54285
        hasr = 112.6103
        srt = sunrise_time(np.deg2rad(hasr), sn)
        assert round(srt, d) == round(0.2300, d)


class TestSunsetTime:
    def test_sunset_time(self, d=4):
        sn = 0.54285
        hasr = 112.6103
        sst = sunset_time(np.deg2rad(hasr), sn)
        assert round(sst, d) == round(0.8557, d)


class TestSunlightDuration:
    def test_sunlight_duration(self, d=2):
        hasr = 112.6103
        sld = sunlight_duration(np.deg2rad(hasr))
        assert round(sld, d) == round(900.88277, d)


class TestTrueSolarTime:
    def test_true_solar_time(self, d=3):
        date = datetime(2010, 6, 21, 0, 6, 0, tzinfo=timezone("US/Central"))  # .astimezone(timezone('GMT'))
        lon = -105
        eot = -1.7063078
        tst = true_solar_time(np.deg2rad(lon), date, eot, tz=-6)  # !!! it doesn't work for automatic time zones
        assert round(tst, d) == round(1384.294, d)


class TestHourAngle:
    def test_hour_angle(self, d=2):
        tst = 1384.294
        ha = np.rad2deg(hour_angle(tst))
        assert round(ha, d) == round(166.0734, d)


class TestSolarZenithAngle:
    def test_solar_zenith_angle(self, d=2):
        lat = 40
        sd = 23.43831
        ha = 166.0734
        sza = np.rad2deg(solar_zenith_angle(np.deg2rad(lat), np.deg2rad(sd), np.deg2rad(ha)))
        assert round(sza, d) == round(115.2457, d)


class TestSolarElevationAngle:
    def test_solar_elevation_angle(self, d=2):
        sza = 115.2457
        sea = np.rad2deg(solar_elevation_angle(np.deg2rad(sza)))
        assert round(sea, d) == round(-25.24572, d)


class TestApproxAtmosphericRefraction:
    def test_approx_atmospheric_refraction(self, d=4):
        sea = -25.24572
        aar = np.rad2deg(approx_atmospheric_refraction(np.deg2rad(sea)))
        assert round(aar, d) == round(0.012237, d)


class TestSolarElevationCorrectedForAtmRefraction:
    def test_solar_elevation_corrected_for_atm_refraction(self, d=2):
        sea = -25.24572
        aar = 0.012237
        secar = np.rad2deg(solar_elevation_corrected_for_atm_refraction(np.deg2rad(sea), np.deg2rad(aar)))
        assert round(secar, d) == round(-25.23348, d)


class TestSolarAzimuthAngle:
    def test_solar_azimuth_angle(self, d=2):
        lat = 40
        ha = 166.0734
        sza = 115.2457
        sd = 23.43831
        ha = np.rad2deg(solar_azimuth_angle(np.deg2rad(lat), np.deg2rad(ha), np.deg2rad(sza), np.deg2rad(sd)))
        assert round(ha, d) == round(345.8691, d)


# if __name__ == '__main__':
#     unittest.main()


#
#
# if __name__ == '__main___':
#     test_julian_day()
#     test_julian_century()
#     test_geom_mean_long_sun()
#     test_geom_mean_anom_sun()
#     test_eccent_earth_orbit()
#     test_sun_eq_of_ctr()
#     test_sun_true_long()
#     test_sun_true_anom()
#     test_sun_rad_vector()
#     test_sun_app_long()
#     test_mean_obliq_ecliptic()
#     test_obliq_corr()
#     test_sun_rt_ascen()
#     test_sun_declin()
#     test_var_y()
#     test_eq_of_time()
#     test_ha_sunrise()
#     test_solar_noon()
#     test_sunrise_time()
#     test_sunset_time()
#     test_sunlight_duration()
#     test_true_solar_time()
#     test_hour_angle()
#     test_solar_zenith_angle()
#     test_solar_elevation_angle()
#     test_approx_atmospheric_refraction()
#     test_solar_elevation_corrected_for_atm_refraction()
#     test_solar_azimuth_angle()
#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     lon, lat = '-101.649200', '19.646402'
#     date = datetime(2020, 11, 5, 12)
#
#     gt_obs = ephem.Observer()
#     gt_obs.lat = lat
#     gt_obs.lon = lon
#     gt_obs.date = date
#     gt_sun = ephem.Sun()
#     gt_sun.compute(gt_obs)
#
#     my_obs = Observer(lon=np.deg2rad(float(lon)), lat=np.deg2rad(float(lat)), date=date)
#     my_sun = Sun(observer=my_obs)
#
#     print("GTEphem alt=%.4f azi=%.4f" % (gt_sun.alt, gt_sun.az))
#     print("MyEphem alt=%.4f azi=%.4f" % (my_sun.alt, my_sun.az))
#
#     my_loc, gt_loc = [], []
#     for h in range(0, 24):
#         gt_obs.date = my_obs.date = datetime(2020, 9, 10, h)
#         gt_sun.compute(gt_obs)
#         my_sun.compute(my_obs)
#
#         gt_loc.append((gt_sun.alt, gt_sun.az))
#         my_loc.append((my_sun.alt, my_sun.az))
#     gt_loc = np.array(gt_loc)
#     my_loc = np.array(my_loc)
#
#     plt.figure("Ephemeris test")
#     plt.plot(gt_loc[:, 1], gt_loc[:, 0], 'g.-')
#     plt.plot(my_loc[:, 1] % (2 * np.pi), my_loc[:, 0], 'b.-')
#     plt.xlim([0, 2 * np.pi])
#     plt.ylim([-np.pi/2, np.pi/2])
#
#     plt.show()
