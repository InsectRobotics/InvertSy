from ephemeris import *
from observer import get_seville_observer
from pytz import timezone

from datetime import datetime

import ephem


def test_julian_day(d=2):
    date = datetime(2010, 6, 21, 0, 6, 0, tzinfo=timezone("US/Central")).astimezone(timezone('GMT'))
    jd = julian_day(date)
    if round(jd, d) != round(2455368.75, d):
        print("test_julian_day: %.2f != 2455368.75" % jd)
    else:
        print("test_julian_day: PASS")


def test_julian_century(d=4):
    jd = 2455368.747916667
    jc = julian_century(jd)
    if round(jc, d) != round(0.10468868, d):
        print("test_julian_century: %.8f != 0.10468868" % jc)
    else:
        print("test_julian_century: PASS")


def test_geom_mean_long_sun(d=3):
    jc = 0.10468868
    gmls = np.rad2deg(geom_mean_long_sun(jc))
    if round(gmls, d) != round(89.33966, d):
        print("test_geom_mean_long_sun: %.5f != 89.33966" % gmls)
    else:
        print("test_geom_mean_long_sun: PASS")


def test_geom_mean_anom_sun(d=3):
    jc = 0.10468868
    gmas = np.rad2deg(geom_mean_anom_sun(jc))
    if round(gmas, d) != round(4126.222, d):
        print("test_geom_mean_anom_sun: %.3f != 4126.222" % gmas)
    else:
        print("test_geom_mean_anom_sun: PASS")


def test_eccent_earth_orbit(d=4):
    jc = 0.10468868
    eeo = eccent_earth_orbit(jc)
    if round(eeo, d) != round(0.016704, d):
        print("test_eccent_earth_orbit: %.6f != 0.016704" % eeo)
    else:
        print("test_eccent_earth_orbit: PASS")


def test_sun_eq_of_ctr(d=4):
    jc = 0.10468868
    gmas = 4126.222
    seoc = np.rad2deg(sun_eq_of_ctr(jc, np.deg2rad(gmas)))
    if round(seoc, d) != round(0.4468, d):
        print("test_sun_eq_of_ctr: %.6f != 0.446800" % seoc)
    else:
        print("test_sun_eq_of_ctr: PASS")


def test_sun_true_long(d=4):
    gmls = 89.33966
    seoc = 0.4468
    stl = np.rad2deg(sun_true_long(np.deg2rad(gmls), np.deg2rad(seoc)))
    if round(stl, d) != round(89.78646, d):
        print("test_sun_true_long: %.5f != 89.78646" % stl)
    else:
        print("test_sun_true_long: PASS")


def test_sun_true_anom(d=3):
    gmas = 4126.222
    seoc = 0.4468
    sta = np.rad2deg(sun_true_anom(np.deg2rad(gmas), np.deg2rad(seoc)))
    if round(sta, d) != round(4126.669, d):
        print("test_sun_true_anom: %.3f != 4126.669" % sta)
    else:
        print("test_sun_true_anom: PASS")


def test_sun_rad_vector(d=4):
    eeo = 0.016704
    sta = 4126.669
    srv = sun_rad_vector(eeo, np.deg2rad(sta))
    if round(srv, d) != round(1.01624, d):
        print("test_sun_rad_vector: %.5f != 1.01624" % srv)
    else:
        print("test_sun_rad_vector: PASS")


def test_sun_app_long(d=4):
    jc = 0.10468868
    stl = 89.78646
    sal = np.rad2deg(sun_app_long(jc, np.deg2rad(stl)))
    if round(sal, d) != round(89.78544, d):
        print("test_sun_app_long: %.5f != 89.78544" % sal)
    else:
        print("test_sun_app_long: PASS")


def test_mean_obliq_ecliptic(d=4):
    jc = 0.10468868
    moe = np.rad2deg(mean_obliq_ecliptic(jc))
    if round(moe, d) != round(23.43793, d):
        print("test_mean_obliq_ecliptic: %.5f != 23.43793" % moe)
    else:
        print("test_mean_obliq_ecliptic: PASS")


def test_obliq_corr(d=4):
    jc = 0.10468868
    moe = 23.43793
    oc = np.rad2deg(obliq_corr(jc, np.deg2rad(moe)))
    if round(oc, d) != round(23.43849, d):
        print("test_obliq_corr: %.5f != 23.43849" % oc)
    else:
        print("test_obliq_corr: PASS")


def test_sun_rt_ascen(d=4):
    sal = 89.78544
    oc = 23.43849
    sra = np.rad2deg(sun_rt_ascen(np.deg2rad(sal), np.deg2rad(oc)))
    if round(sra, d) != round(89.76614, d):
        print("test_sun_rt_ascen: %.5f != 89.76614" % sra)
    else:
        print("test_sun_rt_ascen: PASS")


def test_sun_declin(d=4):
    sal = 89.78544
    oc = 23.43849
    sd = np.rad2deg(sun_declin(np.deg2rad(sal), np.deg2rad(oc)))
    if round(sd, d) != round(23.43831, d):
        print("test_sun_declin: %.5f != 23.43831" % sd)
    else:
        print("test_sun_declin: PASS")


def test_var_y(d=4):
    oc = 23.43849
    vy = var_y(np.deg2rad(oc))
    if round(vy, d) != round(0.043031, d):
        print("test_var_y: %.6f != 0.043031" % vy)
    else:
        print("test_var_y: PASS")


def test_eq_of_time(d=4):
    gmls = 89.33966
    gmas = 4126.222
    eeo = 0.016704
    vy = 0.043031
    eot = eq_of_time(np.deg2rad(gmls), np.deg2rad(gmas), eeo, vy)
    if round(eot, d) != round(-1.7063078, d):
        print("test_eq_of_time: %.7f != -1.7063078" % eot)
    else:
        print("test_eq_of_time: PASS")


def test_ha_sunrise(d=4):
    lat = 40
    sd = 23.43831
    hasr = np.rad2deg(ha_sunrise(np.deg2rad(lat), np.deg2rad(sd)))
    if round(hasr, d) != round(112.6103, d):
        print("test_ha_sunrise: %.4f != 112.6103" % hasr)
    else:
        print("test_ha_sunrise: PASS")


def test_solar_noon(d=2):
    lon = -105
    eot = -1.7063078
    sn = solar_noon(np.deg2rad(lon), eot, tz=-6)
    if round(sn, d) != round(0.54, d):
        print("test_solar_noon: %.2f != 0.54" % sn)
    else:
        print("test_solar_noon: PASS")


def test_sunrise_time(d=4):
    sn = 0.54285
    hasr = 112.6103
    srt = sunrise_time(np.deg2rad(hasr), sn)
    if round(srt, d) != round(0.2300, d):
        print("test_sunrise_time: %.4f != 0.23" % srt)
    else:
        print("test_sunrise_time: PASS")


def test_sunset_time(d=4):
    sn = 0.54285
    hasr = 112.6103
    sst = sunset_time(np.deg2rad(hasr), sn)
    if round(sst, d) != round(0.8557, d):
        print("test_sunset_time: %.4f != 0.8557" % sst)
    else:
        print("test_sunset_time: PASS")


def test_sunlight_duration(d=2):
    hasr = 112.6103
    sld = sunlight_duration(np.deg2rad(hasr))
    if round(sld, d) != round(900.88277, d):
        print("test_sunlight_duration: %.5f != 900.88277" % sld)
    else:
        print("test_sunlight_duration: PASS")


def test_true_solar_time(d=3):
    date = datetime(2010, 6, 21, 0, 6, 0, tzinfo=timezone("US/Central"))  # .astimezone(timezone('GMT'))
    lon = -105
    eot = -1.7063078
    tst = true_solar_time(np.deg2rad(lon), date, eot, tz=-6)  # !!! it doesn't work for automatic time zones
    if round(tst, d) != round(1384.294, d):
        print("test_true_solar_time: %.3f != 1384.294" % tst)
    else:
        print("test_true_solar_time: PASS")


def test_hour_angle(d=2):
    tst = 1384.294
    ha = np.rad2deg(hour_angle(tst))
    if round(ha, d) != round(166.0734, d):
        print("test_hour_angle: %.4f != 166.0734" % ha)
    else:
        print("test_hour_angle: PASS")


def test_solar_zenith_angle(d=2):
    lat = 40
    sd = 23.43831
    ha = 166.0734
    sza = np.rad2deg(solar_zenith_angle(np.deg2rad(lat), np.deg2rad(sd), np.deg2rad(ha)))
    if round(sza, d) != round(115.2457, d):
        print("test_solar_zenith_angle: %.4f != 115.2457" % sza)
    else:
        print("test_solar_zenith_angle: PASS")


def test_solar_elevation_angle(d=2):
    sza = 115.2457
    sea = np.rad2deg(solar_elevation_angle(np.deg2rad(sza)))
    if round(sea, d) != round(-25.24572, d):
        print("test_solar_elevation_angle: %.4f != -25.24572" % sea)
    else:
        print("test_solar_elevation_angle: PASS")


def test_approx_atmospheric_refraction(d=4):
    sea = -25.24572
    aar = np.rad2deg(approx_atmospheric_refraction(np.deg2rad(sea)))
    if round(aar, d) != round(0.012237, d):
        print("test_approx_atmospheric_refraction: %.4f != 0.012237" % aar)
    else:
        print("test_approx_atmospheric_refraction: PASS")


def test_solar_elevation_corrected_for_atm_refraction(d=2):
    sea = -25.24572
    aar = 0.012237
    secar = np.rad2deg(solar_elevation_corrected_for_atm_refraction(np.deg2rad(sea), np.deg2rad(aar)))
    if round(secar, d) != round(-25.23348, d):
        print("test_solar_elevation_corrected_for_atm_refraction: %.4f != -25.23348" % secar)
    else:
        print("test_solar_elevation_corrected_for_atm_refraction: PASS")


def test_solar_azimuth_angle(d=2):
    lat = 40
    ha = 166.0734
    sza = 115.2457
    sd = 23.43831
    ha = np.rad2deg(solar_azimuth_angle(np.deg2rad(lat), np.deg2rad(ha), np.deg2rad(sza), np.deg2rad(sd)))
    if round(ha, d) != round(345.8691, d):
        print("test_hour_angle: %.4f != 345.8691" % ha)
    else:
        print("test_hour_angle: PASS")


if __name__ == '__main___':
    test_julian_day()
    test_julian_century()
    test_geom_mean_long_sun()
    test_geom_mean_anom_sun()
    test_eccent_earth_orbit()
    test_sun_eq_of_ctr()
    test_sun_true_long()
    test_sun_true_anom()
    test_sun_rad_vector()
    test_sun_app_long()
    test_mean_obliq_ecliptic()
    test_obliq_corr()
    test_sun_rt_ascen()
    test_sun_declin()
    test_var_y()
    test_eq_of_time()
    test_ha_sunrise()
    test_solar_noon()
    test_sunrise_time()
    test_sunset_time()
    test_sunlight_duration()
    test_true_solar_time()
    test_hour_angle()
    test_solar_zenith_angle()
    test_solar_elevation_angle()
    test_approx_atmospheric_refraction()
    test_solar_elevation_corrected_for_atm_refraction()
    test_solar_azimuth_angle()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    lon, lat = '-5.983877', '37.392509'
    date = datetime(2020, 6, 10, 12)

    gt_obs = ephem.Observer()
    gt_obs.lat = lat
    gt_obs.lon = lon
    gt_obs.date = date
    gt_sun = ephem.Sun()
    gt_sun.compute(gt_obs)

    my_obs = Observer(lon=np.deg2rad(float(lon)), lat=np.deg2rad(float(lat)), date=date)
    my_sun = Sun(observer=my_obs)

    print("GTEphem alt=%.4f azi=%.4f" % (gt_sun.alt, gt_sun.az))
    print("MyEphem alt=%.4f azi=%.4f" % (my_sun.alt, my_sun.az))

    my_loc, gt_loc = [], []
    for h in range(0, 24):
        gt_obs.date = my_obs.date = datetime(2020, 9, 10, h)
        gt_sun.compute(gt_obs)
        my_sun.compute(my_obs)

        gt_loc.append((gt_sun.alt, gt_sun.az))
        my_loc.append((my_sun.alt, my_sun.az))
    gt_loc = np.array(gt_loc)
    my_loc = np.array(my_loc)

    plt.figure("Ephemeris test")
    plt.plot(gt_loc[:, 1], gt_loc[:, 0], 'g.-')
    plt.plot(my_loc[:, 1] % (2 * np.pi), my_loc[:, 0], 'b.-')
    plt.xlim([0, 2 * np.pi])
    plt.ylim([-np.pi/2, np.pi/2])

    plt.show()
