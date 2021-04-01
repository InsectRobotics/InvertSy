from env import Sky
from env.ephemeris import Sun
from env.observer import Observer, get_distance, interpolate
from sphere import angdist, sph2vec, vec2sph

from datetime import datetime, timedelta
from pytz import timezone
# from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
import numpy as np
import sys


p1 = Observer(lon=-84.628, lat=45.868, inrad=False,
              date=datetime(2020, 8, 20, 12, tzinfo=timezone("US/Eastern")),
              city="Mackinac Island (MI)")  # start

p12 = Observer(lon=-84.628, lat=45.868, inrad=False,
               date=datetime(2020, 9, 8, 12, tzinfo=timezone("US/Central")),
               city="Mackinac Island (MI)")  # start

p13 = Observer(lon=-84.628, lat=45.868, inrad=False,
               date=datetime(2020, 10, 1, 12, tzinfo=timezone("US/Central")),
               city="Mackinac Island (MI)")  # start

p14 = Observer(lon=-84.628, lat=45.868, inrad=False,
               date=datetime(2020, 11, 1, 12, tzinfo=timezone("US/Central")),
               city="Mackinac Island (MI)")  # start
p2 = Observer(lon=-89.963, lat=39.569, inrad=False,
              date=datetime(2020, 9, 8, 12, tzinfo=timezone("US/Central")),
              city="Illinois (IL)")  # stop 1
p21 = Observer(lon=-89.963, lat=39.569, inrad=False,
               date=datetime(2020, 8, 20, 12, tzinfo=timezone("US/Eastern")),
               city="Illinois (IL)")  # stop 1
p23 = Observer(lon=-89.963, lat=39.569, inrad=False,
               date=datetime(2020, 10, 1, 12, tzinfo=timezone("US/Central")),
               city="Illinois (IL)")  # stop 1
p24 = Observer(lon=-89.963, lat=39.569, inrad=False,
               date=datetime(2020, 11, 1, 12, tzinfo=timezone("US/Central")),
               city="Illinois (IL)")  # stop 1
p3 = Observer(lon=-96.314445, lat=30.601389, inrad=False,
              date=datetime(2020, 10, 1, 12, tzinfo=timezone("US/Central")),
              city="College Station (TX)")  # stop 2
p31 = Observer(lon=-96.314445, lat=30.601389, inrad=False,
               date=datetime(2020, 8, 20, 12, tzinfo=timezone("US/Eastern")),
               city="College Station (TX)")  # stop 2
p32 = Observer(lon=-96.314445, lat=30.601389, inrad=False,
               date=datetime(2020, 9, 8, 12, tzinfo=timezone("US/Central")),
               city="College Station (TX)")  # stop 2
p34 = Observer(lon=-96.314445, lat=30.601389, inrad=False,
               date=datetime(2020, 11, 1, 12, tzinfo=timezone("US/Central")),
               city="College Station (TX)")  # stop 2
p4 = Observer(lon=-101.649200, lat=19.646402, inrad=False,
              date=datetime(2020, 11, 1, 12, tzinfo=timezone("US/Central")),
              city="Michoacan (MX)")  # end
p41 = Observer(lon=-101.649200, lat=19.646402, inrad=False,
               date=datetime(2020, 8, 20, 12, tzinfo=timezone("US/Eastern")),
               city="Michoacan (MX)")  # end
p42 = Observer(lon=-101.649200, lat=19.646402, inrad=False,
               date=datetime(2020, 9, 8, 12, tzinfo=timezone("US/Central")),
               city="Michoacan (MX)")  # end
p43 = Observer(lon=-101.649200, lat=19.646402, inrad=False,
               date=datetime(2020, 10, 1, 12, tzinfo=timezone("US/Central")),
               city="Michoacan (MX)")  # end

lobs = interpolate([p1, p2, p3, p4])
print(len(lobs))

plt.figure("butterflies-course", figsize=(5, 7))

x, y = [], []
for obs in lobs:
    x.append(obs.lon)
    y.append(obs.lat)

plt.plot(x, y, 'k-')
for obs in [p1, p2, p3, p4]:
    plt.plot(obs.lon, obs.lat, 'o', label="%s\n[%s]" % (obs.city, obs.date))
    # plt.text(obs.lon + (2 if obs.lat < 35 else -12), obs.lat, "%s\n%s" % (obs.city, obs.date), textalign="centre",
    #          fontsize=8)
plt.xticks([-100, -90, -80])
plt.yticks([10, 20, 30, 40, 50])
plt.xlim([-105, -80])
plt.ylim([15, 50])
plt.grid()
plt.legend()

di12 = p1.angdist(p2)
du12 = p2.date - p1.date
print("%s - %s:" % (p1.city, p2.city))
print("    Distance: %.4f" % np.rad2deg(di12))
print("    Duration: %s" % du12)

di23 = p2.angdist(p3)
du23 = p3.date - p2.date
print("%s - %s:" % (p2.city, p3.city))
print("    Distance: %.4f" % np.rad2deg(di23))
print("    Duration: %s" % du23)

di34 = p3.angdist(p4)
du34 = p4.date - p3.date
print("%s - %s:" % (p3.city, p4.city))
print("    Distance: %.4f" % np.rad2deg(di34))
print("    Duration: %s" % du34)

di1234 = di12 + di23 + di34
du1234 = du12 + du23 + du34

di14 = p1.angdist(p4)
du14 = p4.date - p1.date
print("%s - %s:" % (p1.city, p4.city))
print("    Distance: %.4f (direct), %.4f (with 2 stops)" % (np.rad2deg(di14), np.rad2deg(di1234)))
print("    Duration: %s (direct), %s (with 2 stops)" % (du14, du1234))

plots = {
    "": [p1, p2, p3, p4],
    "-mi": [p1, p12, p13, p14],
    "-il": [p21, p2, p23, p24],
    "-tx": [p31, p32, p3, p34],
    "-mx": [p41, p42, p43, p4]
}

for label in plots:
    labels = []
    azs, alts = [], []
    plt.figure("sun-course%s" % label, figsize=(7, 5))
    for obs in plots[label]:
        sun = Sun(observer=obs)
        print(sun.sunrise, sun.sunset, sun.is_ready)
        az, alt = [], []
        cur = sr = sun.sunrise
        while cur <= sr + timedelta(days=1):
            obs.date = cur
            sun.compute(obs)
            alt.append(sun.alt)
            az.append(sun.az % (2 * np.pi))
            cur += timedelta(minutes=30)
        azs.append(az)
        alts.append(alt)
        labels.append("%s (%s)" % (obs.city, obs.date.strftime("%d/%m/%Y")))
    azs = np.array(azs)
    alts = np.array(alts)
    max_alt = alts.max()
    plt.plot(azs, alts, 'k-', lw=0.5)
    lns = plt.plot(azs.T, alts.T, '.-')
    plt.annotate("max solar altitude: $%.2f^\circ$" % np.rad2deg(max_alt), xy=(np.pi, max_alt),
                 xytext=(np.pi / 8, max_alt + np.pi/16), arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3"))
    plt.ylabel("sun altitude (rad)")
    plt.xlabel("sun azimuth (rad)")
    plt.yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2],
               [r"0", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3 \pi}{8}$", r"$\frac{\pi}{2}$"])
    plt.xticks([np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi,
                9*np.pi/8, 5*np.pi/4, 11*np.pi/8, 3*np.pi/2, 13*np.pi/8, 7*np.pi/4, 15*np.pi/8, 2*np.pi],
               [r"", r"$\frac{\pi}{4}$", r"", r"$\frac{\pi}{2}$", r"", r"$\frac{3\pi}{4}$", r"", r"$\pi$",
                r"", r"$\frac{5\pi}{4}$", r"", r"$\frac{3\pi}{2}$", r"", r"$\frac{7\pi}{4}$", r"", r"$2\pi$"])
    plt.ylim([0, np.pi / 2])
    plt.xlim([0, 2 * np.pi])
    plt.legend(lns, labels)
# plt.show()
