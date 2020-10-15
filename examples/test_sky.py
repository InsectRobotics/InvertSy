from geometry import fibonacci_sphere
from sky import Sky
from observer import get_seville_observer
from plots import plot_sky

import numpy as np
import matplotlib.pyplot as plt


flat = True
samples = 5000
seville_obs = get_seville_observer()

theta, phi = fibonacci_sphere(samples, 180, antiflat=flat)
sky = Sky(np.deg2rad(30), np.pi)
y, p, a = sky(theta, phi)

plt.figure("sky", figsize=(10, 3.33))
# plot_sky(phi, theta, y, p, a).show()
plot_sky(phi, theta, y, p, a, flat=flat).show()
