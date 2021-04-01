from sense._helpers import fibonacci_sphere
from env.sky import Sky
from sense.vision import CompoundEye
from simplot._plots import plot_sky

from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt


flat = True
samples = 5000

phi, theta, _ = fibonacci_sphere(samples, np.pi).T
ori = R.from_euler('ZY', np.vstack([phi, theta]).T, degrees=False)
sky = Sky(np.deg2rad(60), np.pi)

eye = CompoundEye(omm_ori=ori, omm_rho=np.deg2rad(5),
                  ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
                  omm_pol_op=0., noise=0.)
r = eye(sky=sky)
print(eye)

oris = eye.ori * eye.omm_ori
y, p, a = sky(oris)
y = np.square(r[..., 1])
# print(r.min(), r.max())

plt.figure("env", figsize=(10, 3.33))
plot_sky(eye.omm_ori, y, p, a, flat=flat).show()
