from invertpy.sense._helpers import fibonacci_sphere
from invertpy.sense.vision import CompoundEye

from invertsy.env.sky import Sky, visualise_luminance, visualise_angle_of_polarisation, visualise_degree_of_polarisation

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
# y = np.square(r[..., 1])

plt.figure("sky properties", figsize=(10, 3.33))

ax_l = plt.subplot(131, polar=True)
ax_d = plt.subplot(132, polar=True)
ax_a = plt.subplot(133, polar=True)
ax_l.axis('off')
ax_a.axis('off')

visualise_luminance(sky, y, ax=ax_l)
visualise_degree_of_polarisation(sky, ax=ax_d)
visualise_angle_of_polarisation(sky, ax=ax_a)

plt.tight_layout()
plt.show()

