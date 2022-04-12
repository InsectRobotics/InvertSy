from invertpy.sense.polarisation import PolarisationSensor

from invertsy.env.sky import Sky
from invertsy.sim._helpers import create_dra_axis

from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt


flat = True

sky = Sky(np.deg2rad(30), np.pi)

sensor = PolarisationSensor(field_of_view=90, nb_lenses=4, omm_photoreceptor_angle=4, omm_rho=np.deg2rad(.05),
                            omm_res=10000, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
                            omm_pol_op=1., noise=0., name="Rob's sensor")
print(sensor)

r = sensor(sky=sky)

print(sensor.responses)

plt.figure("env", figsize=(3.33, 3.33))
pol = create_dra_axis(sensor, draw_axis=True)
pol.set_array(np.array(r).flatten())
plt.show()
