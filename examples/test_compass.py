from invertpy.sense.polarisation import PolarisationSensor
from invertpy.brain.compass import PolarisationCompass, ring2sph

from invertsy.env.sky import Sky
from invertsy.sim._helpers import create_dra_axis

from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt


flat = True

fov = 90
nb_ommatidia = 4
nb_receptors = 4

sky = Sky(np.deg2rad(30), 0)

sensor = PolarisationSensor(field_of_view=fov, nb_lenses=nb_ommatidia, omm_photoreceptor_angle=2, omm_rho=np.deg2rad(5),
                            omm_res=1, ori=R.from_euler('ZYX', [0, 0, 0], degrees=True),
                            omm_pol_op=1., noise=0., name="Rob's sensor")
compass = PolarisationCompass(nb_pol=nb_ommatidia, loc_ori=sensor.omm_ori,
                              sigma=13, shift=40, nb_receptors=nb_receptors)

print(sensor)
print(compass)

r = sensor(sky=sky)
r_com = compass(r)

print(r)
print(np.rad2deg(ring2sph(r_com.flatten())))

plt.figure("env", figsize=(3.33, 3.33))
pol = create_dra_axis(sensor, draw_axis=True)
pol.set_array(np.array(r).flatten())
plt.show()
