from env import Sky
from sensor import Compass, decode_sph
from transform import tilt

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sky = Sky(phi_s=np.pi, theta_s=np.pi/3)
    # env = Sky(theta_s=0)
    compass = Compass()
    dra = compass.dra
    # dra.theta_t = np.pi/6
    # dra.phi_t = np.pi/3
    r_tcl = compass(sky)
    r_po = dra.r_po

    print(r_tcl)
    print(dra.r_pol)

    fig = plt.figure("Celestial Compass", figsize=(9, 4.5))
    ax = plt.subplot(121, polar=True)
    ax.set_title("POL neural responses")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    ax.scatter(sky.phi, sky.theta, s=100, c=dra.r_pol, marker='.', cmap='coolwarm', vmin=-1, vmax=1)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    ax.scatter(sky.phi_t, sky.theta_t, s=50, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    ax = plt.subplot(122, polar=True)
    ax.set_title("TCL neural responses")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(np.linspace(0, 2*np.pi, 100, endpoint=True), np.zeros(100), 'gray', lw=5)
    ax.plot([yaw, yaw], [-1, 1], 'gray', lw=3)
    ax.plot(np.linspace(0, 2*np.pi, 100, endpoint=True),
            compass.amp * np.cos(compass.yaw - np.linspace(0, 2*np.pi, 100, endpoint=True)), 'k--')
    ax.plot(np.linspace(0, 2*np.pi, 9, endpoint=True), np.r_[r_tcl, r_tcl[0]], 'k-')

    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()

