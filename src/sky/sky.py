__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2019, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"


from ._helpers import eps, add_noise, RNG

from scipy.spatial.transform import Rotation as R

import numpy as np

# Transformation matrix of turbidity to luminance coefficients
T_L = np.array([[ 0.1787, -1.4630],
                [-0.3554,  0.4275],
                [-0.0227,  5.3251],
                [ 0.1206, -2.5771],
                [-0.0670,  0.3703]])


class Sky(object):
    """
    The Sky environment class. This environment class provides skylight cues.
    """

    def __init__(self, theta_s=0., phi_s=0., name="sky"):
        """

        :param theta_s: sun elevation (distance from horizon)
        :param phi_s: sun azimuth (clockwise from North)
        """
        super(Sky, self).__init__()
        self.__a, self.__b, self.__c, self.__d, self.__e = 0., 0., 0., 0., 0.
        self.__tau_L = 2.
        self._update_luminance_coefficients(self.__tau_L)
        self.__c1 = .6
        self.__c2 = 4.
        self.theta_s = theta_s
        self.phi_s = phi_s

        self.__theta = np.full(1, np.nan)
        self.__phi = np.full(1, np.nan)
        self.__aop = np.full(1, np.nan)
        self.__dop = np.full(1, np.nan)
        self.__y = np.full(1, np.nan)
        self.__eta = np.full(1, False)

        self.__is_generated = False
        self.__name = name

    def __call__(self, ori: R = None, irgbu=None, noise=0., eta=None, rng=RNG):
        """
        Call the sky instance to generate the sky cues.

        :param theta: array of points' elevation
        :type theta: np.ndarray
        :param phi: array of points' azimuth
        :type phi: np.ndarray
        :param noise: the noise level (sigma)
        :type noise: float
        :param eta: array of noise level in each point of interest
        :type eta: np.ndarray
        :param uniform_polariser:
        :type uniform_polariser: bool
        :return: Y, P, A
        """

        # set default arguments
        if ori is not None:
            xyz = ori.apply([1, 0, 0])
            phi = np.arctan2(xyz[..., 1], xyz[..., 0])
            theta = np.arccos(xyz[..., 2])
            # theta[xyz[..., 2] > 0] = np.pi - theta[xyz[..., 2] > 0]
            phi = (phi + np.pi) % (2 * np.pi) - np.pi

            # save points of interest
            self.__theta = theta.copy()
            self.__phi = phi.copy()
        else:
            theta = self.__theta
            phi = self.__phi
            ori = R.from_euler('ZY', np.vstack([phi, theta]).T, degrees=False)

        theta_s, phi_s = self.theta_s, self.phi_s

        # SKY INTEGRATION
        gamma = np.arccos(np.cos(theta) * np.cos(theta_s) +
                          np.sin(theta) * np.sin(theta_s) * np.cos(phi - phi_s))

        # Intensity
        i_prez = self.L(gamma, theta)
        i_00 = self.L(0., theta_s)  # the luminance (Cd/m^2) at the zenith point
        i_90 = self.L(np.pi / 2, np.absolute(theta_s - np.pi / 2))  # the luminance (Cd/m^2) on the horizon
        # influence of sky intensity
        i = (1. / (i_prez + eps) - 1. / (i_00 + eps)) * i_00 * i_90 / (i_00 - i_90 + eps)
        y = np.maximum(self.Y_z * i_prez / (i_00 + eps), 0.)  # Illumination

        # Degree of Polarisation
        lp = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
        p = np.clip(2. / np.pi * self.M_p * lp * (theta * np.cos(theta) + (np.pi / 2 - theta) * i), 0., 1.)

        # Angle of polarisation
        ori_s = R.from_euler('ZY', [phi_s, np.pi/2 - theta_s], degrees=False)
        x_s, y_s, _ = ori_s.apply([1, 0, 0]).T
        x_p, y_p, _ = ori.apply([1, 0, 0]).T
        a_x = np.arctan2(y_p - y_s, x_p - x_s) + np.pi/2
        a = (a_x + np.pi) % (2 * np.pi) - np.pi

        # create cloud disturbance
        if eta is None:
            eta = add_noise(noise=noise, rng=rng)
        y[eta] = 0.
        p[eta] = 0.  # destroy the polarisation pattern
        a[eta] = np.nan

        y[theta < 0] = np.nan
        p[theta < 0] = np.nan
        a[theta < 0] = np.nan

        self.__y = y
        self.__dop = p
        self.__aop = a
        self.__eta = eta

        self.__is_generated = True

        if irgbu is not None:
            y = spectrum_influence(y, irgbu).sum(axis=1)

        return y, p, a

    def L(self, chi, z):
        """
        Prez. et. al. Luminance function.
        Combines the scattering indicatrix and luminance gradation functions to compute the total
        luminance observed at the given sky element(s).

        :param chi: angular distance between the observed element and the sun location -- [0, pi]
        :param z: angular distance between the observed element and the zenith point -- [0, pi/2]
        :return: the total observed luminance (Cd/m^2) at the given element(s)
        """
        z = np.array(z)
        i = z < (np.pi / 2)
        f = np.zeros_like(z)
        if z.ndim > 0:
            f[i] = (1. + self.A * np.exp(self.B / (np.cos(z[i]) + eps)))
        elif i:
            f = (1. + self.A * np.exp(self.B / (np.cos(z) + eps)))
        phi = (1. + self.C * np.exp(self.D * chi) + self.E * np.square(np.cos(chi)))
        return f * phi

    @property
    def A(self):
        """
        A: Darkening or brightening of the horizon
        """
        return self.__a

    @A.setter
    def A(self, value):
        """
        :param value: Darkening or brightening of the horizon
        """
        self.__a = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self.__is_generated = False

    @property
    def B(self):
        """
        B: Luminance gradient near the horizon
        """
        return self.__b

    @B.setter
    def B(self, value):
        """
        :param value: Luminance gradient near the horizon
        """
        self.__b = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self.__is_generated = False

    @property
    def C(self):
        """
        C: Relative intensity of the circumsolar region
        """
        return self.__c

    @C.setter
    def C(self, value):
        """
        :param value: Relative intensity of the circumsolar region
        """
        self.__c = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self.__is_generated = False

    @property
    def D(self):
        """
        D: Width of the circumsolar region
        """
        return self.__d

    @D.setter
    def D(self, value):
        """
        :param value: Width of the circumsolar region
        """
        self.__d = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self.__is_generated = False

    @property
    def E(self):
        """
        E: relative backscattered light
        """
        return self.__e

    @E.setter
    def E(self, value):
        """
        :param value: relative backscattered light
        """
        self.__e = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self.__is_generated = False

    @property
    def c1(self):
        """
        :return: 1st coefficient of the maximum degree of polarisation
        """
        return self.__c1

    @property
    def c2(self):
        """
        :return: 2nd coefficient of the maximum degree of polarisation
        """
        return self.__c2

    @property
    def tau_L(self):
        """
        :return: turbidity
        """
        return self.__tau_L

    @tau_L.setter
    def tau_L(self, value):
        """
        :param value: turbidity
        """
        assert value >= 1., "Turbidity must be greater or eaqual to 1."
        self.__is_generated = self.__tau_L == value and self.__is_generated
        self._update_luminance_coefficients(value)

    @property
    def Y_z(self):
        """
        :return: the zenith luminance (K cd/m^2)
        """
        chi = (4. / 9. - self.tau_L / 120.) * (np.pi - 2 * (np.pi/2 - self.theta_s))
        return (4.0453 * self.tau_L - 4.9710) * np.tan(chi) - 0.2155 * self.tau_L + 2.4192

    @property
    def M_p(self):
        """
        :return: maximum degree of polarisation
        """
        return np.exp(-(self.tau_L - self.c1) / (self.c2 + eps))

    @property
    def Y(self):
        """
        :return: luminance of the sky (K cd/m^2)
        """
        assert self.__is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__y

    @property
    def DOP(self):
        """
        :return: the linear degree of polarisation in the sky
        """
        assert self.__is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__dop

    @property
    def AOP(self):
        """
        :return: the angle of linear polarisation in the sky
        """
        assert self.__is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__aop

    @property
    def theta(self):
        assert self.__is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__theta

    @theta.setter
    def theta(self, value):
        self.__theta = value
        self.__is_generated = False

    @property
    def phi(self):
        assert self.__is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__phi

    @phi.setter
    def phi(self, value):
        self.__phi = value
        self.__is_generated = False

    @property
    def eta(self):
        assert self.__is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__eta

    @eta.setter
    def eta(self, value):
        self.__eta = value
        self.__is_generated = False

    def _update_luminance_coefficients(self, tau_L):
        self.__a, self.__b, self.__c, self.__d, self.__e = T_L.dot(np.array([tau_L, 1.]))
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

    def _update_turbidity(self, a, b, c, d, e):
        T_T = np.linalg.pinv(T_L)
        tau_L, c = T_T.dot(np.array([a, b, c, d, e]))
        self.__tau_L = tau_L / c  # turbidity correction

    def copy(self):
        sky = Sky()
        sky.tau_L = self.tau_L
        sky.theta_s = self.theta_s
        sky.phi_s = self.phi_s
        sky.__c1 = self.__c1
        sky.__c2 = self.__c2
        sky.verbose = self.verbose

        sky.__theta = self.__theta
        sky.__phi = self.__phi
        sky.__aop = self.__aop
        sky.__dop = self.__dop
        sky.__y = self.__y
        sky.__eta = self.__eta

        sky.__is_generated = False
        return sky

    @staticmethod
    def from_observer(obs=None, date=None, yaw=0., theta_t=0., phi_t=0.):
        """
        Creates sky using an Ephem observer (Requires Ephem library)
        :param obs: the observer (location on Earth)
        :param date: the date of the observation
        :param yaw: the heading orientation of the observer
        :param theta_t: the heading tilt (pitch)
        :param phi_t: the heading tilt (roll)
        :return:
        """
        from ephemeris import Sun
        from datetime import datetime
        from observer import get_seville_observer

        sun = Sun()
        if obs is None:
            obs = get_seville_observer()
            obs.date = datetime(2017, 6, 21, 10, 0, 0) if date is None else date
        sun.compute(obs)
        theta_s, phi_s = np.pi/2 - sun.alt, (sun.az - yaw + np.pi) % (2 * np.pi) - np.pi

        return Sky(theta_s=theta_s, phi_s=phi_s, theta_t=theta_t, phi_t=phi_t)

    @staticmethod
    def from_type(sky_type):
        """

        :param sky_type:
        :return:
        """
        import os
        import yaml

        dir = os.path.dirname(os.path.realpath(__file__))
        with open(dir + "/standard-parameters.yaml", 'r') as f:
            try:
                sp = yaml.load(f)
            except yaml.YAMLError as exc:
                print("Could not load the sky types.", exc)
                return None

        rep = sp['type'][sky_type-1]
        a = sp['gradation'][rep['gradation']]['a']
        b = sp['gradation'][rep['gradation']]['b']
        c = sp['indicatrix'][rep['indicatrix']]['c']
        d = sp['indicatrix'][rep['indicatrix']]['d']
        e = sp['indicatrix'][rep['indicatrix']]['e']

        s = Sky()
        s._update_turbidity(a, b, c, d, e)
        # s.__tau_L = 2.

        for description in rep['description']:
            print(description)

        return s


def spectrum_influence(v, irgbu):
    wl = np.array([1200, 715, 535, 475, 350], dtype='float32')
    v = v[..., np.newaxis]
    l1 = 10.0 * irgbu * np.power(wl / 1000., 8) * np.square(v) / float(v.size)
    l2 = 0.001 * irgbu * np.power(1000. / wl, 8) * np.square(v).sum() / float(v.size)

    v_max = v.max()
    w_max = (v + l1 + l2).max()
    w = v_max * (v + l1 + l2) / w_max
    if isinstance(irgbu, np.ndarray):
        if irgbu.shape[0] == 1 and w.shape[0] > irgbu.shape[0]:
            irgbu = np.vstack([irgbu] * w.shape[0])
        w[irgbu < 0] = np.hstack([v] * irgbu.shape[1])[irgbu < 0]
    elif irgbu < 0:
        w = v
    return w


def visualise_luminance(sky):
    import matplotlib.pyplot as plt

    plt.figure("Luminance", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = sky.theta_s, sky.phi_s
    ax.scatter(sky.phi, sky.theta, s=20, c=sky.Y, marker='.', cmap='Blues_r', vmin=0, vmax=6)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()


def visualise_degree_of_polarisation(sky):
    import matplotlib.pyplot as plt

    plt.figure("degree-of-polarisation", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = sky.theta_s, sky.phi_s
    print(theta_s, phi_s)
    ax.scatter(sky.phi, sky.theta, s=10, c=sky.DOP, marker='.', cmap='Greys', vmin=0, vmax=1)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()


def visualise_angle_of_polarisation(sky):
    import matplotlib.pyplot as plt

    plt.figure("angle-of-polarisation", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = sky.theta_s, sky.phi_s
    ax.scatter(sky.phi, sky.theta, s=10, c=sky.AOP, marker='.', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()
