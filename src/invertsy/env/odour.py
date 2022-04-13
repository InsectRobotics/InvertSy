from ._helpers import RNG, add_noise

import numpy as np


class StaticOdour(object):
    def __init__(self, centre=None, spread=1., max_intensity=1., dtype='float32', name='my_odour'):
        """
        Static odour class that created particles based on the Gaussian distibution.

        This odour class does not take wind into account.

        Parameters
        ----------
        centre : np.ndarray[float], optional
            the 3D centre of the Gaussian (source of odour). Default is [0, 0, 0]
        spread : float, optional
            the spread of the odour (variance of the Gaussion distribution) in meters. Default is 1m
        max_intensity : float, optional
            the maximum intensity of the odour. Default is 1
        dtype : np.dtype, optional
            the type of the values to handle. Default is 'float32'
        name : str, optional
            the name of the odour. Default is 'my_odour'
        """
        if centre is None:
            centre = np.zeros(3, dtype=dtype)

        self._centre = centre
        self._spread = spread
        self._max_intensity = max_intensity
        self.dtype = dtype
        self.name = name

    def __call__(self, pos, noise=0., eta=None, rng=RNG):
        """
        Calculates the intensity of the current odour in the given position(s).

        Parameters
        ----------
        pos : np.ndarray[float]
            the 3D positions of the points of interest
        noise: float, optional
            percentage of noise to be added to the odour intensity. Default is 0
        eta: np.ndarray[bool], optional
            map of positions to apply the noise on. Default is None
        rng
            the random generator

        Returns
        -------
        np.ndarray[float]
            the intensity in all the given positions.
        """
        rel_int = np.exp(-0.5 * np.linalg.norm(pos - self._centre.reshape((1, -1)), axis=-1) / np.square(self._spread))
        if eta is None:
            eta = add_noise(noise=noise, shape=rel_int.shape, rng=rng)
        rel_int[eta] = 0.

        return self._max_intensity * rel_int

    @property
    def centre(self):
        """
        The position of the source of the odour.

        Returns
        -------
        np.ndarray[float]
        """
        return self._centre

    @property
    def spread(self):
        """
        The spread of the odour around its source.

        Returns
        -------
        float
        """
        return self._spread

    @property
    def max_intensity(self):
        """
        The maximum intensity of the odour (above at the source).

        Returns
        -------
        float
        """
        return self._max_intensity
