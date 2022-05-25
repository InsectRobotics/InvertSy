"""
Python package that implements the environments where the agents can get their observations
from. These are models of the sky, 3D object (e.g. vegetation), odour gradients, etc, that
affect what the agents observe using their sensors. These environments are in abstract form
and are rendered using the sensors of the agents, designed for invertebrate observations,
which allows for more invertebrate-like stimulation.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from .world import Seville2009, SimpleWorld, WorldBase
from .sky import Sky, UniformSky
from .odour import StaticOdour
from ._helpers import reset_data_directory
