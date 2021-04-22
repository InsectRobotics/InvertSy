"""

"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

import invertsy.agent
import invertsy.env
import invertsy.sim


def set_data_directory(data_dir):
    """
    Changes the root directory where the data are stored.

    Parameters
    ----------
    data_dir: str
        the new directory
    """
    from .__helpers import __data__

    __data__ = data_dir
