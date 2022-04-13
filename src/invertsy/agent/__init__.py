"""
Python package that allows the development of agents that observe their environment with
their sensors, process this information using their brain components and act-back in their
environment. This is useful for closed-loop experiments as the actions of the agents can
also change their environment and therefore affect their observations.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"


from .agent import Agent, PathIntegrationAgent, VisualNavigationAgent, NavigatingAgent
from .agent import VectorMemoryAgent, RouteFollowingAgent
