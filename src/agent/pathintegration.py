from .agent import Agent
from env.sky import Sky

from invertsensing.polarisation import PolarisationSensor
from invertbrain.compass import PolarisationCompass, decode_sph
from invertbrain.centralcomplex import CentralComplex
from invertbrain.cx_helpers import get_steering

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class PathIntegrationAgent(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pol_sensor = PolarisationSensor(nb_input=60, field_of_view=56, degrees=True)
        pol_brain = PolarisationCompass(nb_pol=60, loc_ori=copy(pol_sensor.omm_ori), nb_sol=8, integrated=True)
        cx = CentralComplex(nb_tb1=8)

        self.add_sensor(pol_sensor, local=True)
        self.add_brain_component(pol_brain)
        self.add_brain_component(cx)

        self._pol_sensor = pol_sensor
        self._pol_brain = pol_brain
        self._cx = cx

        self._default_flow = self._dx * np.ones(2) / np.sqrt(2)

    def _sense(self, sky=None, scene=None, flow=None, **kwargs):
        if sky is None:
            r = 0.
        else:
            r = self._pol_sensor(sky=sky, scene=scene)

        if flow is None:
            flow = self._default_flow
            # if scene is None:
            #     flow = self._default_flow
            # else:
            #     flow = optic_flow(world, self._dx)

        r_tcl = self._pol_brain(r_pol=r, ori=self._pol_sensor.ori)
        _, phi = decode_sph(r_tcl)
        return self._cx(phi=phi, flow=flow)

    def _act(self):
        steer = get_steering(self._cx)
        self.rotate(R.from_euler('Z', steer, degrees=False))
        self.move_forward()


def pi_routine(agent: PathIntegrationAgent, route: np.ndarray):

    sky = Sky(theta_s=60, phi_s=170, degrees=True)

    agent.xyz = route[0, :3]
    agent.ori = R.from_euler('Z', route[0, 3], degrees=True)

    stats = {
        "path": [],
        "L": [],  # straight distance from the nest
        "C": [],  # distance towards the nest that the agent has covered
    }

    def callback_all(a: PathIntegrationAgent):
        stats["path"].append([a.x, a.y, a.z, a.yaw])
        stats["L"].append(np.linalg.norm(a.xyz - route[0, :3]))

    def callback_outbound(a: PathIntegrationAgent):
        callback_all(a)
        stats["C"].append(0.)

    def callback_inbound(a: PathIntegrationAgent):
        callback_all(a)
        stats["C"].append(stats["C"][-1] + a.step_size)

    # outbound path
    for x, y, z, yaw in route:
        dx = np.linalg.norm(np.array([x, y, z]) - agent.xyz)
        flow = dx * np.ones(2) / np.sqrt(2)
        agent(sky=sky, flow=flow, act=False, callback=callback_outbound)
        agent.xyz = [x, y, z]
        agent.ori = R.from_euler('Z', yaw, degrees=True)
        print(agent)

    # inbound path
    while stats["C"][-1] < 15:
        agent(sky=sky, act=True, callback=callback_inbound)
        print(agent)

    return stats
