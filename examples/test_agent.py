from invertpy.sense.vision import CompoundEye

from invertsy.agent.agent import Agent

from scipy.spatial.transform import Rotation as R

import sys


def main(*args):
    a = Agent(xyz=[1, 0, 0])
    a.add_sensor(CompoundEye(xyz=[1, 0, 0], ori=R.from_euler('Z', 30, degrees=True)), local=True)
    print(a)
    print(a.sensors[0])

    a.translate([1, 0, 0])
    print(a)
    print(a.sensors[0])

    a.rotate(R.from_euler('ZYX', [10, -5, 0], degrees=True))
    print(a)
    print(a.sensors[0])


if __name__ == '__main__':
    main(*sys.argv)
