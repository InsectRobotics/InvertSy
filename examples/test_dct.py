from invertpy.sense.vision import CompoundEye
from invertpy.brain.synapses import dct_synapses, dct_omm_synapses

from invertsy.env.sky import Sky
from invertsy.env.world import Seville2009

import matplotlib.pyplot as plt
import numpy as np

import sys


def main(*args):
    nb_input = 2000
    nb_output = 500
    shift = 1500
    eye = CompoundEye(nb_input=nb_input, xyz=[5, 5, .001])
    dct = dct_synapses(nb_input)
    dct_omm = dct_omm_synapses(eye.omm_ori)

    sky = Sky(60, 0, degrees=True)
    scene = Seville2009()
    r = eye(sky=sky, scene=scene).mean(axis=1)
    print("R:", r.min(), r.max())
    print()

    r_ima_dct = r @ dct
    r_ima = r_ima_dct @ dct.T
    print("Image DCT:")
    print("Reconstruction:", r_ima.min(), r_ima.max())
    print("Close: %d / %d" % (np.sum(np.isclose(r, r_ima)), r.shape[0]))
    print()

    r_ima_dct_low = r @ dct[:, shift:shift+nb_output]
    r_ima_low = r_ima_dct_low @ dct[:, shift:shift+nb_output].T
    print("Image DCT (low dimensional):")
    print("Reconstruction:", r_ima_low.min(), r_ima_low.max())
    print("Close: %d / %d" % (np.sum(np.isclose(r, r_ima_low)), r.shape[0]))
    print()

    r_omm_dct = r @ dct_omm
    r_omm = r_omm_dct @ dct_omm.T
    print("Ommatidia DCT:")
    print("Reconstruction:", r_omm.min(), r_omm.max())
    print("Close: %d / %d" % (np.sum(np.isclose(r, r_omm)), r.shape[0]))

    r_omm_dct_low = r @ dct_omm[:, shift:shift+nb_output]
    r_omm_low = r_omm_dct_low @ dct_omm[:, shift:shift+nb_output].T
    print("Ommatidia DCT (low dimensional):")
    print("Reconstruction:", r_omm_low.min(), r_omm_low.max())
    print("Close: %d / %d" % (np.sum(np.isclose(r, r_omm_low)), r.shape[0]))

    # ================== FIGURE ====================

    plt.figure("DCT", figsize=(8., 4.5))
    i1, i2, i3 = 0, 1, 2

    m_size = 20000 // nb_input
    yaw, pitch, _ = eye.omm_ori.as_euler('ZYX', degrees=True).T
    plt.subplot(331)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r, cmap='Greys_r', vmin=0, vmax=1)
    plt.ylabel("raw")
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(334)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r_ima, cmap='Greys_r', vmin=0, vmax=1)
    plt.ylabel("Image DCT")
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(337)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r_omm, cmap='Greys_r', vmin=0, vmax=1)
    plt.ylabel("Ommatidia DCT")
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(332)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r_ima_dct, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(335)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r_ima_low, cmap='Greys_r', vmin=0, vmax=1)
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(338)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r_omm_low, cmap='Greys_r', vmin=0, vmax=1)
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(333)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=r_omm_dct, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(336)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=dct_omm[i2], cmap='coolwarm')  # , vmin=-1, vmax=1)
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])
    plt.subplot(339)
    plt.scatter(yaw, np.sin(np.deg2rad(-pitch)), s=m_size, c=dct_omm[i3], cmap='coolwarm')  # , vmin=-1, vmax=1)
    plt.xlim([-180, 180])
    plt.ylim([-1, 1])
    plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
