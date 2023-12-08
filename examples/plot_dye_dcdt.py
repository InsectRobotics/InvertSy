import invertpy.brain.centralcomplex.fanshapedbody_dye as dye
import matplotlib.pyplot as plt
import numpy as np

N_A = dye.AVOGADRO_CONSTANT
h = dye.PLANK_CONSTANT
speed_of_light = dye.SPEED_OF_LIGHT
dye_parameters = {
    "before annealing": {
        "drop cast (S1, thicker)": {
            "epsilon": 1.58e+05,
            "l": 10.0e-04,
            "k": 3.89e-03,
            "c_tot": 5e-03,
            "phi": 0.002,
            "lambda": 6.53e-07,
            "V": 1e-18,
            "W_max": 1.0e-15
        },
        "drop cast (S2, thinner)": {
            "epsilon": 1.58e+05,
            "l": 3.5e-04,
            "k": 9.6e-04,
            "c_tot": 5e-03,
            "phi": 0.002,
            "lambda": 6.53e-07,
            "V": 1e-18,
            "W_max": 1.0e-15
        },
        "spin coated": {
            "epsilon": 1.58e+05,
            "l": 2.5e-04,
            "k": 6.1e-05,
            "c_tot": 5e-03,
            "phi": 0.002,
            "lambda": 6.53e-07,
            "V": 1e-18,
            "W_max": 1.0e-15
        },
    }, "after annealing": {
        "drop cast (S1, thicker)": {
            "epsilon": 1.58e+05,
            "l": 10.0e-04,
            "k": 1.2e-05,
            "c_tot": 5e-03,
            "phi": 0.002,
            "lambda": 6.53e-07,
            "V": 1e-18,
            "W_max": 1.0e-15
        },
        "drop cast (S2, thinner)": {
            "epsilon": 1.58e+05,
            "l": 3.5e-04,
            "k": 1.8e-05,
            "c_tot": 5e-03,
            "phi": 0.002,
            "lambda": 6.53e-07,
            "V": 1e-18,
            "W_max": 1.0e-15
        },
        "spin coated": {
            "epsilon": 1.58e+05,
            "l": 2.5e-04,
            "k": 5.8e-05,
            "c_tot": 5e-03,
            "phi": 0.002,
            "lambda": 6.53e-07,
            "V": 1e-18,
            "W_max": 1.0e-15
        },
    }
}

if __name__ == "__main__":

    u = np.zeros(25000)
    u[500:1500] = 1

    plt.figure("Dye parameters", figsize=(7, 2.5))
    for i, (title, group) in enumerate(dye_parameters.items()):
        plt.subplot(1, len(dye_parameters), 1 + i)
        plt.title(title)
        for name, params in group.items():
            E = h * speed_of_light / params["lambda"]
            k_phi = params["W_max"] / (E * params["V"] * N_A)

            transmittance = lambda x: dye.transmittance(
                x, epsilon=params["epsilon"], length=params["l"], c_tot=params["c_tot"])

            c_off = np.zeros(2500)
            for t in range(1, 2500):
                dc_dt = dye.dcdt(u[t-1], transmittance, k=params["k"], phi=params["phi"], k_phi=k_phi)
                c_off[t] = c_off[t-1] + dc_dt(0, c_off[t-1])

            plt.plot(transmittance(c_off), label=name)
        plt.xlim(0, 2499)
        plt.ylim(0, 1)
        plt.xlabel("time (sec)")
        plt.ylabel("transmittance")

    plt.legend()
    plt.tight_layout()
    plt.show()
