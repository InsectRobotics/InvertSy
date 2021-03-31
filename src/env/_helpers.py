import numpy as np

RNG = np.random.RandomState(2021)
eps = np.finfo(float).eps


def set_rng(seed):
    global RNG
    RNG = np.random.RandomState(seed)


def add_noise(v=None, noise=0., shape=None, fill_value=0, rng=RNG):
    if shape is None and v is not None:
        shape = v.shape
    if shape is not None:
        size = np.sum(shape)
    elif v is not None:
        size = v.size
    else:
        size = None
    if isinstance(noise, np.ndarray):
        if size is None or noise.size == size:
            eta = np.array(noise, dtype=bool)
        else:
            eta = np.zeros(shape, dtype=bool)
            eta[:noise.size] = noise
    elif noise > 0:
        if shape is not None:
            eta = np.argsort(np.absolute(rng.randn(*shape)))[:int(noise * shape[0])]
        else:
            eta = rng.randn()
    else:
        eta = np.zeros_like(v, dtype=bool)

    if v is not None:
        v[eta] = fill_value

    return eta