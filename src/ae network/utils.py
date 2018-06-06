# utility functions for scalar neural networks
import numpy as np


def theoretical_dynamics(x, y, w1_init, w2_init, n_epoch, learning_rate):
    dynamics = []
    tau = 1 / learning_rate
    u0 = w1_init * w2_init
    for t in range(n_epoch):
        E = np.exp((2 * x * y * t) / tau)
        uf = (y * E) / (x * E - x + y / u0)
        dynamics.append(uf)
    return dynamics


def generate_circle_points(r, n=1):
    x = []
    y = []
    for i in range(0, n):
        angle = np.random.uniform(0, 1) * (np.pi * 2)
        x.append(np.cos(angle) * r)
        y.append(np.sin(angle) * r)
    return x, y
