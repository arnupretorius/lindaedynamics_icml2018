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


def hyperbolic_learning_dynamics(x, y, n_epoch, learning_rate, w1_init, w2_init, l2=0):
    dynamics = []
    y = y - l2 / x  # this is for the regularization case (see below)
    c0 = np.abs(w2_init**2 - w1_init**2)
    theta0 = np.arcsinh(2 * (w2_init * w1_init) / c0)
    g = np.sqrt(c0**2 * x**2 + 4 * y**2)
    tau = 1 / learning_rate
    for t in range(n_epoch):
        E = np.exp(g * x * t / tau)
        num = (1 - E) * (g**2 - c0**2 * x**2 - 2 * c0 * y *
                         np.tanh(theta0 / 2)) - 2 * (E + 1) * g * y * np.tanh(theta0 / 2)
        denom = (1 - E) * (2 * c0 * x * y + 4 * y**2 *
                           np.tanh(theta0 / 2)) - 2 * (E + 1) * g * y
        thetaf = 2 * np.arctanh(num / denom)
        uf = 0.5 * c0 * np.sinh(thetaf)
        dynamics.append(uf)
    return dynamics
