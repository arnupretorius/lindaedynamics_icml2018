# Neural network components
import numpy as np

# --- Layers ---

# Affine layers


def affine_forward(x, w):
    out = w*x
    cache = (x, w)
    return out, cache


def affine_backward(dout, cache):
    x, w = cache
    if type(x) is np.ndarray:
        dx = np.repeat(dout*w, len(x))
    else:
        dx = dout*w
    dw = np.mean(dout*x)
    return dx, dw

# activation layers


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx