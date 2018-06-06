# Neural network components
import numpy as np

# --- Layers ---

# Affine layers


def affine_forward(W, X):
    out = np.dot(W, X)
    cache = (X, W)
    return out, cache


def affine_backward(dout, cache):
    X, W = cache
    dx = np.dot(dout, W)
    dw = np.dot(dout, X)
    return dx, dw

# activation layers


def relu_forward(X):
    out = np.maximum(0, X)
    cache = X
    return out, cache


def relu_backward(dout, cache):
    X = cache
    dx = dout
    dx[X <= 0] = 0
    return dx