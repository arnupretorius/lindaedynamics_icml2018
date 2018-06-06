# --- Loss functions ---
import numpy as np

# mean squared error


def mean_squared_error_forward(y_pred, y_true):
    err = (1 - y_pred)
    out = 0.5 * y_true * np.mean(np.power(err, 2))
    cache = (err, y_true)
    return out, cache


def mean_squared_error_backward(dout, cache):
    err, y_true = cache
    dmse = dout * -np.mean(err)*y_true
    return dmse

# root mean squared error


#def root_mean_squared_error(y_pred, y_true):
#	err = (y_true - y_pred)
#    out = np.sqrt(np.mean(err**2))
#    return out
