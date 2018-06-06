# --- Loss functions ---
import numpy as np

# mean squared error


def mean_squared_error_forward(y_pred, y_true):
    err = (y_true - y_pred)
    out = 0.5 * np.mean(err**2)
    cache = err
    return out, cache


def mean_squared_error_backward(dout, cache):
    err = cache
    dmse = dout * -np.mean(err)
    return dmse

# root mean squared error


#def root_mean_squared_error(y_pred, y_true):
#	err = (y_true - y_pred)
#    out = np.sqrt(np.mean(err**2))
#    return out
