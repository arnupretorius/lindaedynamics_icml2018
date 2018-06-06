# --- optimisers ---


# Global optimiser


def optimise(params, grads, optimiser, learning_rate):
    for name, grad in grads.items():
        params[name] = optimiser(params[name], grad, learning_rate)
    return params


# First order methods


def gradient_descent(param, grad, learning_rate):
    param += -learning_rate*grad
    return param