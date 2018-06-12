# --- regularizers ---


# weight decay


def l2_loss(params, l2_reg):
    reg = 0.0
    for key, param in params.items():
        reg += 0.5*param**2
    return reg
