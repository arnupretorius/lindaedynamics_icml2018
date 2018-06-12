import numpy as np


# Helper functions to compute theoretical learning dynamics
def dae_learning_dynamics(lam, var, gamma, N, n_epoch, learning_rate, u0):
    dynamics = []
    tau = N/learning_rate
    g = N*gamma
    xi = lam + N*var
    
    for t in range(n_epoch):
        E = np.exp((2*(lam - g)*t)/tau)
        num = (lam - g)*E
        denom = xi*(E - 1) + (lam - g)/u0
        uf = num/denom
        dynamics.append(uf)
    
    return np.asarray(dynamics)
  
def compute_correlation_matrix(x, y):
    mat = np.zeros((y.shape[1], x.shape[1]))
    for x_vec, y_vec in zip(x, y):
        mat += np.outer(y_vec, x_vec)
    return mat
  
def theoretical_learning_dynamics(X, y, n_epoch, lr, var, reg, u0 = 2.5e-7):
    dyns = np.zeros((1,X.shape[1]))
    corr_mat = compute_correlation_matrix(X, y)
    U, S, V = np.linalg.svd(corr_mat)
    dyns = dae_learning_dynamics(S, var, reg, X.shape[0], n_epoch, lr, u0)
    return dyns