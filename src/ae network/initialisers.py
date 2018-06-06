# weight initialisers

# uniform initialisation
import numpy as np
from scipy.stats import ortho_group


def uniform_init(size, weight_scale=1, a=-1, b=1):
    W = weight_scale * np.random.uniform(low=a, high=b, size=size)
    return W

# standard normal initialisation


def normal_init(size, weight_scale=1, loc=0, scale=1):
    W = weight_scale * np.random.normal(loc=loc, scale=scale, size=size)
    return W


def orthogonal_init(size, weight_scale=1, first=False, last=False, X=None):

	if first:
		A, V = np.linalg.eig(X.T.dot(X))
		R = ortho_group.rvs(dim=size[0])
		D = weight_scale * np.random.normal(size=size)
		W = R.dot(D).dot(V.T)
	elif last:
		A, V = np.linalg.eig(X.T.dot(X))
		R = ortho_group.rvs(dim=size[1])
		D = weight_scale * np.random.normal(size=size)
		W = V.dot(D).dot(R.T)
	else:
		R1 = ortho_group.rvs(dim=size[0])
		R2 = ortho_group.rvs(dim=size[1])
		D = weight_scale * np.random.normal(size=size)
		W = R1.dot(D).dot(R2)
	return W


