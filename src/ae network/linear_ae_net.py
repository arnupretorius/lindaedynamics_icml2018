import numpy as np
from .layers import affine_forward, affine_backward
from .loss_functions import mean_squared_error_forward, mean_squared_error_backward
from .regularizers import l2_loss
from .optimisers import optimise


# create linear autoencoder class
class LinearAutoEncoder():
    '''Neural network class'''

    def __init__(self, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # parameter dictionary
        self.params = {}
        self.history = {'W1': [], 'W2': [], 'loss': []}

    def optimiser(self, optimiser):
        self.optimiser = optimiser

    def initialiser(self, input_dim, hidden_dim, initialiser=None, W1=None, W2=None, **kwargs):


        if initialiser is None:
            # initialise weights
            if w1 is None or w2 is None:
                self.params['W1'] = 0.001 * \
                    np.random.randn(hidden_dim, input_dim)
                self.params['W2'] = 0.001 * \
                    np.random.randn(input_dim, hidden_dim)
                self.history['W1'].append(self.params['W1'])
                self.history['W2'].append(self.params['W2'])
            else:
                self.params['W1'] = W1
                self.params['W2'] = W2
                self.history['W1'].append(self.params['W1'])
                self.history['W2'].append(self.params['W2'])
        else:
            self.initialiser = initialiser
            self.params['W1'] = self.initialiser(size = (hidden_dim, input_dim), first=True, **kwargs)
            self.params['W2'] = self.initialiser(size = (input_dim, hidden_dim), last=True, **kwargs)
            self.history['W1'].append(self.params['W1'])
            self.history['W2'].append(self.params['W2'])

    def train(self, X, n_epoch=100, learning_rate=0.1, reg_param=0, verbose=False):

        W1 = self.params['W1']
        W2 = self.params['W2']

        for t in range(n_epoch):

            # compute scores
            H, H_cache = affine_forward(W1, X.T)
            scores, scores_cache = affine_forward(W2, H)

            # compute loss
            data_loss, loss_cache = mean_squared_error_forward(scores.T, X)
            reg_loss = 0.5 * reg_param * (np.sum(W1*W1) + np.sum(W2*W2))
            loss = data_loss + reg_loss
            if verbose:
                if t % 10 == 0:
                    print('iteration: ', t, 'loss: ', loss)

            # compute the gradient using backpropagation
            dscores = mean_squared_error_backward(1, loss_cache)
            dH, dW2 = affine_backward(dscores, scores_cache)
            dX, dW1 = affine_backward(dH, H_cache)

            # add regularisation gradients
            dW2 += reg_param * W2
            dW1 += reg_param * W1

            # store gradients
            grads['W1'] = dW1
            grads['W2'] = dW2

            # perform optimisation step
            self.params = optimise(
                self.params, grads, self.optimiser, learning_rate)

            # store weights
            self.history['W1'].append(self.params['W1'])
            self.history['W2'].append(self.params['W2'])

    def predict(self, X):
        h1 = np.dot(X, self.params['W1'])
        scores = np.dot(h1, self.params['W2'])
        return scores
