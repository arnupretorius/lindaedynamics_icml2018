# --- MODEL ---
import numpy as np
from src.scalar_net.layers import affine_forward, affine_backward
from src.scalar_net.optimisers import optimise
from src.scalar_net.loss_functions import mean_squared_error_forward, mean_squared_error_backward


class ScalarNeuralNetwork(object):

    def __init__(self, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # parameter dictionary
        self.params = {}
        self.history = {'w1': [], 'w2': [], 'loss': []}

    def optimiser(self, optimiser):
        self.optimiser = optimiser

    def initialiser(self, initialiser=None, w1=None, w2=None, **kwargs):

        if initialiser is None:
            # initialise weights
            if w1 is None or w2 is None:
                self.params['w1'] = 0.001 * np.random.randn()
                self.params['w2'] = 0.001 * np.random.randn()
                self.history['w1'].append(self.params['w1'])
                self.history['w2'].append(self.params['w2'])
            else:
                self.params['w1'] = w1
                self.params['w2'] = w2
                self.history['w1'].append(self.params['w1'])
                self.history['w2'].append(self.params['w2'])
        else:
            self.initialiser = initialiser
            self.params['w1'] = self.initialiser(**kwargs)
            self.params['w2'] = self.initialiser(**kwargs)
            self.history['w1'].append(self.params['w1'])
            self.history['w2'].append(self.params['w2'])

    def train(self, x, y, n_epoch, learning_rate, v=0.0, l2=0.0, verbose=False):

        grads = {}

        for t in range(n_epoch):

            # Forward pass
            hidden, hidden_cache = affine_forward(x, self.params['w1'])
            scores, scores_cache = affine_forward(hidden, self.params['w2'])

            # compute loss
            loss, loss_cache = mean_squared_error_forward(scores, y)

            # compute regularization loss
            # l2_loss(self.params, l2)
            reg_loss = 0.5 * l2 * (self.params['w1']**2 + self.params['w2']**2)
            noise_loss = 0.5 * v * (self.params['w1']*self.params['w2'])**2
            loss += reg_loss + noise_loss

            # store loss
            self.history['loss'].append(loss)

            if verbose:
                print(loss)

            # Backward pass
            dloss = mean_squared_error_backward(1, loss_cache)
            #print('dloss: ', dloss)
            dh, dw2 = affine_backward(dloss, scores_cache)
            #print('dh: ', dh)
            #print('dw2: ', dw2)
            dx, dw1 = affine_backward(dh, hidden_cache)
            #print('dx: ', dx)
            #print('dw1: ', dw1)

            # regularization gradients
            dw1 += l2 * self.params['w1']
            dw2 += l2 * self.params['w2']

            # noise gradients
            dw1 += v * self.params['w1']*self.params['w2']**2
            dw2 += v * self.params['w2']*self.params['w1']**2

            # store gradients
            grads['w1'] = dw1
            grads['w2'] = dw2

            # perform optimisation step
            self.params = optimise(
                self.params, grads, self.optimiser, learning_rate)

            # store weights
            self.history['w1'].append(self.params['w1'])
            self.history['w2'].append(self.params['w2'])

    def predict(self, x):
        hidden = self.params['w1'] * x
        scores = self.params['w2'] * hidden
        return scores
