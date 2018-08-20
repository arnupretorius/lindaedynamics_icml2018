import numpy as np
import torch

# create linear autoencoder class 
class LinearAutoEncoder():
    '''Neural network class'''
    def __init__(self):
        self.history = {'loss': [], 'val_loss': [], 'weights': {'W1':[], 'W2':[]}, 'activations': []}
    
    def train(self, X, X_val=None, input_dim=5, n_epoch=100, hidden_dim=100, learning_rate=0.1, reg_param=0.0001, 
              noise='gaussian', noise_scale=1, verbose=False):
        
        dtype = torch.cuda.FloatTensor
        X_in = X.clone().type(dtype)
        X = X.type(dtype)
        N = X.shape[0]
        D = X.shape[1]
        
        self.strenghts = torch.zeros((1, D)).type(dtype)
        self.W1 = 0.0001*torch.randn(X.shape[1], hidden_dim).type(dtype)
        self.W2 = 0.0001*torch.randn(hidden_dim, input_dim).type(dtype)

        W = self.W1.mm(self.W2)
        s_int = np.linalg.svd(W.cpu().numpy(), compute_uv=0)
        self.init = np.sum(s_int[:100])/len(s_int)
        
        for t in range(n_epoch):
            
            W1 = self.W1
            W2 = self.W2
            
            # compute scores
            h1 = X_in.mm(W1)
            scores = h1.mm(W2)

            # compute loss
            data_loss = 0.5*((scores - X).pow(2).sum())/N
            noise_loss = 0.5*noise_scale*torch.trace(W2.t().mm(W1.t()).mm(W1).mm(W2))
            reg_loss = 0.5*reg_param*torch.sum(W1*W1) + 0.5*reg_param*torch.sum(W2*W2)
            loss = data_loss + noise_loss + reg_loss
                    
            # store loss
            self.history['loss'].append(loss)
        
            # compute the gradient using backpropagation
            dscores = (scores - X)/N
            dW2 = h1.t().mm(dscores)
            dhidden = dscores.mm(W2.t())
            dW1 = X_in.t().mm(dhidden)
            
            # add regularisation gradients
            dW2 += reg_param*W2
            dW1 += reg_param*W1
            
            # add noise gradients
            dW1 += noise_scale*W1.mm(W2).mm(W2.t())
            dW2 += noise_scale*W1.t().mm(W1).mm(W2)

            # perform parameter updates
            self.W1 += -learning_rate*dW1
            self.W2 += -learning_rate*dW2
            
            # compute validation loss
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = 0.5*(X_val - val_pred).pow(2).sum()/N
                self.history['val_loss'].append(val_loss)
                if verbose:
                    if t % 100 == 0:
                        print('iteration: ', t, 'training loss: ', loss, ' validation loss: ', val_loss)
            else:
                if verbose:
                    if t % 100 == 0:
                        print('iteration: ', t, 'training loss: ', loss)
            
            # compute eigenvalues
            if t % 100 == 0:
                W = self.W1.mm(self.W2)
                s = np.linalg.svd(W.cpu().numpy(), compute_uv=0)
                s = torch.unsqueeze(torch.from_numpy(s).type(dtype), 1).t()
                self.strenghts = torch.cat((self.strenghts, s), dim=0)
                
                self.history['weights']['W1'].append(W1.cpu().numpy().copy())
                self.history['weights']['W2'].append(W2.cpu().numpy().copy())
                self.history['activations'].append(h1.cpu().numpy().copy())
        
    def predict(self, X):
        W1 = self.W1
        W2 = self.W2
        h1 = X.type(torch.cuda.FloatTensor).mm(W1)
        scores = h1.mm(W2)
        return scores