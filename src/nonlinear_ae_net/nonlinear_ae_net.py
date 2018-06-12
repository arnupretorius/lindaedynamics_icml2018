import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

# custom import 
from src.linear_ae_net.dynamics import compute_correlation_matrix

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din

def create_model(D_in, H, D_out, act_fn='tanh', noise=None):
    # define layers
    l1 = nn.Linear(D_in, H, bias=False)
    l1.weight.data.normal_(0, 0.001)
    l2 = nn.Linear(H, D_out, bias=False)
    l2.weight.data.normal_(0, 0.001)
    
    if act_fn=='tanh':
        act_function = nn.Tanh()
    elif act_fn=='relu':
        act_function = nn.ReLU()
    else:
        print("Not a valid activation function.")
        return
    
    if noise is None:
        model = nn.Sequential(l1, act_function, l2).cuda()
    else:
        model = nn.Sequential(GaussianNoise(noise), l1, act_function, l2).cuda()
    return model

def train_model(x, model, loss_criterion, n_epoch, learning_rate, weight_decay=0, noise=None, num_eig=4):
    
    # learning dynamics
    dynamics = np.zeros((1, num_eig))
    
    # compute true input-input covariance matrix
    x_np = x.data.cpu().numpy()
    corr_mat = compute_correlation_matrix(x_np, x_np)
    U, S, V = np.linalg.svd(corr_mat)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for t in range(n_epoch):
        # Forward pass
        x_pred = model(x)
        loss = loss_criterion(x_pred, x)
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # update parameters
        optimizer.step()
        
        # compute eigen-values (for higher resolution change frequency of computation)
        if t % 10 == 0:
            corr_mat_hat = compute_correlation_matrix(x_np, x_pred.data.cpu().numpy())
            S_hat = np.diag(U.T.dot(corr_mat_hat).dot(V.T))
            S_true = S[:4]
            S_pred = S_hat[:4]
            I_pred = S_pred/S_true
            dynamics = np.vstack((dynamics, I_pred))
            
    return dynamics