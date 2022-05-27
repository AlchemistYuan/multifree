from typing import Union

import torch
from torch import nn
from torch import optim

from utils import *
from coor_transform import *
from loss import *


__all__ = ['Discriminator']


class Discriminator(nn.Module):
    """
    Implementation of a discriminator for the adversarial autoencoder.
    
    Attributes
    ----------
    discriminator : nn.Module
        The discriminator model
    
    Parameters
    ----------
    in_features : int, default=2
        The dimension of the input features
    discriminator_hidden_dims : list, default=[128,128]
        The dimensions of the each hidden layer in the discriminator model
    activation : str, default='relu'
        The activation function
    """
    def __init__(self, in_features: int=2, discriminator_hidden_dims: list=[128,128], 
                 activation='relu',) -> None:
        super(Discriminator, self).__init__()
        modules = []
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation function must be relu, leakyrelu, tanh, or sigmoid.')
            
        for i in range(len(discriminator_hidden_dims)):
            modules.append(nn.Sequential(nn.Linear(in_features, discriminator_hidden_dims[i]),
                                         self.activation))
            in_features = discriminator_hidden_dims[i]
        modules.append(nn.Sequential(nn.Linear(in_features, 1),))
        self.discriminator = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the discriminator model.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data
            
        Returns
        -------
        outputs : torch.Tensor
            The output of the discriminator.
        """
        outputs = self.discriminator(x)
        return outputs