from typing import Union

import torch
from torch import nn
from torch import optim

from multifree.utils.utils import *
from multifree.utils.coor_transform import *
from multifree.utils.loss import *


class Encoder(nn.Module):
    """
    Define a deterministic encoder.
    
    Parameters
    ----------
    activation : str, default='relu'
        The activation function
    in_features : int, default=100
        The dimension of the input features
    latent_dim : int, default=2
        The dimension of the latent space
    hidden_dims : list, default=None
        The list of the dimensions of each hidden layer
    batchnorm : bool, default=False
        Whether to use batch normalization before each activation function
    """
    def __init__(self, activation: str='relu', in_features: int=100, latent_dim: int=2, hidden_dims: list=None, batchnorm: bool=False) -> None:
        super(Encoder, self).__init__()
        self.hidden_dims = hidden_dims
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
            
        self.in_features = in_features
        self.latent = latent_dim
        self.batchnorm = batchnorm 
        if hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        else:
            self.hidden_dims = hidden_dims
        self.encoder = self._encoder(self.activation, in_features=self.in_features, hidden_dims=self.hidden_dims)
        self.fc1 = nn.Linear(self.hidden_dims[-1], self.latent) 
        
    def _encoder(self, activation, in_features: int=100, hidden_dims: list=None) -> nn.Module:
        """
        A private method to build the encoder model.
        
        Parameters
        ----------
        activation : nn.Module
            The activation function
        in_features : int, default=100
            The dimension of the input features
        hidden_dims : list, default=None
            The list of the dimensions of each hidden layer
        
        Returns
        -------
        encoder : nn.Module
            The encoder model
        """
        modules = []
        for i in range(len(hidden_dims)):
            if self.batchnorm:
                modules.append(nn.Sequential(nn.Linear(in_features, hidden_dims[i]),
                                             nn.BatchNorm1d(hidden_dims[i]),
                                             activation))
            else:
                modules.append(nn.Sequential(nn.Linear(in_features, hidden_dims[i]),
                                             activation))
            in_features = hidden_dims[i]
        encoder = nn.Sequential(*modules)
        return encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data x into a set of parameters.
        
        Parameters
        ----------
        x: torch.Tensor
            Input data to encoder with shape (batchsize, nfeatures)
            
        Returns
        -------
        z: torch.Tensor
            The latent space codes with shape (batchsize, latent)
        """
        z = self.fc1(self.encoder(x))
        return z
    
class VariationalEncoder(Encoder):
    """
    Define a variational encoder.
    
    Parameters
    ----------
    activation : str, default='relu'
        The activation function
    in_features : int, default=100
        The dimension of the input features
    latent_dim : int, default=2
        The dimension of the latent space
    hidden_dims : list, default=None
        The list of the dimensions of each hidden layer
    batchnorm : bool, default=False
        Whether to use batch normalization before each activation function
    """
    def __init__(self, activation: str='relu', in_features: int=100, latent: int=2, 
                 hidden_dims: list=None, batchnorm: bool=False) -> None:
        super(VariationalEncoder, self).__init__(activation, in_features, latent, hidden_dims, batchnorm)
        
        self.encoder = self._encoder(self.activation, in_features=self.in_features, hidden_dims=self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.latent) 
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample N(mu, var) from N(0,1).
        
        Parameters
        ----------
        mu: torch.Tensor 
            Mean of the latent Gaussian with shape (batchsize, latent)
        log_var: torch.Tensor 
            Standard deviation of the latent Gaussian with shape (batchsize, latent)
            
        Returns
        -------
        z: torch.Tensor
            The latent space variables with shape (batchsize, latent)
        """
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass of the VAE model.
        
        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batchsize, nfeatures)
            
        Returns
        -------
        z: torch.Tensor
            The latent space variables with shape (batchsize, latent)
        mu: torch.Tensor 
            Mean of the latent Gaussian with shape (batchsize, latent)
        log_var: torch.Tensor 
            Standard deviation of the latent Gaussian with shape (batchsize, latent)
        """
        params = self.encoder(x)
        mu = self.fc1(params)
        log_var = self.fc2(params)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
class Decoder(nn.Module):
    """
    Define a dencoder.
    
    Parameters
    ----------
    activation : str, default='relu'
        The activation function
    decoder_final_layer : str, default='linear'
        The activation function of the final layer of the decoder
    out_features : int, default=100
        The dimension of the output features
    latent_dim : int, default=2
        The dimension of the latent space
    hidden_dims : list, default=None
        The list of the dimensions of each hidden layer
    batchnorm : bool, default=False
        Whether to use batch normalization before each activation function
    """
    def __init__(self, activation: str='relu', decoder_final_layer: str='linear',
                 out_features: int=100, latent_dim: int=2, 
                 hidden_dims: list=None, batchnorm: bool=False) -> None:
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
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
            
        self.out_features = out_features
        self.latent = latent_dim
        self.decoder_final_layer = decoder_final_layer
        self.batchnorm = batchnorm 
        if hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        else:
            self.hidden_dims = hidden_dims
            
        self.decoder = self._decoder(self.activation, final_layer=self.decoder_final_layer, 
                                     out_features=self.out_features, latent=self.latent, 
                                     hidden_dims=self.hidden_dims[::-1]) 
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input data from the latent space.
        
        Parameters
        ----------
        z: torch.Tensor
            The latent space variable with shape (batchsize, latent)
            
        Returns
        -------
        x_hat: torch.Tensor
            The reconstructed input data with shape (batchsize, nfeatures)
        """
        x_hat = self.decoder(z)
        return x_hat
    
    def _decoder(self, activation, final_layer: str='linear',
                 out_features: int=100, latent: int=2, hidden_dims: list=None) -> nn.Module:
        """
        A private method to build the decoder model.
        
        Parameters
        ----------
        activation : str
            The activation function
        decoder_final_layer : str, default='linear'
            The activation function of the final layer of the decoder
        out_features : int, default=100
            The dimension of the output features
        latent_dim : int, default=2
            The dimension of the latent space
        hidden_dims : list, default=None
            The list of the dimensions of each hidden layer
            
        Returns
        -------
        decoder : nn.Module
            The decoder model
        """
        modules = [nn.Linear(latent, hidden_dims[0])]
        in_features = hidden_dims[0]
        for i in range(1,len(hidden_dims)):
            if self.batchnorm:
                modules.append(nn.Sequential(nn.Linear(in_features, hidden_dims[i]),
                                             nn.BatchNorm1d(hidden_dims[i]),
                                             activation))
            else:
                modules.append(nn.Sequential(nn.Linear(in_features, hidden_dims[i]),
                                             activation))
            in_features = hidden_dims[i]
        if final_layer == 'sigmoid':
            modules.append(nn.Sequential(nn.Linear(in_features, out_features),nn.Sigmoid()))
        elif final_layer == 'tanh':
            modules.append(nn.Sequential(nn.Linear(in_features, out_features),nn.Tanh()))
        elif final_layer == 'hardtanh':
            modules.append(nn.Sequential(nn.Linear(in_features, out_features),nn.Hardtanh()))
        elif final_layer == 'linear':
            modules.append(nn.Sequential(nn.Linear(in_features, out_features),))
        else:
            raise ValueError('Activation function in the final layer must be linear, sigmoid, or tanh.')
        decoder = nn.Sequential(*modules)
        return decoder
    

class AutoencoderBase(nn.Module):
    """
    Base class for all autoencoder model.
    
    Parameters
    ----------
    encoder : nn.Module
        The encoder model
    decoder : nn.Module
        The decoder model
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(AutoencoderBase, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._weight_initialization(self.encoder.encoder)
        self._weight_initialization(self.decoder.decoder)
        
    def encode(self, x: torch.Tensor) -> Union[torch.Tensor, tuple]:
        """
        Map the input data into the latent space.
        This method will be overriden in subclasses.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data           
        """
        raise NotImplementedError
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input data from the latent space.
        
        Parameters
        ----------
        z: torch.Tensor
            The latent space variable with shape (batchsize, latent)
            
        Returns
        -------
        x_hat: torch.Tensor
            The reconstructed input data with shape (batchsize, nfeatures)
        """
        x_hat = self.decoder(z)
        return x_hat
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        The forward pass of the autoencoder model.
        This method will be overriden in subclasses.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data           
        """
        raise NotImplementedError
    
    def sample(self, arg: torch.Tensor) -> torch.Tensor:
        """
        Sample latent space from a trained model and pass through decoder.
        
        Parameters
        ----------
        arg : torch.Tensor
            The input argument for sampling.
            
        Returns
        -------
        samples : torch.Tensor
            The samples (nsample, nfeatures)
        """
        samples = self._sample(arg)
        return samples
    
    def _sample(self, arg: torch.Tensor) -> torch.Tensor:
        """
        Sample latent space from a trained model and pass through decoder.
        This method will be overriden in subclasses.
        
        Parameters
        ----------
        arg : torch.Tensor
            The input argument for sampling.
        """
        raise NotImplementedError
        
    def _weight_initialization(self, model: nn.Module) -> None:
        """
        Apply xavier normal initialization to weights of each linear layer.
        """
        nlayer = len(model)
        for i in range(nlayer):
            name = model[i].__class__.__name__
            if name == 'Sequential':
                torch.nn.init.normal_(model[i][0].weight, std=0.01)
            elif name == 'Linear':
                torch.nn.init.normal_(model[i].weight, std=0.01)
                
class DeterministicAutoencoder(AutoencoderBase):
    """
    Define a deterministic autoencoder by building the encoder and decoder models.
    
    Parameters
    ----------
    encoder : nn.Module
        The encoder model
    decoder : nn.Module
        The decoder model
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(DeterministicAutoencoder, self).__init__(encoder, decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data x into a set of parameters.
        Overriden from the parent class.
        
        Parameters
        ----------
        input: torch.Tensor
            Input data to encoder with shape (batchsize, nfeatures)
            
        Returns
        -------
        z: torch.Tensor
            The latent space variables
        """
        z = self.encoder(x)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass of the deterministic autoencoder model.
        Overriden from the parent class.
        
        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batchsize, nfeatures)
            
        Returns
        -------
        x_hat: torch.Tensor
            The reconstructed input data with shape (batchsize, nfeatures)
        z: torch.Tensor
            The latent space variables
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def _sample(self, z) -> torch.Tensor:
        """
        Sample latent space from a trained model and pass through decoder.
        Overriden from the parent class.
        
        Parameters
        ----------
        z : torch.Tensor
            The low-dimensinoal codes with shape (nsample, latent)
            
        Returns
        -------
        samples : torch.Tensor
            The samples (nsample, nfeatures)
        """
        samples = self.decode(z)
        return samples


class VAE(AutoencoderBase):
    """
    Define an variational autoencoder (VAE) with a gaussian prior by building the encoder and decoder models.

    Parameters
    ----------
    encoder : nn.Module
        The variational encoder model
    decoder : nn.Module
        The decoder model
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(VAE, self).__init__(encoder, decoder)

    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the input data x into a set of parameters.
        Overriden from the parent class.
        
        Parameters
        ----------
        input: torch.Tensor
            Input data to encoder with shape (batchsize, nfeatures)
            
        Returns
        -------
        z : torch.Tensor
            The latent space variables sampled from the gaussian prior
        mu : torch.Tensor
            The mean of the gaussian distribution
        log_var : torch.Tensor
            the log variance of the gaussian distribution
        """
        z, mu, log_var = self.encoder(x)
        return z, mu, log_var
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass of the VAE model.
        Overriden from the parent class.
        
        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batchsize, nfeatures)
            
        Returns
        -------
        x_hat: torch.Tensor
            The reconstructed input data with shape (batchsize, nfeatures)
        z : torch.Tensor
            The latent space variables sampled from the gaussian prior
        mu : torch.Tensor
            The mean of the gaussian distribution
        log_var : torch.Tensor
            the log variance of the gaussian distribution 
        """
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mu, log_var
    
    def _sample(self, nsample) -> torch.Tensor:
        """
        Sample latent space from a trained model and pass through decoder.
        Overriden from the parent class.
        
        Parameters
        ----------
        nsample : int
            The number of samples to be sampled
        device : str
            The current device.
            
        Returns
        -------
        samples : torch.Tensor
            The samples (nsample, nfeatures)
        """
        if nsample.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        z = torch.randn(nsample, self.latent).to(device)
        samples = self.decode(z)
        return samples
