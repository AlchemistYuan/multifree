import numpy as np
import MDAnalysis as mda


import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as T
from torch import nn
from torch import optim
from torch.utils import data
from torch.nn import functional as F

from .utils import *
from .coor_transform import *
from .loss import *


__all__ = [
    "Encoder", "Decoder", "AutoencoderBase", "DeterministicAutoencoder",
    "VariationalEncoder", "VAE", "Discriminator",
    "AAE", "SemiSupervisedAAE", "XYZDihderalAAE", "XYZDihderalSSAAE"  
]


class Encoder(nn.Module):
    """
    Define a deterministic encoder.
    """
    def __init__(self, activation: str='relu', in_features: int=100, latent_dim: int=2, hidden_dims: list=None) -> None:
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
        
        if hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        else:
            self.hidden_dims = hidden_dims
        self.encoder = self._encoder(self.activation, in_features=self.in_features, hidden_dims=self.hidden_dims)
        self.fc1 = nn.Linear(self.hidden_dims[-1], self.latent) 
        
    def _encoder(self, activation, in_features: int=100, hidden_dims: list=None) -> nn.Module:
        """
        A private method to build the encoder model.
        """
        modules = []
        for i in range(len(hidden_dims)):
            modules.append(nn.Sequential(nn.Linear(in_features, hidden_dims[i]),
                                         nn.BatchNorm1d(hidden_dims[i]),
                                         activation))
            in_features = hidden_dims[i]
        encoder = nn.Sequential(*modules)
        return encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data x into a set of parameters.
        
        Parameters
        ----------
        input: torch.Tensor
            Input data to encoder with shape (batchsize, nfeatures)
            
        Returns
        -------
        z: list of torch.Tensor
            The latent space codes with shape (batchsize, latent)
        """
        z = self.fc1(self.encoder(x))
        return z

class VariationalEncoder(Encoder):
    """
    Define a variational encoder.
    """
    def __init__(self, activation: str='relu', in_features: int=100, latent: int=2, hidden_dims: list=None) -> None:
        super(VariationalEncoder, self).__init__(activation, in_features, latent, hidden_dims)
        
        self.encoder = self._encoder(self.activation, in_features=self.in_features, hidden_dims=self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.latent) 
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
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
    
    def forward(self, x: torch.Tensor) -> list:
        """
        The forward pass of the VAE model.
        
        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batchsize, nfeatures)
            
        Returns
        -------
        
        """
        params = self.encoder(x)
        mu = self.fc1(params)
        log_var = self.fc2(params)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
class Decoder(nn.Module):
    """
    Define a dencoder.
    """
    def __init__(self, activation: str='relu', decoder_final_layer: str='linear',
                 out_features: int=100, latent_dim: int=2, hidden_dims: list=None) -> None:
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
        """
        modules = [nn.Linear(latent, hidden_dims[0])]
        in_features = hidden_dims[0]
        for i in range(1,len(hidden_dims)):
            modules.append(nn.Sequential(nn.Linear(in_features, hidden_dims[i]),
                                     nn.BatchNorm1d(hidden_dims[i]),
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
    
    ...

    Attributes
    ----------
    encoder : torch.nn.Module object
        The encoder model
    deoder : torch.nn.Module object
        The decoder model
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(AutoencoderBase, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._weight_initialization(self.encoder.encoder)
        self._weight_initialization(self.decoder.decoder)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
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
        
    def forward(self, x: torch.Tensor) -> list:
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
        
        Parameters
        ----------
        arg : torch.Tensor
            The input argument for sampling.
            
        Returns
        -------
        samples : torch.Tensor
            The samples (nsample, nfeatures)
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
    
    ...

    Attributes
    ----------
    encoder : torch.nn.Module object
        The encoder model
    deoder : torch.nn.Module object
        The decoder model
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(DeterministicAutoencoder, self).__init__(encoder, decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data x into a set of parameters.
        
        Parameters
        ----------
        input: torch.Tensor
            Input data to encoder with shape (batchsize, nfeatures)
            
        Returns
        -------
        params: list of torch.Tensor
            the mean and the variance of the gaussian distribution
        """
        z = self.encoder(x)
        return z

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
    
    def forward(self, x: torch.Tensor) -> list:
        """
        The forward pass of the VAE model.
        
        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batchsize, nfeatures)
            
        Returns
        -------
        
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def _sample(self, z) -> torch.Tensor:
        """
        Sample latent space from a trained model and pass through decoder.
        
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
    Define an variational autoencoder (VAE) by building the encoder and decoder models.

    Attributes
    ----------
    encoder : torch.nn.Module object
        The encoder model
    deoder : torch.nn.Module object
        The decoder model
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(VAE, self).__init__(encoder, decoder)

    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data x into a set of parameters.
        
        Parameters
        ----------
        input: torch.Tensor
            Input data to encoder with shape (batchsize, nfeatures)
            
        Returns
        -------
        params: list of torch.Tensor
            the mean and the variance of the gaussian distribution
        """
        z, mu, log_var = self.encoder(x)
        return z, mu, log_var
    
    def forward(self, x: torch.Tensor) -> list:
        """
        The forward pass of the VAE model.
        
        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batchsize, nfeatures)
            
        Returns
        -------
        
        """
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mu, log_var
    
    def _sample(self, nsample) -> torch.Tensor:
        """
        Sample latent space from a trained model and pass through decoder.
        
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

class Discriminator(nn.Module):
    """
    Implementation of a discriminator for the adversarial autoencoder.
    """
    def __init__(self, latent_dim: int=2, discriminator_hidden_dims: list=[128,128], 
                 activation='relu',) -> None:
        super(Discriminator, self).__init__()
        modules = []
        in_features = latent_dim
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
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the discriminator model.
        
        Parameters
        ----------
        z : torch.Tensor
            The latent codes with the shape of (batchsize, latent)
            
        Returns
        -------
        outputs : torch.Tensor
            The output of the discriminator, which is the probability of z coming from true distribution.
        """
        outputs = self.discriminator(z)
        return outputs
    
class AAE(nn.Module):
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module, 
                 optim_g, optim_d, generator_loss: nn.Module, discriminator_loss: nn.Module, 
                 true_prior: torch.distributions.Distribution, 
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss: bool=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(AAE, self).__init__()
        self.params = params
        self.variational = variational
        self.generator = generator
        self.discriminator = discriminator
        self.whitening = whitening
        self.verbose = verbose
        self.true_prior = true_prior
        self.whitened_loss = whitened_loss
       
        # The whitening layer 
        if self.whitening:
            if (PCAWhitening is None) or (PCAUnWhitening is None):
                raise ValueError('Please specify whitening and unwhitening methods.')
            else:
                self.pcawhitening = PCAWhitening
                self.pcaunwhitening = PCAUnWhitening
        else:
            self.pcawhitening = None
            self.pcaunwhitening = None

        # Optimizers for the generator and the discriminator
        self.optim_G = optim_g
        self.optim_D = optim_d
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optim_G, gamma=0.99)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optim_D, gamma=0.99)

        # The loss function for the generator and the discriminator
        self.criterion_G = generator_loss
        self.criterion_D = discriminator_loss
    
        self.loss_g_train = {}
        self.loss_g_val = {}
        self.loss_d_train = []
        
        self.z_real = torch.ones((self.params['batchsize'], 1)).to(self.params['device'])
        self.z_fake = torch.zeros((self.params['batchsize'], 1)).to(self.params['device'])
        
    def train_generator(self, x_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train the generator.
        
        Parameters
        ----------
        x_batch : data.Dataset
            The minibatch to be trained
        """
        x_hat, args = self.generator(x_batch)
        if not isinstance(args, list):
            generator_args = [args]
        else:
            generator_args = args
        return x_hat, generator_args
    
    def train_discriminator(self, z: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_real = self.discriminator(noise)
        target_generator = self.discriminator(z.detach())
        real_loss = self.criterion_D(target_real, self.z_real)
        fake_loss = self.criterion_D(target_generator, self.z_fake)
        loss_d = 0.5 * (real_loss + fake_loss)
        return target_real, target_generator, loss_d
    
    def sample(self, nsample: int=100, z: torch.Tensor=None) -> torch.Tensor:
        if self.variational:
            samples = self.generator.sample(nsample)
        else:
            samples = self.generator.sample(z)
        if self.whitening:
            samples = self.pcaunwhitening(samples)
        return samples
    
    def forward(self, train_data, val_data=None) -> None:
        """
        The forward pass of the AAE model.
        
        Parameters
        ----------
        train_data : data.Dataloader
            A dataloader for the training dataset
        val_data : data.Dataloader
            A dataloader for the validation dataset
            
        Returns
        -------
        None
        """
        for i in range(self.params['nepoch']):
            self._train_one_epoch(i, train_data, val_data)
    
    def _train_one_epoch(self, i: int, train_data, val_data, train_label=None, val_label=None) -> None:
        # Enumerate each minibatch training data
        for j, train_batch in enumerate(train_data):
            loss_g_dict, loss_d = self._train_one_step(j, train_batch)
                
        # Record loss after each epoch
        if i == 0:
            keys = list(loss_g_dict.keys())
            for k in keys:
                self.loss_g_train[k] = [loss_g_dict[k].item()]
                self.loss_g_val[k] = []
        else:
            keys = list(loss_g_dict.keys())
            for k in keys:
                self.loss_g_train[k].append(loss_g_dict[k].item())
                
        self.loss_d_train.append(loss_d.item())
        
        # Report loss
        if self.verbose:
            report = 'TRAIN LOSS - Generator '
            keys = list(loss_g_dict.keys())
            for k in keys:
                report += '{0:s}: {1:f} '.format(k, loss_g_dict[k].item())
            report = report[:-1] + '; Discriminator loss {0:f}'.format(loss_d)
            print(report[:-1])
   
        # Update scheduler
        self.scheduler_G.step()
        self.scheduler_D.step()
 
    def _train_one_step(self, j: int, train_batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(train_batch, list):
            x_batch = train_batch[0].to(self.params['device'])
        else:
            x_batch = train_batch.to(self.params['device'])
        if self.whitening:
            x = self.pcawhitening(x_batch)
            assert x.shape[0] == x_batch.shape[0]
            assert x.shape[1] == x_batch.shape[1] - self.params['dof']
        else:
            x = x_batch
            assert x.shape == x_batch.shape
            
        # Train the generator
        x_hat, generator_args = self.train_generator(x)
        z = generator_args[0]
        if self.whitening:
            x_rec = self.pcaunwhitening(x_hat)
        else:
            x_rec = x_hat
        # Compute generator loss
        loss_g_dict = self.criterion_G(x_batch, x_rec, generator_args[1:], self.params['beta'])
        loss_g = 0.999 * loss_g_dict['loss'] + 0.001 * self.criterion_D(self.discriminator(z), self.z_real)
        # Backward pass
        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()
                
        # Train the discriminator
        noise = self.true_prior.sample(sample_shape=torch.tensor([self.params['batchsize'], 
                                       self.params['latent']])).to(self.params['device'])
        target_real, target_generator, loss_d = self.train_discriminator(z, noise)
        self.optim_D.zero_grad()
        loss_d.backward()
        self.optim_D.step()
        
        return loss_g_dict, loss_d

class SemiSupervisedAAE(AAE):
    """
    Define a semi-supervised adversarial autoencoder. 
    Label information is incorporated with latent codes for the discriminator.

    Attributes
    ----------
         
    """
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module, 
                 optim_g, optim_d, generator_loss, discriminator_loss, true_prior, 
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(SemiSupervisedAAE, self).__init__(params, generator, discriminator, 
                                                optim_g, optim_d, generator_loss, discriminator_loss, true_prior, 
                                                PCAWhitening, PCAUnWhitening, whitened_loss,
                                                variational, whitening, verbose)
    
    def _label_to_one_hot_vector(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Turn the label tensor to tensor of one-hot vector.
        
        Parameters
        labels : torch.Tensor
            The label tensor. The elements should be within [0,nclass-1]
            
        Returns
        -------
        labels_one_hot : torch.Tensor
            The tensor of one-hot vector with the shape of (batch_size, nclass)
        """
        labels_one_hot = torch.zeros((self.params['batchsize'], self.params['nclass']), dtype=torch.int32)
        for k, label in enumerate(labels):
            label = label.int()
            labels_one_hot[k, label] = label
        return labels_one_hot
                
    def _train_one_step(self, j: int, train_batch) -> tuple[torch.Tensor, torch.Tensor]:
        x_batch = train_batch[0].to(self.params['device'])
        label_batch = train_batch[1].to(self.params['device'])
        labels_one_hot = self._label_to_one_hot_vector(label_batch).to(self.params['device'])
        if self.whitening:
            x = self.pcawhitening(x_batch)
            assert x.shape[0] == x_batch.shape[0]
            assert x.shape[1] == x_batch.shape[1] - self.params['dof']
        else:
            x = x_batch
            assert x.shape == x_batch.shape
            
        # Train the generator
        x_hat, generator_args = self.train_generator(x)
        z = generator_args[0]
        z_label_batch = torch.cat((z, labels_one_hot), dim=1)

        if self.whitening:
            x_rec = self.pcaunwhitening(x_hat)
        else:
            x_rec = x_hat
        # Compute generator loss
        loss_g_dict = self.criterion_G(x_batch, x_rec, generator_args[1:], self.params['beta'])
        loss_g = 0.99 * loss_g_dict['loss'] + 0.01 * self.criterion_D(self.discriminator(z_label_batch), self.z_real)
        # Backward pass
        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()
                
        # Train the discriminator
        noise, noise_one_hot_label = self.true_prior.sample(sample_shape=torch.tensor([self.params['batchsize'], 
                                                            self.params['latent']]), 
                                                            return_components=True)
        noise_label_batch = torch.cat((noise, noise_one_hot_label), dim=1)
        target_real, target_generator, loss_d = self.train_discriminator(z_label_batch, noise_label_batch)
        self.optim_D.zero_grad()
        loss_d.backward()
        self.optim_D.step()        
        return loss_g_dict, loss_d

class SemiSupervisedAAEwithGP(AAE):
    """
    Define a semi-supervised adversarial autoencoder with gradient penalty on distriminator loss. 
    Label information is incorporated with latent codes for the discriminator.

    Attributes
    ----------
         
    """
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module,
                 optim_g, optim_d, generator_loss, discriminator_loss, true_prior,
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(SemiSupervisedAAEwithGP, self).__init__(params, generator, discriminator,
                                                optim_g, optim_d, generator_loss, discriminator_loss, true_prior,
                                                PCAWhitening, PCAUnWhitening, whitened_loss,
                                                variational, whitening, verbose)

    def _gradient_penalty(z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient penalty term.

        Parameters
        ----------
        z : torch.Tensor
            The generated latent variables
        noise : torch.Tensor
            The true latent variables

        Returns
        -------
        gp : torch.Tensor
            The computed gradient penalty
        """
        eps = torch.rand(z.shape[0], z.shape[1])
        z_bar = eps * noise + (1 - eps) * z
        target_z_bar = self.discriminator(z_bar)

        return gp

    def train_discriminator(self, z: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gp = self._gradient_penalty(z, noise)
        target_real = self.discriminator(noise)
        target_generator = self.discriminator(z.detach())
        loss_d = torch.mean(target_generator) - torch.mean(target_real) + self.gpfactor * gp
        return target_real, target_generator, loss_d
 
class XYZDihderalAAE(AAE):
    """
    The XYZDihderalAAE model takes xyz coordinates and phi/psi dihedrals as input.
    The output is the reconstructed xyz coordinates.
    The generator and discriminator archetectures are the same as the standard AAE model.
    The difference lies in the input of the generator: a combination of xyz and phi/psi (in sin/cos space).
    During the traning of the generator, the xyz parts are optionally whitened. 
    """
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module, 
                 optim_g, optim_d, generator_loss, discriminator_loss, true_prior,
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(XYZDihderalAAE, self).__init__(params, generator, discriminator, 
                                             optim_g, optim_d, generator_loss, discriminator_loss, true_prior, 
                                             PCAWhitening, PCAUnWhitening, whitened_loss,
                                             variational, whitening, verbose)
        self.split = self.params['split']
    
    def _train_one_step(self, j: int, train_batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(train_batch, list):
            x_batch = train_batch[0].to(self.params['device'])
        else:
            x_batch = train_batch.to(self.params['device'])
        if self.whitening:
            if self.split <= 0:
                split = x_batch.shape[1]
            else:
                split = x_batch.shape[1] - self.split
            x = self.pcawhitening(x_batch[:,:split])
            x_cat = torch.cat((x, x_batch[:,split:]), dim=1)
            assert x_cat.shape[0] == x_batch.shape[0]
            assert x_cat.shape[1] == x_batch.shape[1] - self.params['dof']
        else:
            x_cat = x_batch
            split = 0
            assert x_cat.shape[0] == x_batch.shape[0]
            assert x_cat.shape[1] == x_batch.shape[1]
            
        # Train the generator
        x_hat, generator_args = self.train_generator(x_cat)
        z = generator_args[0]
        if self.whitening:
            x_rec = self.pcaunwhitening(x_hat)
        else:
            x_rec = x_hat
        # Compute the phi/psi angles from the reconstructed xyz coordinates
        sincos_phipsi = self.xyz2dihedrals(x_rec, idx=self.params['idx'])
        # If the reconstruction loss is computed on the whitened data
        if self.whitened_loss:
            # Combine the whitened xyz coordinates and the computed phi/psi
            x_hat_sincos_phipsi = torch.cat((x_hat, sincos_phipsi), dim=1).to(self.params['device'])
            loss_g_dict = self.criterion_G(x_cat, x_hat_sincos_phipsi, generator_args[1:], 
                                       self.params['beta'], x_hat.shape[1])
        else:
            # Combine the unwhitened xyz coordinates and the computed phi/psi
            x_rec_sincos_phipsi = torch.cat((x_rec, sincos_phipsi), dim=1).to(self.params['device'])
            loss_g_dict = self.criterion_G(x_batch, x_rec_sincos_phipsi, generator_args[1:], 
                                       self.params['beta'], x_rec.shape[1])
        # Compute generator loss
        
        loss_g = 0.999 * loss_g_dict['loss'] + 0.001 * self.criterion_D(self.discriminator(z), self.z_real)
        # Backward pass
        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()
                
        # Train the discriminator
        noise = self.true_prior.sample(sample_shape=torch.tensor([self.params['batchsize'], 
                                       self.params['latent']])).to(self.params['device'])
        target_real, target_generator, loss_d = self.train_discriminator(z, noise)
        # Backward pass
        self.optim_D.zero_grad()
        loss_d.backward()
        self.optim_D.step()
        return loss_g_dict, loss_d
    
    def xyz2dihedrals(self, xyz: torch.Tensor, idx: list=None) -> torch.Tensor:
        xyz_reshaped = xyz.reshape(xyz.shape[0], xyz.shape[1]//3, 3)
        phi_atoms = xyz_reshaped[:,idx[0],:]
        psi_atoms = xyz_reshaped[:,idx[1],:]
        
        # Get the phi/psi angles in radians
        phi_angles = self._compute_dihedral(phi_atoms[:,0,:], phi_atoms[:,1,:], phi_atoms[:,2,:], phi_atoms[:,3,:])
        psi_angles = self._compute_dihedral(psi_atoms[:,0,:], psi_atoms[:,1,:], psi_atoms[:,2,:], psi_atoms[:,3,:])
        sincos_phipsi = torch.zeros((phi_angles.shape[0], 4), device=self.params['device'])
        sincos_phipsi[:,0] = torch.sin(phi_angles)
        sincos_phipsi[:,1] = torch.cos(phi_angles)
        sincos_phipsi[:,2] = torch.sin(psi_angles)
        sincos_phipsi[:,3] = torch.cos(psi_angles)
        return sincos_phipsi
    
    def _compute_angle(self, p1, p2) -> torch.Tensor:
        """
        https://stackoverflow.com/questions/56918164/use-tensorflow-pytorch-to-speed-up-minimisation-of-a-custom-function
        """
        inner_product = (p1*p2).sum(dim=-1)
        p1_norm = torch.linalg.norm(p1, dim=-1)
        p2_norm = torch.linalg.norm(p2, dim=-1)
        cos = inner_product / (p1_norm * p2_norm)
        cos = torch.clamp(cos, -0.99999, 0.99999)
        angle = torch.acos(cos)
        return angle
    
    def _compute_dihedral(self, v1, v2, v3, v4) -> torch.Tensor:
        """
        https://stackoverflow.com/questions/56918164/use-tensorflow-pytorch-to-speed-up-minimisation-of-a-custom-function
        """
        ab = v1 - v2
        cb = v3 - v2
        db = v4 - v3
        u = torch.linalg.cross(ab, cb, dim=-1)
        v = torch.linalg.cross(db, cb, dim=-1)
        w = torch.linalg.cross(u, v, dim=-1)
        angle = self._compute_angle(u, v)
        angle = torch.where(self._compute_angle(cb, w) > 1, -angle, angle)
        return angle

class XYZDihderalSSAAE(XYZDihderalAAE):
    """
    The semi-supervised XYZDihderalAAE model.
    """
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module, 
                 optim_g, optim_d, generator_loss, discriminator_loss, true_prior,
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(XYZDihderalSSAAE, self).__init__(params, generator, discriminator, 
                                               optim_g, optim_d, generator_loss, discriminator_loss, true_prior, 
                                               PCAWhitening, PCAUnWhitening, whitened_loss,
                                               variational, whitening, verbose)
        self.split = self.params['split']
    
    def _label_to_one_hot_vector(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Turn the label tensor to tensor of one-hot vector.
        
        Parameters
        labels : torch.Tensor
            The label tensor. The elements should be within [0,nclass-1]
            
        Returns
        -------
        labels_one_hot : torch.Tensor
            The tensor of one-hot vector with the shape of (batch_size, nclass)
        """
        labels_one_hot = torch.zeros((self.params['batchsize'], self.params['nclass']), dtype=torch.int32)
        for k, label in enumerate(labels):
            label = label.int()
            labels_one_hot[k, label] = label
        return labels_one_hot
    
    def _train_one_step(self, j: int, train_batch) -> tuple[torch.Tensor, torch.Tensor]:
        x_batch = train_batch[0].to(self.params['device'])
        label_batch = train_batch[1].to(self.params['device'])
        labels_one_hot = self._label_to_one_hot_vector(label_batch).to(self.params['device'])
        
        if self.whitening:
            if self.split <= 0:
                split = x_batch.shape[1]
            else:
                split = x_batch.shape[1] - self.split
            x = self.pcawhitening(x_batch[:,:split])
            x_cat = torch.cat((x, x_batch[:,split:]), dim=1)
            assert x_cat.shape[0] == x_batch.shape[0]
            assert x_cat.shape[1] == x_batch.shape[1] - self.params['dof']
        else:
            x_cat = x_batch
            split = 0
            assert x_cat.shape[0] == x_batch.shape[0]
            assert x_cat.shape[1] == x_batch.shape[1]
            
        # Train the generator
        x_hat, generator_args = self.train_generator(x_cat)
        z = generator_args[0]
        z_label_batch = torch.cat((z, labels_one_hot), dim=1)
        
        if self.whitening:
            x_rec = self.pcaunwhitening(x_hat)
        else:
            x_rec = x_hat
        # Compute the phi/psi angles from the reconstructed xyz coordinates
        sincos_phipsi = self.xyz2dihedrals(x_rec, idx=self.params['idx'])
        # If the reconstruction loss is computed on the whitened data
        if self.whitened_loss:
            # Combine the whitened xyz coordinates and the computed phi/psi
            x_hat_sincos_phipsi = torch.cat((x_hat, sincos_phipsi), dim=1).to(self.params['device'])
            loss_g_dict = self.criterion_G(x_cat, x_hat_sincos_phipsi, generator_args[1:], 
                                       self.params['beta'], x_hat.shape[1])
        else:
            # Combine the unwhitened xyz coordinates and the computed phi/psi
            x_rec_sincos_phipsi = torch.cat((x_rec, sincos_phipsi), dim=1).to(self.params['device'])
            loss_g_dict = self.criterion_G(x_batch, x_rec_sincos_phipsi, generator_args[1:], 
                                       self.params['beta'], x_rec.shape[1])
        # Compute generator loss
        
        loss_g = 0.999 * loss_g_dict['loss'] + 0.001 * self.criterion_D(self.discriminator(z_label_batch), self.z_real)
        # Backward pass
        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()
                
        # Train the discriminator
        noise, noise_one_hot_label = self.true_prior.sample(sample_shape=torch.tensor([self.params['batchsize'], 
                                                            self.params['latent']]), 
                                                            return_components=True)
        noise_label_batch = torch.cat((noise, noise_one_hot_label), dim=1)
        target_real, target_generator, loss_d = self.train_discriminator(z_label_batch, noise_label_batch)
        # Backward pass
        self.optim_D.zero_grad()
        loss_d.backward()
        self.optim_D.step()
        return loss_g_dict, loss_d
