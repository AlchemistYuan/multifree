import torch
from torch import nn
from torch import optim

from multifree.utils.utils import *
from multifree.utils.coor_transform import *
from multifree.utils.loss import *


__all__ = [
    "AAE", "SupervisedAAE", "EnergyGapSAAE"  
]

class AAE(nn.Module):
    """
    The adversarial autoencoder model
    
    Parameters
    ----------
    params : dict
        The dictionary of the model parameters
    generator : nn.Module
        The generative model in AAE
    discriminator : nn.Module
        The discriminative model in AAE
    optim_g : optim.Optimizer
        The optimizer for the generator
    optim_d : optim.Optimizer
        The optimizer for the discriminator
    generator_loss : nn.Module
        The loss function for the generator
    discriminator_loss : nn.Module
        The loss function for the discriminator
    true_prior : torch.distributions.Distribution
        The true prior distribution for the latent space of the generator
    PCAWhitening : nn.Module, default=None
        The PCA whitening layer
    PCAUnWhitening : nn.Module, default=None
        The PCA un-whitening layer
    whitened_loss : bool, default=True
        Whether calcualte on the whitened or un-whitened output
    variational : bool, default=True
        Whether the encoder is variational
    whitening : bool, default=True
        Whether to perform whitening on the input data and unwhitening on the output data
    verbose : bool, defatult=True
        Control how much information will be printed as the training progresses
    """
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
        if self.params['schedulerg']:
            self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optim_G, gamma=0.99)
        else:
            self.scheduler_G = None
        if self.params['schedulerd']:
            self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optim_D, gamma=0.99)
        else:
            self.scheduler_D = None
        # The loss function for the generator and the discriminator
        self.criterion_G = generator_loss
        self.criterion_D = discriminator_loss
    
        self.loss_g_train = {}
        self.loss_g_val = {}
        self.loss_d_train = []
        self.loss_d_val = []
        
        self.z_real = torch.ones((self.params['batchsize'], 1)).to(self.params['device'])
        self.z_fake = torch.zeros((self.params['batchsize'], 1)).to(self.params['device'])
        
    def train_generator(self, x_batch) -> tuple[torch.Tensor, tuple]:
        """
        Train the generator.
        
        Parameters
        ----------
        x_batch : torch.utils.data.Dataset
            The minibatch to be trained
            
        Returns
        -------
        x_hat : torch.Tensor
            The reconstructed input data from the decoder
        other_args : tuple
            The output from the encoder.
            If the encoder is deterministic, the output is the latent space codes z.
            If the encoder is variational, the outputs are z, mu, and log_var.
        """
        args = self.generator(x_batch)
        x_hat = args[0]
        other_args = args[1:]
        assert isinstance(other_args, tuple)
        return x_hat, other_args
    
    def train_discriminator(self, z: torch.Tensor, noise: torch.Tensor, z_real: torch.Tensor=None, z_fake: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train the discriminator.
        
        Parameters
        ----------
        z : torch.Tensor
            The input data for the discriminator, which is the latent space from the encoder.
        noise : torch.Tensor
            The noise input vector with the same size as the z which is drawn from the true prior distribution.
        
        Returns
        -------
        target_real : torch.Tensor
            The prediction of the discriminator for the noise input
        target_generator : torch.Tensor
            The prediction of the discriminator for the latent space from the encoder
        loss_d : torch.Tensor
            The discriminator loss value
        z_real : torch.Tensor
            The real latent space
        z_fake : torch.Tensor
            The fake latent space
        """
        if z_real is None:
            z_real = self.z_real
        if z_fake is None:
            z_fake = self.z_fake
        target_real = self.discriminator(noise)
        target_generator = self.discriminator(z.detach())
        real_loss = self.criterion_D(target_real, z_real)
        fake_loss = self.criterion_D(target_generator, z_fake)
        loss_d = 0.5 * (real_loss + fake_loss)
        return target_real, target_generator, loss_d
    
    def sample(self, nsample: int=100) -> torch.Tensor:
        """
        Sample from the true prior and pass the samples through the decoder.
        
        Parameters
        ----------
        nsample : int, default=100
            The number of samples to be generated
        
        Returns
        -------
        samples : torch.Tensor
            The reconstructed data from the sampled latent space
        """
        z = self.true_prior.sample(sample_shape=torch.tensor([nsample, self.params['latent']])).to(self.params['device'])
        samples = self.generator.decode(z)
        if self.whitening:
            samples = self.pcaunwhitening(samples)
        return samples
    
    def forward(self, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader=None) -> None:
        """
        The forward pass of the AAE model.
        
        Parameters
        ----------
        train_data : torch.utils.data.DataLoader
            A dataloader for the training dataset
        val_data : torch.utils.data.DataLoader
            A dataloader for the validation dataset
            
        Returns
        -------
        None
        """
        for i in range(self.params['nepoch']):
            self._train_one_epoch(i, train_data, val_data)
   
    def _validate_after_one_epoch(self, val_data: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """
        A private method to run on validation set after  one epoch.
        
        Parameters
        ----------
        val_data : 
            The validation set to be run at this step
            
        Returns
        -------
        loss_g_dict : dict
            A dictionary to store each component of the generator loss
        loss_d : torch.Tensor
            The discriminator loss
        """
        self.generator.eval()
        self.discriminator.eval()
        val_z_real = torch.ones((len(val_data), 1)).to(self.params['device'])
        val_z_fake = torch.zeros((len(val_data), 1)).to(self.params['device'])
        
        x_batch = val_data.to(self.params['device'])
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
        loss_g = 0.99 * loss_g_dict['loss'] + 0.01 * self.criterion_D(self.discriminator(z), val_z_real)

        # Train the discriminator
        noise = self.true_prior.sample(sample_shape=torch.tensor([len(z),
                                       self.params['latent']])).to(self.params['device'])
        target_real, target_generator, loss_d = self.train_discriminator(z, noise, z_real=val_z_real, z_fake=val_z_fake)
        return loss_g_dict, loss_d
 
    def _train_one_epoch(self, i: int, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader=None) -> None:
        """
        A private method to train one epoch of all training data.
        
        Parameters
        ----------
        i : int
            The number of the current epoch
        train_data : torch.utils.data.DataLoader
            A dataloader for the training dataset
        val_data : torch.utils.data.DataLoader, default=None
            A dataloader for the validation dataset
            
        Returns
        -------
        None
        """
        self.generator.train()
        self.discriminator.train()
        # Enumerate each minibatch training data
        for j, train_batch in enumerate(train_data):
            loss_g_dict, loss_d = self._train_one_step(j, train_batch)
        
        # The component of the generator loss
        keys = list(loss_g_dict.keys())
            
        # Record loss after each epoch
        if i == 0:
            for k in keys:
                self.loss_g_train[k] = [loss_g_dict[k].item()]
        else:
            for k in keys:
                self.loss_g_train[k].append(loss_g_dict[k].item())
                
        self.loss_d_train.append(loss_d.item())
        
        # Report loss
        if self.verbose:
            report = 'TRAIN LOSS - Generator '
            for k in keys:
                report += '{0:s}: {1:f} '.format(k, loss_g_dict[k].item())
            report = report[:-1] + '; Discriminator loss {0:f}'.format(loss_d)
            print(report[:-1])
  
        # If val data is provided
        if val_data is not None:
            loss_g_dict, loss_d = self._validate_after_one_epoch(val_data)
            # Record loss after each epoch
            if i == 0:
                for k in keys:
                    self.loss_g_val[k] = [loss_g_dict[k].item()]
            else:
                for k in keys:
                    self.loss_g_val[k].append(loss_g_dict[k].item())
            self.loss_d_val.append(loss_d.item())
            if self.verbose:
                report = 'VAL LOSS - Generator '
                for k in keys:
                    report += '{0:s}: {1:f} '.format(k, loss_g_dict[k].item())
                report = report[:-1] + '; Discriminator loss {0:f}'.format(loss_d)
                print(report[:-1])
        
        # Update scheduler after each epoch
        if self.scheduler_G is not None:
            self.scheduler_G.step()
        if self.scheduler_D is not None:
            self.scheduler_D.step()
 
    def _train_one_step(self, j: int, train_batch) -> tuple[dict, torch.Tensor]:
        """
        A private method to train one step in an epoch.
        
        Parameters
        ----------
        j : int
            The number of training step in an epoch
        train_batch : 
            The minibatch to be trained at this step
            
        Returns
        -------
        loss_g_dict : dict
            A dictionary to store each component of the generator loss
        loss_d : torch.Tensor
            The discriminator loss
        """
        # In case we feed a labeled dataset into the model, we only select the training data but not the labels
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
        loss_g = 0.99 * loss_g_dict['loss'] + 0.01 * self.criterion_D(self.discriminator(z), self.z_real)
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

class SupervisedAAE(AAE):
    """
    Define a supervised adversarial autoencoder. 
    Label information is incorporated with latent space codes for the discriminator.

    This class is a subclass of the AAE class and the parameters for the constructor are the same.
    """
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module, 
                 optim_g, optim_d, generator_loss, discriminator_loss, true_prior, 
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(SupervisedAAE, self).__init__(params, generator, discriminator, 
                                            optim_g, optim_d, generator_loss, discriminator_loss, true_prior, 
                                            PCAWhitening, PCAUnWhitening, whitened_loss,
                                            variational, whitening, verbose)
    
    def _label_to_one_hot_vector(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Turn the label tensor to tensor of one-hot vector.
        
        Parameters
        ----------
        labels : torch.Tensor
            The label tensor. The elements should be within [0,nclass-1]
            
        Returns
        -------
        labels_one_hot : torch.Tensor
            The tensor of one-hot vector with the shape of (batch_size, nclass)
        """
        labels_one_hot = torch.zeros((labels.shape[0], self.params['nclass']), dtype=torch.int32)
        for k, label in enumerate(labels):
            label = label.int()
            labels_one_hot[k, label] = 1
        return labels_one_hot
               
    def _validate_after_one_epoch(self, val_data: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """
        A private method to run on validation set after  one epoch.
        
        Parameters
        ----------
        val_data : 
            The validation set to be run at this step
            
        Returns
        -------
        loss_g_dict : dict
            A dictionary to store each component of the generator loss
        loss_d : torch.Tensor
            The discriminator loss
        """
        self.generator.eval()
        self.discriminator.eval()
        val_z_real = torch.ones((len(val_data[0]), 1)).to(self.params['device'])
        val_z_fake = torch.zeros((len(val_data[0]), 1)).to(self.params['device'])

        x_batch = val_data[0].to(self.params['device'])
        label_batch = val_data[1].to(self.params['device'])
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
        loss_g = 0.99 * loss_g_dict['loss'] + 0.01 * self.criterion_D(self.discriminator(z_label_batch), val_z_real)
        # Train the discriminator
        noise, noise_one_hot_label = self.true_prior.sample(sample_shape=torch.tensor([len(z),
                                                            self.params['latent']]),
                                                            return_components=True)
        noise_label_batch = torch.cat((noise, noise_one_hot_label), dim=1)
        target_real, target_generator, loss_d = self.train_discriminator(z_label_batch, noise_label_batch, z_real=val_z_real, z_fake=val_z_real)
        return loss_g_dict, loss_d
 
    def _train_one_step(self, j: int, train_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A private method to train one step in an epoch.
        Overriden from the parent class. Parameters and returns are the same as the parent class.
        
        Parameters
        ----------
        j : int
            The number of training step in an epoch
        train_batch : list
            The minibatch to be trained at this step and the corresponding labels
            
        Returns
        -------
        loss_g_dict : dict
            A dictionary to store each component of the generator loss
        loss_d : torch.Tensor
            The discriminator loss
        """
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

class EnergyGapSAAE(SupervisedAAE):
    """
    A supervised aae model in which the decoder predicts the energy gap associated with each input data point.
    
    This class is a subclass of the SupervisedAAE class and the parameters for the constructor are the same.
    
    The PCAUnWhitening should be always ``None`` as we don't need to perform any whitening on the energy gap data.
    """
    def __init__(self, params: dict, generator: nn.Module, discriminator: nn.Module,
                 optim_g, optim_d, generator_loss, discriminator_loss, true_prior,
                 PCAWhitening: nn.Module=None, PCAUnWhitening: nn.Module=None, whitened_loss=True,
                 variational: bool=False, whitening: bool=True, verbose: bool=True) -> None:
        super(EnergyGapSAAE, self).__init__(params, generator, discriminator,
                                            optim_g, optim_d, generator_loss, discriminator_loss, true_prior,
                                            PCAWhitening, PCAUnWhitening, whitened_loss,
                                            variational, whitening, verbose)

    def _validate_after_one_epoch(self, val_data: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """
        A private method to run on validation set after  one epoch.
        
        Parameters
        ----------
        val_data : 
            The validation set to be run at this step
            
        Returns
        -------
        loss_g_dict : dict
            A dictionary to store each component of the generator loss
        loss_d : torch.Tensor
            The discriminator loss
        """
        self.generator.eval()
        self.discriminator.eval()
        val_z_real = torch.ones((len(val_data[0]), 1)).to(self.params['device'])
        val_z_fake = torch.zeros((len(val_data[0]), 1)).to(self.params['device'])

        x_batch = val_data[0].to(self.params['device'])
        energy_batch = val_data[1].to(self.params['device'])
        label_batch = val_data[2].to(self.params['device'])
        labels_one_hot = self._label_to_one_hot_vector(label_batch).to(self.params['device'])
        if self.whitening:
            x = self.pcawhitening(x_batch)
            assert x.shape[0] == x_batch.shape[0]
            assert x.shape[1] == x_batch.shape[1] - self.params['dof']
        else:
            x = x_batch
            assert x.shape == x_batch.shape

        # Train the generator
        energy_hat, generator_args = self.train_generator(x)
        z = generator_args[0]
        z_label_batch = torch.cat((z, labels_one_hot), dim=1)

        # Compute generator loss
        # Here the decoder outputs the energy gap so we compare the decoder learned
        # energy gap with the real energy gap
        loss_g_dict = self.criterion_G(energy_batch, energy_hat, generator_args[1:], self.params['beta'])
        loss_g = 0.99 * loss_g_dict['loss'] + 0.01 * self.criterion_D(self.discriminator(z_label_batch), val_z_real)
        # Train the discriminator
        noise, noise_one_hot_label = self.true_prior.sample(sample_shape=torch.tensor([len(z),
                                                            self.params['latent']]),
                                                            return_components=True)
        noise_label_batch = torch.cat((noise, noise_one_hot_label), dim=1)
        target_real, target_generator, loss_d = self.train_discriminator(z_label_batch, noise_label_batch, z_real=val_z_real, z_fake=val_z_fake)
        return loss_g_dict, loss_d

    def _train_one_step(self, j: int, train_batch) -> tuple[dict, torch.Tensor]:
        """
        A private method to train one step in an epoch.
        Overriden from the parent class. Parameters and returns are the same as the parent class.
        
        Parameters
        ----------
        j : int
            The number of training step in an epoch
        train_batch : list
            The minibatch to be trained at this step and the corresponding labels
            
        Returns
        -------
        loss_g_dict : dict
            A dictionary to store each component of the generator loss
        loss_d : torch.Tensor
            The discriminator loss
        """
        x_batch = train_batch[0].to(self.params['device'])
        energy_batch = train_batch[1].to(self.params['device'])
        label_batch = train_batch[2].to(self.params['device'])
        labels_one_hot = self._label_to_one_hot_vector(label_batch).to(self.params['device'])
        if self.whitening:
            x = self.pcawhitening(x_batch)
            assert x.shape[0] == x_batch.shape[0]
            assert x.shape[1] == x_batch.shape[1] - self.params['dof']
        else:
            x = x_batch
            assert x.shape == x_batch.shape

        # Train the generator
        energy_hat, generator_args = self.train_generator(x)
        z = generator_args[0]
        z_label_batch = torch.cat((z, labels_one_hot), dim=1)

        # Compute generator loss
        # Here the decoder outputs the energy gap so we compare the decoder learned
        # energy gap with the real energy gap
        loss_g_dict = self.criterion_G(energy_batch, energy_hat, generator_args[1:], self.params['beta'])
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

