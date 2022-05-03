from typing import Union

import numpy as np

import torch
import torch.distributions as D
from torch import nn
from torch.utils import data


__all__ = [
    "GaussianMixtureBase", "GaussianMixture1D", "GaussianMixture2D", 
    "histogram_1d", "histogram_2d", "create_gaussian_mixtures_2d_prior"
]


def histogram_1d(data, nbins=50, density=True):
    hist, xedges = np.histogram(data, bins=nbins, density=density)
    x = (xedges[:-1] + xedges[1:]) / 2
    return hist.squeeze(), x.squeeze()

def histogram_2d(data_x, data_y, nbins=50, density=True):
    hist, xedges, yedges = np.histogram2d(data_x, data_y, bins=nbins, density=density)
    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2
    return hist, x, y

def create_gaussian_mixtures_2d_prior(n, weights=None):
    locations = torch.zeros((n,2))
    std = torch.zeros((n,2,2))
    
    for l in range(n):
        mean = torch.tensor([(n+0.1)*np.cos((l*2*np.pi)/n), (n+0.1)*np.sin((l*2*np.pi)/n)])
        locations[l,:] = mean
        v1 = [np.cos((l*2*np.pi)/n), np.sin((l*2*np.pi)/n)]
        v2 = [-np.sin((l*2*np.pi)/n), np.cos((l*2*np.pi)/n)]
        a1 = 0.95
        a2 = 1
        M =np.vstack((v1,v2)).T
        S = np.array([[a1, 0], [0, a2]])
        cov = torch.tensor(np.dot(np.dot(M, S), np.linalg.inv(M)))
        std[l,:,:] = cov
    p = GaussianMixture2D(locations, std, weights=weights)
    return p

class GaussianMixtureBase(D.Distribution):
    def __init__(self, means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor=None, device: str='cuda') -> None:
        self.ncomponents = int(means.shape[0])
        self.device = device
        if weights is None:
            self.weights = torch.ones((self.ncomponents,)) / self.ncomponents
        else:
            assert weights.shape[0] == self.ncomponents
            assert weights.sum() == 1.0
            self.weights = weights
            
        self.components = self._prepare_components(means, std)
    def _prepare_components(self, means, std):
        raise NotImplementedError
       
    @property
    def get_weights(self) -> torch.Tensor:
        return self.weights
  
    @property
    def get_num_component(self) -> int:
        return self.ncomponents
 
    def set_weights(self, value: torch.Tensor) -> torch.Tensor:
        """
        Update the weights of each component.
        """
        self.weights = value
        
    def sample(self, sample_shape: torch.Tensor=None, return_components: bool=False) -> torch.Tensor:
        return self.rsample(sample_shape=sample_shape, return_components=return_components)
    
    def rsample(self, sample_shape: torch.Tensor=None, 
                return_components: bool=False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from the mixture distribution.
        """
        if sample_shape is None:
            samples = torch.zeros((1,self.ndim))
        else:
            samples = torch.zeros((sample_shape[0], self.ndim))
        samples = samples.squeeze()
        sampled_components = list(data.WeightedRandomSampler(self.weights, samples.shape[0]))
        idx, counts = np.unique(sampled_components, return_counts=True)
        for i, index in enumerate(idx):
            component_position = np.argwhere(sampled_components == index).squeeze()
            current = self.components[index].rsample(sample_shape=[counts[i],1]).squeeze().float()
            samples[component_position] = current
        if return_components:
            label_one_hot = torch.zeros((samples.shape[0], self.ncomponents), dtype=torch.int32)
            for k, label in enumerate(sampled_components):
                label_one_hot[k, label] = label
            return samples.to(self.device), label_one_hot.to(self.device)
        else:
            return samples.to(self.device)
    
    def rsample_component(self, component: int=-1, sample_shape: torch.Tensor=None) -> torch.Tensor:
        """
        Sample from the one component of the mixture distribution.
        """
        if component < 0:
            samples = self.rsample(sample_shape=sample_shape)
        elif component >= self.ncomponents:
            raise ValueError('Incorrect component index!')
        else:
            d = self.components[component]
            samples = d.rsample(sample_shape=sample_shape)
        return samples.squeeze()
    
class GaussianMixture1D(GaussianMixtureBase):
    def __init__(self, means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor=None) -> None:
        super(GaussianMixture1D, self).__init__(means, std, weights)
        self.ndim = 1
        
    def _prepare_components(self, means, std):
        components = []
        for i in range(self.ncomponents):
            d = D.Normal(loc=means[i], scale=std[i])
            components.append(d)
        return components
    
class GaussianMixture2D(GaussianMixtureBase):
    def __init__(self, means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor=None) -> None:
        super(GaussianMixture2D, self).__init__(means, std, weights)
        self.ndim = len(means[0])
    
    def _prepare_components(self, means, std):
        components = []
        for i in range(self.ncomponents):
            d = D.MultivariateNormal(loc=means[i], covariance_matrix=std[i])
            components.append(d)
        return components

class PotentialModelBase(nn.Module):
    '''
    The base class for all potential energy models.
    '''

    def potential_energy(self, x):
        log_p = self.log_prob(x)
        u = self.f - log_p
        return u

    def partition_function(self):
        return torch.exp(-self.f)

class DoubleWellPotential(D.MixtureSameFamily, PotentialModelBase):
    '''
    Implementation of a double-well potential model as a mixture of two bivariate normal distributions.
    '''

    def __init__(self, mean, std, f=3, weight=torch.tensor([0.5, 0.5])):
        mix = D.Categorical(weight)
        comp = D.MultivariateNormal(mean, covariance_matrix=std)
        super().__init__(mix, comp)
        self.f = f  # free energy

    def sample(self, shape):
        samples = super().sample(shape)
        return samples.float()

    def log_prob(self, x):
        return super().log_prob(x)

    def density(self, x):
        log_p = self.log_prob(x)
        p = torch.exp(log_p)
        return p
