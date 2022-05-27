from typing import Union

import numpy as np

import torch
import torch.distributions as D
from torch import nn
from torch.utils import data


__all__ = [
    "GaussianMixtureBase", "GaussianMixture1D", "GaussianMixture2D", "DoubleWellPotential",
    "histogram_1d", "histogram_2d", "create_gaussian_mixtures_2d_prior"
]


def histogram_1d(data: np.ndarray, nbins: int=50, density: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram for a one-dimensinoal array
    
    Parameters
    ----------
    data : np.ndarray
        The input data
    nbins : int, default=50
        The number of bins
    density : bool, default=True
        Whether to compute the probability density
        If it's False, the counts will be returned
    
    Returns
    -------
    hist : np.ndarray
        The computed 1D histogram
    x : np.ndarray
        The bin centers
    """
    hist, xedges = np.histogram(data, bins=nbins, density=density)
    x = (xedges[:-1] + xedges[1:]) / 2
    return hist.squeeze(), x.squeeze()

def histogram_2d(data_x: np.ndarray, data_y: np.ndarray, 
                 nbins: int=50, density: bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the histogram for a two-dimensinoal array
    
    Parameters
    ----------
    data_x : np.ndarray
        The input data along the first dimension
    data_y : np.ndarray
        The input data along the second dimension
    nbins : int, default=50
        The number of bins
    density : bool, default=True
        Whether to compute the probability density
        If it's False, the counts will be returned
    
    Returns
    -------
    hist : np.ndarray
        The computed 1D histogram
    x : np.ndarray
        The bin centers along the first dimension
    y : np.ndarray
        The bin centers along the second dimension   
    """
    hist, xedges, yedges = np.histogram2d(data_x, data_y, bins=nbins, density=density)
    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2
    return hist, x, y

def create_gaussian_mixtures_2d_prior(n: int, weights: list) -> D.Distribution:
    """
    Create a 2d mixture of gaussian distributions as the prior distribution.
    
    Parameters
    ----------
    n : int
        The number of component
    weights : torch.Tensor
        The weights of each component. The sum of each weight should be one.
        
    Returns
    -------
    p : D.Distribution
        The created mixture distributions
    """
    locations = torch.zeros((n,2))
    std = torch.zeros((n,2,2))
    
    for l in range(n):
        mean = 4.1 * torch.tensor([np.cos((l*2*np.pi)/n), np.sin(np.pi/4 + (l*2*np.pi)/n)])
        locations[l,:] = mean
        #v1 = [np.cos((l*2*np.pi)/n), np.sin((l*2*np.pi)/n)]
        #v2 = [-np.sin((l*2*np.pi)/n), np.cos((l*2*np.pi)/n)]
        #a1 = 0.95
        #a2 = 1
        #M =np.vstack((v1,v2)).T
        #S = np.array([[a1, 0], [0, a2]])
        #cov = torch.tensor(np.dot(np.dot(M, S), np.linalg.inv(M)))
        std[l,:,:] = torch.eye(2) * 5 #cov
    p = GaussianMixture2D(locations, std, weights=weights)
    return p

class GaussianMixtureBase(D.Distribution):
    """
    The base class for the mixture of gaussian distributions.
    
    Parameters
    ----------
    means : torch.Tensor
        The means of each component in the mixture distribution
    std : torch.Tensor
        The standard deviation of each component
    weights: torch.Tensor
        The weights of each component. The sum of each weight should be one.
    device : str, default='cuda'
        The selected device
    """
    def __init__(self, means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor, device: str='cuda') -> None:
        self.ncomponents = int(means.shape[0])
        self.device = device
        if weights is None:
            self.weights = torch.ones((self.ncomponents,)) / self.ncomponents
        else:
            assert weights.shape[0] == self.ncomponents
            assert weights.sum() == 1.0
            self.weights = weights
            
        self.components = self._prepare_components(means, std)
        
    def _prepare_components(self, means: torch.Tensor, std: torch.Tensor):
        """
        The private method to prepare the components.
    
        Parameters
        ----------
        means : torch.Tensor
            The means of each component in the mixture distribution
        std : torch.Tensor
            The standard deviation of each component
        """
        raise NotImplementedError
       
    @property
    def get_weights(self) -> torch.Tensor:
        """
        Return the current weights
        """
        return self.weights
  
    @property
    def get_num_component(self) -> int:
        """
        Return the number of components
        """
        return self.ncomponents
 
    def set_weights(self, value: torch.Tensor) -> torch.Tensor:
        """
        Update the weights of each component.
        """
        self.weights = value
        
    def sample(self, sample_shape: torch.Tensor=None, return_components: bool=False) -> torch.Tensor:
        """
        Sample from the mixture distribution.
        """
        return self.rsample(sample_shape=sample_shape, return_components=return_components)
    
    def rsample(self, sample_shape: torch.Tensor=None, 
                return_components: bool=False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from the mixture distribution.
        
        Parameters
        ----------
        sample_shape : torch.Tensor, default=None
            The shape of the samples
        return_components : bool, default=False
            Whether to return the component id associated with each sample. The default is False.
            
        Returns
        -------
        samples : torch.Tensor
            The samples generated from the mixture component
        label_one_hot : torch.Tensor, optional
            The one hot vector of the component id associated with each sample
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
        Sample from one component of the mixture distribution.
        
        Parameters
        ----------
        component : int, default=-1
            The component id to be sampled from
        sample_shape : torch.Tensor, default=None
            The shape of the samples
            
        Returns
        -------
        samples : torch.Tensor
            The samples drawn from the specified component
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
    """
    The mixture of one-dimensional gaussian distributions.
    This class is a subclass of the GaussianMixtureBase class and parameters are the same as GaussianMixtureBase.
    """
    def __init__(self, means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor=None) -> None:
        super(GaussianMixture1D, self).__init__(means, std, weights)
        
    def _prepare_components(self, means: torch.Tensor, std: torch.Tensor) -> list:
        components = []
        for i in range(self.ncomponents):
            d = D.Normal(loc=means[i], scale=std[i])
            components.append(d)
        return components
    
class GaussianMixture2D(GaussianMixtureBase):
    """
    The mixture of two-dimensional gaussian distributions.
    This class is a subclass of the GaussianMixtureBase class and parameters are the same as GaussianMixtureBase.
    """
    def __init__(self, means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor=None) -> None:
        super(GaussianMixture2D, self).__init__(means, std, weights)
    
    def _prepare_components(self, means: torch.Tensor, std: torch.Tensor) -> list:
        components = []
        for i in range(self.ncomponents):
            d = D.MultivariateNormal(loc=means[i], covariance_matrix=std[i])
            components.append(d)
        return components

class DoubleWellPotential(D.MixtureSameFamily):
    """
    Implementation of a double-well potential model as a mixture of two bivariate normal distributions.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, 
                 f: int=3, weight: torch.Tensor=torch.tensor([0.5, 0.5])):
        mix = D.Categorical(weight)
        comp = D.MultivariateNormal(mean, covariance_matrix=std)
        super().__init__(mix, comp)
        self.f = f  # free energy

    def density(self, x: torch.Tensor) -> torch.Tensor:
        log_p = self.log_prob(x)
        p = torch.exp(log_p)
        return p
    
    def partition_function(self) -> torch.Tensor:
        return torch.exp(-self.f)
