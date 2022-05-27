import torch
from torch import nn
from torch.utils import data
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


__all__ = [
   "PCAWhitening", "kmeans_scikit", "pca", "minmaxscaler" 
]


def kmeans_scikit(data: np.array, k: int=5, batch_size: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    This function performs kmeans clustering using sci-kit learn. 

    Parameters
    ----------
    data : list
        The array of input features of shape (nframe, nfeatures)
    k : int, default=5
        The number of the clusters
    batch_size : int, default=0
        Batch size for MiniBatchKmeans. Default is 0, which means not to use MiniBatchKMeans.
        
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    labels : ndarray of shape (n_samples,)
        The assignment of the each frame to each cluster.
    cluster_center_indices : ndarray of shape (n_clusters,)
        The frame indices cloesest to each cluster
    '''
    if batch_size > 0:
        kmeans = MiniBatchKMeans(init="k-means++", n_clusters=k, batch_size=batch_size)
    else:
        kmeans = KMeans(init="k-means++", n_clusters=k)
    distances = kmeans.fit_transform(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels = np.asarray(labels, dtype=int)
    cluster_center_indices = np.argmin(distances, axis=0)
    cluster_center_indices = np.asarray(cluster_center_indices, dtype=int)
    return centers, labels, cluster_center_indices

def pca(x, keepdims: int=6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function implements principal component analysis in scikit-learn.
    
    Parameters
    ----------
    x : torch.Tensor
        The entire training dataset with the shape of (nsamples, nfeatures)
    keepdims : int, default=6
        The last a few columns to be discarded, which correspond to 
        the global translational and rotational degrees of freedom.
        
    Returns
    -------
    x_mean : torch.Tensor
        The mean of the training dataset with shape of (nfeatures,)
    pc_matrix : torch.Tensor
        The pca eigenvecgtors matrix with the shape of (nfeatures, nfeatures-keepdims)
    variance_sqrt : torch.Tensor
        The square root of the pca eigenvalues with the shape of (nfeatures-keepdims,)
    featrange : torch.Tensor
        The range of the input features
    """
    x_mean = np.mean(x, axis=0)
    x_meanfree = (x - x_mean) 
    pca_runner = PCA(keepdims)
    x_rotated = pca_runner.fit_transform(x_meanfree)
    eigval = pca_runner.explained_variance_
    eigvec = pca_runner.components_
    # sort eigen values in the descending order
    idx_descending = np.argsort(eigval)[::-1]#, descending=True)
    eigval = eigval[idx_descending]
    eigvec = eigvec[idx_descending,:]
    assert eigval.min() > 0.0
    variance_sqrt = np.sqrt(eigval.flatten())
    x_rotated_scaled = x_rotated / variance_sqrt 
    featmin = torch.tensor(np.min(x_rotated_scaled, axis=0))
    featmax = torch.tensor(np.max(x_rotated_scaled, axis=0))
    featrange = featmax - featmin
    return torch.tensor(x_mean), torch.tensor(eigvec), torch.tensor(variance_sqrt), featrange, featmin

def minmaxscaler(features, featrange, featmin, backscale=False) -> torch.Tensor:
    """
    Scale the range of features to [-1,1]
    scaled_features = 2 * (features - featmin) / featrange - 1
    
    Parameters
    ----------
    features : torch.Tensor
        The features to be scaled
    featrange : torch.Tensor
        The range of the input features
    featmin : torch.Tensor
        The minimum values of each feature 
    backscale : bool, default=False
        If it's true, the input features have already been scaled and will be backscaled to the original range
        
    Returns
    -------
    scaled_features : torch.Tensor
        The scaled features
    """
    if backscale:
        scaled_features = featrange * ((features + 1) / 2) + featmin
    else:
        scaled_features = (2 * (features - featmin) / featrange) - 1
    return scaled_features

class PCAWhitening(nn.Module):
    """
    Implementation of the PCA whitening and unwhitening process.
    """
    def __init__(self, dataset: data.Dataset, dof: int=6, inverse: bool=False, minmax: bool=True) -> None:
        super(PCAWhitening, self).__init__()
        if dof is None:
            keepdims = dataset.shape[1]
        else:
            keepdims = dataset.shape[1] - dof
        x_mean, eigvec, variance_sqrt, featrange, featmin = pca(dataset, keepdims=keepdims)
        self.register_buffer("x_mean", x_mean.float().requires_grad_(False))
        self.register_buffer("variance_sqrt", variance_sqrt.float().requires_grad_(False))
        self.register_buffer("eigvec", eigvec.float().requires_grad_(False))
        self.register_buffer("featrange", featrange.float().requires_grad_(False))
        self.register_buffer("featmin", featmin.float().requires_grad_(False))
        self.inverse = inverse
        self.minmax = minmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inverse:
            if self.minmax:
                x_backscaled = minmaxscaler(x, self.featrange, self.featmin,
                                            backscale=True)
            else:
                x_backscaled = x
            x_rotated = x_backscaled * self.variance_sqrt
            x_transformed = torch.matmul(x_rotated, self.eigvec) + self.x_mean
        else:
            x_rotated = torch.matmul((x-self.x_mean), self.eigvec.T) 
            x_transformed_unscaled = x_rotated / self.variance_sqrt
            if self.minmax:
                # Convert the range of features to [-1,1]
                x_transformed = minmaxscaler(x_transformed_unscaled,
                                             self.featrange, self.featmin) 
            else:
                x_transformed = x_transformed_unscaled
        return x_transformed
