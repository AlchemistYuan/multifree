import torch
from torch import nn
from torch.utils import data
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


__all__ = [
   "PCAWhitening", "kmeans_scikit", "pca", "minmaxscaler", "xyz2dihedrals" 
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
    This function performs principal component analysis using scikit-learn.
    
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

def xyz2dihedrals(xyz: torch.Tensor, idx: list=None) -> torch.Tensor:
    """
    Compute the phi/psi dihedral angles from the cartesian coordinates.
    
    Parameters
    ----------
    xyz : torch.Tensor
        The cartesian coordinates
    idx : list, default=None
        The atom indices for the phi and psi angles
    
    Returns
    -------
    sincos_phipsi : torch.Tensor
        The phi/psi angles in sin/cos form
    """
    xyz_reshaped = xyz.reshape(xyz.shape[0], xyz.shape[1]//3, 3)
    phi_atoms = xyz_reshaped[:,idx[0],:]
    psi_atoms = xyz_reshaped[:,idx[1],:]

    # Get the phi/psi angles in radians
    phi_angles = _compute_dihedral(phi_atoms[:,0,:], phi_atoms[:,1,:], phi_atoms[:,2,:], phi_atoms[:,3,:])
    psi_angles = _compute_dihedral(psi_atoms[:,0,:], psi_atoms[:,1,:], psi_atoms[:,2,:], psi_atoms[:,3,:])
    sincos_phipsi = torch.zeros((phi_angles.shape[0], 4), device=self.params['device'])
    sincos_phipsi[:,0] = torch.sin(phi_angles)
    sincos_phipsi[:,1] = torch.cos(phi_angles)
    sincos_phipsi[:,2] = torch.sin(psi_angles)
    sincos_phipsi[:,3] = torch.cos(psi_angles)
    return sincos_phipsi

def _compute_angle(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    This method is based on the code in the following link:
    https://stackoverflow.com/questions/56918164/use-tensorflow-pytorch-to-speed-up-minimisation-of-a-custom-function
    
    Parameters
    ----------
    p1, p2 : torch.Tensor
        Two vectors which represent two line
    
    Returns
    -------
    angle : torch.Tensor
        The angle formed by the two lines
    """
    inner_product = (p1*p2).sum(dim=-1)
    p1_norm = torch.linalg.norm(p1, dim=-1)
    p2_norm = torch.linalg.norm(p2, dim=-1)
    cos = inner_product / (p1_norm * p2_norm)
    cos = torch.clamp(cos, -0.99999, 0.99999)
    angle = torch.acos(cos)
    return angle

def _compute_dihedral(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, v4: torch.Tensor) -> torch.Tensor:
    """
    This method is based on the code in the following link:
    https://stackoverflow.com/questions/56918164/use-tensorflow-pytorch-to-speed-up-minimisation-of-a-custom-function
    
    Parameters
    ----------
    v1, v2, v3, v4 : torch.Tensor
        cartesian coordinates of four atoms

    Returns
    -------
    angle : torch.Tensor
        The dihedral angle defined by the four atoms
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
