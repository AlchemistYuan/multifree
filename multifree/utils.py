import argparse
from typing import Union

import torch
from torch.utils import data
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

from .distributions import *


def read_inputs() -> argparse.Namespace:
    '''
    This function reads all command line arguments and return the Namespace object.
    
    Parameters
    ----------
    None

    Returns
    -------
    args : argparse.Namespace
        The Namespace object to store all arguments
    '''
    # Read the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', dest='indir', help='Dataset directory', default='./')
    parser.add_argument('--outdir', dest='outdir', help='Output directory', default='./')
    parser.add_argument('--traindata', dest='traindata', help='Training dataset')
    parser.add_argument('--label', dest='label', help='Labels for training data')
    parser.add_argument('--activation', dest='activ', help='Activation function', default='relu')
    parser.add_argument('--latent', dest='latent', help='Latent space dimension', default=2, type=int)
    parser.add_argument('--infeatures', dest='infeatures', help='Encoder input feature dimension', type=int)
    parser.add_argument('--outfeatures', dest='outfeatures', help='Decoder output feature dimension', type=int)
    parser.add_argument('--hidden', dest='hidden', help='Dimensions of hidden layers', nargs='+', type=int)
    parser.add_argument('--nepoch', dest='nepoch', help='Number of epoch', default=200, type=int)
    parser.add_argument('--batchsize', dest='batchsize', help='Batch size', default=100, type=int)
    parser.add_argument('--dof', dest='dof', help='Translational and rotational degrees of freedom', default=6, type=int)
    parser.add_argument('--beta', dest='beta', help='Weight in the KL loss term in betaVAE', default=1.0, type=float)
    parser.add_argument('--lr', dest='lr', help='Learning rate', default=0.00025, type=float)
    parser.add_argument('--nclass', dest='nclass', help='Number of classes in (semi-)supervised task', default=4, type=int)
    parser.add_argument('--device', dest='device', help='CUDA or CPU device', default='cuda:0')
    parser.add_argument('--dischidden', dest='dischidden', help='Dimensions of the discriminator hidden layers', nargs='+', type=int)
    parser.add_argument('--disclatent', dest='disclatent', help='Dimension of the discriminator latent layer', type=int)
    parser.add_argument('--decoderfinal', dest='decoderfinal', help='Final layer of the decoder', default='linear')
    parser.add_argument('--idx', dest='idx', help='List of atom indices for dihedral angles', nargs='+', action='append', type=int)
    parser.add_argument('--split', dest='split', help='Number of features not to be whitened', default=4, type=int)
    parser.add_argument('--xyzreconloss', dest='xyzreconloss', help='Cartesian reconstruction loss', default='mse')
    parser.add_argument('--dihereconloss', dest='dihereconloss', help='Dihedral reconstruction loss', default='mse')
    parser.add_argument('--kl', dest='kl', help='Whether to use variational aotuencoder', action='store_true')
    parser.add_argument('--whitening', dest='whitening', help='Whether to include a whitening layer', action='store_true')
    parser.add_argument('--whitenedloss', dest='whitenedloss', help='Whether to calaulte loss on the whitened output', action='store_true')
    parser.add_argument('--lossfile', dest='lossfile', help='Save loss to a file', default='./loss.txt')
    parser.add_argument('--losspng', dest='losspng', help='Plot loss and save the figure', default='./loss.png')
    parser.add_argument('--samplepng', dest='samplepng', help='Plot the generated samples and save the figure', default='./samples.png')
    parser.add_argument('--paramslog', dest='paramslog', help='A json file to log the model parameters', default='./params.json')
    parser.add_argument('--modelfile', dest='modelfile', help='Save the trained model', default='./model.pt')
    args = parser.parse_args()
    return args

def default_params() -> dict:
    model_params = {'activation': 'relu','latent': 2, 'infeatures': None,
                    'outfeatures': None, 'hidden': [64,32], 'nepoch': 20,
                    'batchsize': 100, 'dof': 6, 'beta': 1.0, 'lr': 0.0001, 'nclass': 4,
                    'device': 'cuda:0', 'idx': None, 'split': 4,
                    'dischidden': [32,16], 'disclatent': 6, 
                    'decoderfinal': 'linear'}
    return model_params
 
def plot_aae_loss(loss_g, loss_d, output=None):
    fig, ax = plt.subplots(constrained_layout=True)
    ax2 = ax.twinx()
    lns = ax.plot(loss_g, label='Generator loss')
    lns += ax2.plot(loss_d, color='orange', label='Discriminator loss')
    ax.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Generator Loss', fontsize=16)
    ax2.set_ylabel('Discriminator Loss', fontsize=16)
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    if output is not None:
        fig.savefig(output, dpi=300)
    return fig, ax, ax2

def generate_grid(xmin=-1, xmax=1, ymin=-1, ymax=1, steps=1000):
    X = torch.linspace(xmin, xmax, steps)
    Y = torch.linspace(ymin, ymax, steps)
    grid_X, grid_Y = torch.meshgrid(X, Y)
    grid = torch.stack([grid_X, grid_Y], dim = -1)
    Z = grid.reshape((-1, 2))
    return Z    

def train_val_split(dataset: data.Dataset, val_splitting: float=0.1) -> list:
    """
    Take a dataset and split it into training and validation sets.
    """
    n = len(dataset)  # how many total elements you have
    n_val = int(n * val_splitting)  # number of test/val elements
    n_train = n - n_val
    train_set, val_set = data.random_split(dataset, (n_train, n_val))
    return [train_set, val_set]

def autoencoder2conformations(models: list, u: mda.Universe, atoms: str='all', z=None,
                              outname: str='output.pdb', whitening: bool=True, nsample: int=1000) -> None:
    """
    Save the generated conformations to a trajectory file.
    """
    if whitening:
        [pca_whitening, autoencoder, pca_unwhitening] = models
    else:
        [autoencoder] = models
    autoencoder.eval()
    with torch.no_grad():
        if z is not None:
            samples = autoencoder.sample(z)
        elif nsample is not None:
            samples = autoencoder.sample(nsample)
        elif (z is None) and (nsample is None):
            raise ValueError("Either z or nsample should be provided.")
        elif (z is not None) and (nsample is not None):
            raise ValueError("z and nsample cannot be provided simultaneously.")
            
        if whitening:
            x_rec = pca_unwhitening(samples).cpu().detach().numpy()
        else:
            x_rec = samples
        x_rec_reshaped = x_rec.reshape(nsample,-1,3)
        prot = u.select_atoms(atoms)
        with mda.Writer(outname, multiframe=True) as W:
            for i in range(x_rec_reshaped.shape[0]):
                frame = x_rec_reshaped[i,:,:]
                prot.atoms.positions = frame
                W.write(prot)
               
def generate_ssaae_samples_from_checkpoint(modelfile, paramfile, sample_shape=[1000,1], 
                                           component=None, device='cuda'):
    model, optimizer_G, optimizer_D = get_xyz_dihedral_ssaae(paramfile)
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])
    with open(paramfile) as f:
        model_params = json.load(f)
    n = model_params['nclass']
    prior = create_gaussian_mixtures_2d_prior(n)
    
    # Dont forget to switch to the evaluation mode
    model.eval()
    
    if component is not None:
        z = prior.rsample_component(component=i, sample_shape=sample_shape).squeeze().to(device)
    else:
        z = prior.sample(sample_shape=sample_shape).squeeze().to(device)
    samples = model.sample(z=z)
    return samples
 
class CustomDataset(data.Dataset):
    """
    Custom pytorch dataset.
    """
    def __init__(self, data: torch.Tensor, labels: torch.Tensor=None, transform=None) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform
        self.length = data.shape[0]

    def __getitem__(self, index: int) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        sample = self.data[index]
        if self.labels is not None:
            label = self.labels[index]
            return sample, label
        else:
            return sample

    def __len__(self) -> int:
        return self.length
