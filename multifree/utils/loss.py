import torch
import torch.nn.functional as F
from torch import nn


__all__ = [
    "RMSDLoss", "AutoencoderLoss", "XYZDihderalAAEGeneratorLoss"
]

class RMSDLoss(nn.Module):
    """
    Create a custom loss function in pytorch that measures The root mean square of deviation (RMSD) between the output and the target.

    Parameters
    ----------
    reduction : str, default=``'mean'``
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied; ``'mean'``: the sum of the output will be divided by the number of elements in the output.
    """
    def __init__(self, reduction: str='mean'):
        super(RMSDLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the loss function

        Parameters
        ----------
        x : torch.Tensor
            The input data
        x_hat : torch.Tensor
            The output data

        Returns
        -------
        loss : torch.Tensor
            The loss values
        """
        x_reshaped = x.reshape(x.shape[0], x.shape[1]//3, 3)
        x_hat_reshaped = x_hat.reshape(x_hat.shape[0], x_hat.shape[1]//3, 3)
        assert x_reshaped.shape == x_hat_reshaped.shape
        norm = torch.linalg.vector_norm(x_hat_reshaped - x_reshaped, dim=2)
        loss = torch.sqrt(torch.mean(norm** 2, dim=1))
        if self.reduction == 'mean':
             loss = torch.mean(loss)
        return loss

class AutoencoderLoss(nn.Module):
    """
    The common API for the autoencoder loss function.
    The loss functions include ``'mse'``, ``'mae'``, ``'bcewithlogits'``,
    ``'rmsd'``, and ``'gaussian'``.

    Parameters
    ----------
    loss_fn : str, default=``'mse'``
        The name for the loss function 
    kl : bool, default=``False``
        Whether a KL loss term will be added to the loss function.
        kl should be set to ``True`` if a variational autoencoder is used.
    """
    def __init__(self, loss_fn: str='mse', kl: bool=False):
        super(AutoencoderLoss, self).__init__()
        self.kl = kl
        if loss_fn == 'mse':
            self.loss_fn = nn.MSELoss() 
        elif loss_fn == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_fn == 'bcewithlogits':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_fn == 'rmsd':
            self.loss_fn = RMSDLoss()
        elif loss_fn == 'gaussian':
            self.loss_fn = torch.nn.GaussianNLLLoss()
        self.loss_fn_name = loss_fn
        
    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, 
                *args: torch.Tensor) -> dict:
        """
        The forward pass for the custom loss function.
 
        Parameters
        ----------
        x : torch.Tensor
            The input data
        x_hat : torch.Tensor
            The output data
        args : torch.Tensor
            The positional arguments. 
            These arguments should be provided if kl is ``True``.     
        
        Returns
        -------
        report : dict
            A dictionary containing each component of the loss function.
        """
        if self.loss_fn_name == 'gaussian':
            var = torch.ones_like(x_hat)
            recons_loss = self.loss_fn(x_hat, x, var)
        else:
            recons_loss = self.loss_fn(x_hat, x)
        if self.kl:
            mu = args[0]
            log_var = args[1]
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            beta = args[2]
            loss = recons_loss + beta * kld_loss
            report = {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}
        else:
            loss = recons_loss
            report = {'loss': loss, 'Reconstruction_Loss':recons_loss.detach()}
        return report
    
    
class XYZDihderalAAEGeneratorLoss(nn.Module):
    def __init__(self, split: int=4, recon_loss_cartesian: str='mse', recon_loss_dihedral: str='mse', 
                 kl: bool=False, device: str='cuda:0') -> torch.tensor:
        super(XYZDihderalAAEGeneratorLoss, self).__init__()
        self.cartensian_loss_fn = AutoencoderLoss(recon_loss_fn=recon_loss_cartesian, 
                                                  kl=kl).to(device)
        self.dihedral_loss_fn = AutoencoderLoss(recon_loss_fn=recon_loss_dihedral, 
                                                  kl=kl).to(device)
        self.split = split
        
    def forward(self, x: torch.tensor, x_hat: torch.tensor, *args) -> list:
        split = args[-1]
        other_args = args[:-1]

        loss_dict = self.cartensian_loss_fn(x[:,:split], x_hat[:,:split], *other_args)
        dihderal_loss = self.dihedral_loss_fn(x[:,split:], x_hat[:,split:], *other_args)
        loss_dict['loss'] = loss_dict['loss'] + dihderal_loss['loss']
        loss_dict['Dihdedral_loss'] = dihderal_loss['loss'].detach()
        return loss_dict
