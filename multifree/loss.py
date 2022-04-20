import torch
import torch.nn.functional as F
from torch import nn


class RMSDLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSDLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, x, x_hat):
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
    The implementation of the autoencoder loss function.
    """
    def __init__(self, recon_loss_fn: str='mse', kl: bool=True):
        super(AutoencoderLoss, self).__init__()
        self.kl = kl
        if recon_loss_fn == 'mse':
            self.recon_loss = nn.MSELoss() 
        elif recon_loss_fn == 'mae':
            self.recon_loss = nn.L1Loss()
        elif recon_loss_fn == 'bcewithlogits':
            self.recon_loss = nn.BCEWithLogitsLoss()
        elif recon_loss_fn == 'rmsd':
            self.recon_loss = RMSDLoss()
        elif recon_loss_fn == 'gaussian':
            self.recon_loss = torch.nn.GaussianNLLLoss()
        self.recon_loss_fn = recon_loss_fn
        
    def forward(self, x, x_hat, *args):
        if self.recon_loss_fn == 'gaussian':
            var = torch.ones_like(x_hat)
            recons_loss = self.recon_loss(x_hat, x, var)
        else:
            recons_loss = self.recon_loss(x_hat, x)
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
