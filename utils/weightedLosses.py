# weighted versions of losses commonly used by me

from typing import Optional
import torch
import torch.nn.functional as F


def _apply_weights(loss:torch.Tensor, weights:torch.Tensor):
    loss = loss.squeeze() * weights.squeeze()
    loss = loss.sum()/weights.sum()
    return loss


def weighted_MSEloss(input:torch.Tensor, target:torch.Tensor, weights:Optional[torch.Tensor]=None):
    """ a weighted version of mse_loss """
    loss = F.mse_loss(input=input, target=target, reduction='none')
    if weights is None:
        return loss.mean()
    return _apply_weights(loss, weights)


def weighted_HuberLoss(input:torch.Tensor, target:torch.Tensor, weights:Optional[torch.Tensor]=None):
    """ a weighted version of HuberLoss (a.k.a smooth L1 loss) """
    loss = F.smooth_l1_loss(input=input, target=target, reduction='none')
    if weights is None:
        return loss.mean()
    return _apply_weights(loss, weights)