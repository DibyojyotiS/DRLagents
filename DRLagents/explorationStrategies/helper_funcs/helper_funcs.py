import torch

def entropy(log_probs:torch.Tensor):
    """ compute the entropy given log probablities
    log_probs: Tensor of shape (N, m)
    """
    probs = torch.exp(log_probs)
    p_log_p = probs * log_probs
    entropy = -p_log_p.sum(-1, keepdim=True)
    return entropy 