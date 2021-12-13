import torch
from torch import nn

# base class for strategy
class Strategy:
    def __init__(self, model:nn.Module, **kwargs) -> None:
        # define required params
        pass

    def select_action(self, state:torch.tensor):
        """select an action using model and state
        according to the strategy you want. This method
        should be implemented in every derived class"""
        raise NotImplementedError()

    def decay(self):
        """ called by the traning function after the end of 
        each episode. use this to update any paramters that
        should be updated once every episode """
        # decay params if applicable
        pass
