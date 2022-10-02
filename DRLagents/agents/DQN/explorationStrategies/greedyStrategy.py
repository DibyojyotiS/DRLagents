import torch
import torch.nn.functional as F
from torch import Tensor
from .Strategy import Strategy


# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here
class greedyAction(Strategy):
    ''' selects an action greedly. The greedy strategy is 
    usually used in evaluation. '''

    def __init__(self) -> None:
        pass

    def select_action(self, qvalues:Tensor) -> Tensor:
        greedyAction = torch.argmax(qvalues.detach(), dim=-1, keepdim=True)
        return greedyAction