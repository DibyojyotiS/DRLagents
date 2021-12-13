import torch
from torch import nn

from explorationStrategies import Strategy

# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here
class greedyAction(Strategy):

    def __init__(self, model:nn.Module) -> None:
        self.model = model

    def select_action(self, state:torch.tensor):
        ''' preferablely a single state '''
        with torch.no_grad():
            Qpred = self.model(state)
            greedyAction = torch.argmax(Qpred, dim=-1, keepdim=True)
        return greedyAction

