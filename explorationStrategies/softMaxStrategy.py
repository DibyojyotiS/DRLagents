import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from explorationStrategies import Strategy

# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here
class softMaxAction(Strategy):

    def __init__(self, model:nn.Module, temperature=1, finaltemperature=None, decaySteps=None) -> None:
        '''decays temperature to 1/e of initial - final in decaySteps if not None'''
        self.model = model
        self.temperature = temperature
        self.init_temperature = temperature
        self.final_temperature = finaltemperature
        self.decaySteps = decaySteps

        self.episode = 0


    def select_action(self, state:torch.tensor):

        with torch.no_grad():
            Probs = F.softmax(self.model(state)/self.temperature, dim=-1)
            softAction = torch.distributions.Categorical(Probs).sample()
        return softAction.view(-1,1)


    def decay(self):
        if self.decaySteps is None or self.final_temperature is None: return
        self.episode += 1
        self.temperature = self.final_temperature + \
                            (self.init_temperature-self.final_temperature)*np.exp(-1 * self.episode/self.decaySteps)
