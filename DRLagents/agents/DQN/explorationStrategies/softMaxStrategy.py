from typing import Union
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch import Tensor, nn

from .Strategy import Strategy

# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here
class softMaxAction(Strategy):

    def __init__(self, temperature=1, finaltemperature=None, decaySteps=None) -> None:
        ''' decays temperature by 1/e of initial-final in decaySteps if not None 

        - temperature: float (default 1)
                - the initial temperature for gumbel softmax
        - finaltemperature: float|None (default None)
                - the asymptotically final value of temperature
                - If final_temperature=None, then temperature will not be decayed.
        - decaySteps: int|None (default None) 
                - episodes in which temperaure decays to 1/e the way to final
                - If decaySteps=None, then temperature will not be decayed.
        - print_args: bool (default False)
                - print the agruments passed in init
        '''
        self.temperature = temperature
        self.init_temperature = temperature
        self.final_temperature = finaltemperature
        self.decaySteps = decaySteps
        self.episode = 0

    def select_action(self, qvalues: Tensor) \
                            -> Union[Tensor, 'tuple[Tensor, Tensor, Tensor]']:
        probs = Categorical(F.softmax(qvalues.detach()/self.temperature, dim=-1))
        softAction = probs.sample().view((-1,1))
        return softAction

    def decay(self):
        if self.decaySteps is None or self.final_temperature is None: return
        self.episode += 1
        self.temperature = self.final_temperature + \
                            (self.init_temperature-self.final_temperature)*np.exp(-1 * self.episode/self.decaySteps)