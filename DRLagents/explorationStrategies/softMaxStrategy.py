from typing import Union
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch import Tensor, nn

from .Strategy import Strategy
from ..utils import printDict


# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here
class softMaxAction(Strategy):

    def __init__(self, model: nn.Module, temperature=1, finaltemperature=None, 
                    decaySteps=None, outputs_LogProbs=False, print_args=False) -> None:
        ''' decays temperature by 1/e of initial-final in decaySteps if not None 

        temperature: the initial temperature for gumbel softmax

        finaltemperature: the asymptotically final value of temperature

        decaySteps: episodes in which temperaure decays to 1/e the way to final
                    If decaySteps=None, then temperature will not be decayed.

        outputs_LogProbs: whether the model returns log-probablities of actions

        NOTE: for outputs_LogProbs = True the action will be sampled from the 
                distribution returned by the model. 
                (corresponds to temperature=1, decaySteps=None)
        '''
        if print_args: printDict(self.__class__.__name__, locals())

        self.model = model
        self.temperature = temperature
        self.init_temperature = temperature
        self.final_temperature = finaltemperature
        self.decaySteps = decaySteps
        self.outputs_LogProbs = outputs_LogProbs

        self.episode = 0

        if outputs_LogProbs and decaySteps is not None:
            print('''Warning: for outputs_LogProbs = True the action will be sampled from the 
                distribution returned by the model without appying the 
                gumbel-softmax. (corresponds to temperature=1, decaySteps=None)''')


    def select_action(self, state: Tensor, 
                        logProb_n_entropy=False, grad=False) \
                            -> Union[Tensor, 'tuple[Tensor, Tensor, Tensor]']:
        if not grad: # block gradients (do not store computation graph)
            with torch.no_grad():
                outputs = self._softMaxActionUtil(state, logProb_n_entropy)
        else: # otherwise allow gradients
            outputs = self._softMaxActionUtil(state, logProb_n_entropy)   
        return outputs   


    def _softMaxActionUtil(self, state:Tensor, logProb_n_entropy:bool):

        # compute the action-probablities
        action_scores = self.model(state)
        if not self.outputs_LogProbs:
            probs = Categorical(F.softmax(action_scores/self.temperature, dim=-1))
        else:
            probs = Categorical(torch.exp(action_scores))
        
        softAction = probs.sample().view((-1,1))
        
        if not logProb_n_entropy: return softAction

        # compute entropy
        _entropy = probs.entropy().view((-1,1))
        log_prob = probs.log_prob(softAction).view((-1,1))

        return softAction, log_prob, _entropy #.view((*action_scores.shape[:-1],1))


    def decay(self):
        if self.decaySteps is None or self.final_temperature is None: return
        self.episode += 1
        self.temperature = self.final_temperature + \
                            (self.init_temperature-self.final_temperature)*np.exp(-1 * self.episode/self.decaySteps)