from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .Strategy import Strategy
from .helper_funcs import entropy
from DRLagents.utils.helper_funcs import printDict


# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here
class greedyAction(Strategy):
    ''' selects an action greedly. The greedy strategy is 
    usually used in evaluation. '''

    def __init__(self, outputs_LogProbs=False,print_args=False) -> None:
        if print_args: printDict(self.__class__.__name__, locals())
        self.outputs_LogProbs = outputs_LogProbs


    def select_action(self, model: nn.Module, state: torch.Tensor, 
                        logProb_n_entropy=False, grad=False) \
                            -> Union[Tensor, 'tuple[Tensor, Tensor, Tensor]']:
        
        if not grad: # block gradients (do not store computation graph)
            with torch.no_grad():
                outputs = self._greedyActionUtil(model, state, logProb_n_entropy)
        else: # otherwise allow gradients
            outputs = self._greedyActionUtil(model, state, logProb_n_entropy)

        return outputs
    
    def _greedyActionUtil(self, model: nn.Module, state:torch.Tensor, logProb_n_entropy:bool):
            action_scores = model(state) # Q-values or action-log-probablities
            greedyAction = torch.argmax(action_scores, dim=-1, keepdim=True)

            if not logProb_n_entropy: return greedyAction

            # compute log-probs and entropy
            if not self.outputs_LogProbs:
                # convert Q-values to log probablitites
                log_probs = F.log_softmax(action_scores, dim=-1)
            else:
                log_probs = action_scores

            _entropy = entropy(log_probs)
            
            return greedyAction, log_probs.gather(-1, greedyAction), _entropy





