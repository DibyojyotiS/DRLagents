from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from explorationStrategies import Strategy
from explorationStrategies.helper_funcs import entropy

# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here

class epsilonGreedyAction(Strategy):
    '''decays epsilon to 1/e of initial - final in decaySteps.
        doesnot decay epsilon if decaySteps is None'''

    def __init__(self, model: nn.Module, epsilon=0.5, finalepsilon=None, 
                    decaySteps=None, outputs_LogProbs=False) -> None:

        self.model = model
        self.epsilon = epsilon
        self.initepsilon = epsilon
        self.finalepsilon = finalepsilon
        self.decaySteps = decaySteps
        self.outputs_LogProbs = outputs_LogProbs

        # required inits
        self.episode = 0
        self.output_shape = None # will be updated lazily
        self.device = None # will be updated lazily


    def select_action(self, state:Tensor, 
                        logProb_n_entropy=False, grad=False) \
                            -> Union[Tensor, 'tuple[Tensor, Tensor, Tensor]']:
        
        if self.device is None:
            self._lazy_init_details(self.model, state)
        
        if not grad: # block gradients (do not store computation graph)
            with torch.no_grad():
                outputs = self._epsilonGreedyActionUtil(state, logProb_n_entropy)
        else: # otherwise allow gradients
            outputs = self._epsilonGreedyActionUtil(state, logProb_n_entropy)   

        return outputs     


    def _epsilonGreedyActionUtil(self, state:Tensor, logProb_n_entropy:bool):

        sample = torch.rand(1)
        action_scores = None # dummy init action_scores
        if sample < self.epsilon:
                eGreedyAction = torch.randint(self.output_shape[-1], 
                                                size = (*self.output_shape[:-1],1), 
                                                device=self.device)
        else:
            action_scores = self.model(state) # Q-values or action-log-probablities
            eGreedyAction = torch.argmax(action_scores, dim=-1, keepdim=True)

        # if entropy and log-probablities are not required 
        if not logProb_n_entropy: return eGreedyAction

        # if model outputs action-Q-values, convert them to log-probs
        if not self.outputs_LogProbs:
            log_probs = F.log_softmax(action_scores, dim=-1)
        else:
            log_probs = action_scores

        # compute the entropy and return log-prob of selected action
        _entropy = entropy(log_probs)

        return eGreedyAction, log_probs.gather(-1, eGreedyAction), _entropy


    def decay(self):
        if self.decaySteps is None or self.finalepsilon is None: return
        self.episode += 1
        self.epsilon = self.finalepsilon + (self.initepsilon-self.finalepsilon)*np.exp(-1 * self.episode/self.decaySteps)


    def _lazy_init_details(self, model:nn.Module, state:Tensor):
        ''' to lazily init the output shape, device of the model '''
        predictions = model(state)
        self.output_shape = predictions.shape
        self.device = predictions.device



    # def select_action(self, state:Tensor):
    #     ''' a single state not a batch! '''
    #     with torch.no_grad():
    #         Qpred = self.model(state)
    #         sample = torch.rand(1)
    #         if sample < self.epsilon:
    #             eGreedyAction = torch.randint(Qpred.shape[-1], (*Qpred.shape[:-1],1), device=Qpred.device)
    #         else:
    #             eGreedyAction = torch.argmax(Qpred, keepdim=True)
    #     return eGreedyAction
