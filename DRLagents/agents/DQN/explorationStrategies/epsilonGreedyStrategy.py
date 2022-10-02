from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .Strategy import Strategy
from DRLagents.utils.helper_funcs import printDict

# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here

class epsilonGreedyAction(Strategy):
    '''decays epsilon from initial to final in decaySteps.
        doesnot decay epsilon if decaySteps is None'''

    def __init__(self, epsilon=0.5, finalepsilon=None, 
                    decaySteps=None, decay_type='lin',
                    print_args=False) -> None:
        """ 
        ### parameters
        - epsilon: float (default 0.5) must be in [0,1)
        - finalepsilon: float (default None)
                - must be greater than 0
                - epsilon not decayed if finalepsilon is None
        - decaySteps: int (default None)
                - #calls to decay(...) in which epsilon decays to finalepsilon
                - if None, epsilon is not decayed
        - decay_type: str (default 'lin')
                - if 'lin' then epsilon decayed linearly
                - if 'exp' then epsilon decayer exponentialy
        - print_args: bool (default False)
                - print the agruments passed in init
         """

        if print_args: printDict(self.__class__.__name__, locals())
        assert decay_type in ['lin', 'exp']

        self.epsilon = epsilon
        self.initepsilon = epsilon
        self.finalepsilon = finalepsilon
        self.decaySteps = decaySteps
        self.decay_type = decay_type

        # required inits
        self.episode = 0
        self.output_shape = None # will be updated lazily
        self.device = None # will be updated lazily

    def select_action(self, qvalues: Tensor) -> Tensor:
        if self.device is None:
            self._lazy_init_details(qvalues)

        sample = torch.rand(1)
        if sample < self.epsilon:
                eGreedyAction = torch.randint(self.output_shape[-1], 
                                                size = (*self.output_shape[:-1],1), 
                                                device=self.device)
        else:
            eGreedyAction = torch.argmax(qvalues.detach(), dim=-1, keepdim=True)
        
        return eGreedyAction

    def decay(self):
        if self.decaySteps is None or self.finalepsilon is None: return
        self.episode += 1
        if self.decay_type == 'exp':
            self.epsilon = self._exponential_decay(self.episode)
        else:
            self.epsilon = self._linear_decay(self.episode)

    def _lazy_init_details(self, qvalues:Tensor):
        ''' to lazily init the output shape, device of the model '''
        self.output_shape = qvalues.shape
        self.device = qvalues.device

    exp_decay_factor = None
    def _exponential_decay(self, episode):
        if self.exp_decay_factor is None:
            self.exp_decay_factor = (np.math.log(epsilon) - \
                np.math.log(self.finalepsilon))/self.decaySteps
        epsilon = self.initepsilon*np.exp(-self.exp_decay_factor*episode)
        return max(epsilon, self.finalepsilon)

    lin_decay_factor = None
    def _linear_decay(self,episode):
        if self.lin_decay_factor is None:
            self.lin_decay_factor = \
                (self.initepsilon - self.finalepsilon)/self.decaySteps
        epsilon = self.initepsilon - self.lin_decay_factor * episode
        return max(epsilon, self.finalepsilon)
