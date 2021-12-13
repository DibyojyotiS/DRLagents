import numpy as np
import torch
from torch import nn

from explorationStrategies import Strategy

# we have the input model here so that the user can have a model of multiple submodels and then use only one submodel in
# training strategies like the DQN, or alternatively the same model that would be trained can be sent here

class epsilonGreedyAction(Strategy):

    def __init__(self, model, epsilon=0.5, finalepsilon=None, decaySteps=None) -> None:
        '''decays epsilon to 1/e of initial - final in decaySteps if not None'''
        self.model = model
        self.epsilon = epsilon
        self.initepsilon = epsilon
        self.finalepsilon = finalepsilon
        self.decaySteps = decaySteps

        self.episode = 0


    def select_action(self, state:torch.tensor):
        ''' a single state not a batch! '''
        with torch.no_grad():
            Qpred = self.model(state)
            sample = torch.rand(1)
            if sample < self.epsilon:
                eGreedyAction = torch.randint(Qpred.shape[-1], (*Qpred.shape[:-1],1), device=Qpred.device)
            else:
                eGreedyAction = torch.argmax(Qpred, keepdim=True)
        return eGreedyAction


    def decay(self):
        if self.decaySteps is None or self.finalepsilon is None: return
        self.episode += 1
        self.epsilon = self.finalepsilon + (self.initepsilon-self.finalepsilon)*np.exp(-1 * self.episode/self.decaySteps)
