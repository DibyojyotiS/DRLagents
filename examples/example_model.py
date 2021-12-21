# still being written

from typing import Any
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules.module import T_co


# feedforward stuff is straight forward (pun intended)
class duellingDNN(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(duellingDNN, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.valuelayer = nn.Linear(hDim[-1], 1)
        self.actionadv = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            x = self.activation(layer(x))
        advantages = self.actionadv(x)
        values = self.valuelayer(x)
        qvals = values + (advantages - advantages.mean())
        return qvals


# how to handle models that return more than a Tensor in __call__?
# like models that include recurrent behaviour
# sollution: use a state-full class
class RnnPolicyModel(nn.Module):

    def __init__(self, input_dimension) -> None:
        super().__init__()
        self.rnn_cell = nn.LSTM(input_size=4, hidden_size=8, num_layers=1)

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return super().forward(*input, **kwargs)