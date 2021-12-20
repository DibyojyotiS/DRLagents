from typing import Any
import torch
from torch import nn, Tensor
from torch.nn.modules.module import T_co

# how to handle models that return more than a Tensor in __call__?
# like models that include recurrent behaviour
# sollution: use a state-full class

class RnnPolicyModel(nn.Module):

    def __init__(self, input_dimension) -> None:
        super().__init__()
        self.rnn_cell = nn.LSTM(input_size=4, hidden_size=8, num_layers=1)

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return super().forward(*input, **kwargs)