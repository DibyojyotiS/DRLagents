from tkinter import N
from torch import Tensor


class MaxQvalue:
    def __init__(self) -> None:
        """stores the max of each q-value seen,
        this is expected to be called by a 1D tensor
        or a 2D tensor of shape (1,x)"""
        self.max_qvals = None

    def __call__(self, qvalue:Tensor) -> None:
        if self.max_qvals is None:
            self.max_qvals = qvalue.detach().clone()
        else:
            self.max_qvals = self.max_qvals.max(qvalue.detach())

    def get_max_qvalues(self):
        return self.max_qvals.flatten().tolist()