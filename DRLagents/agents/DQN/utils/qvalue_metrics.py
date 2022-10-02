from tkinter import N
from torch import Tensor


class MaxQvalue:
    def __init__(self) -> None:
        self.max_qvals = None

    def __call__(self, qvalue:Tensor) -> None:
        if self.max_qvals is None:
            self.max_qvals = qvalue.detach().clone()
        else:
            self.max_qvals = self.max_qvals.max(qvalue.detach())

    def get_max_qvalues(self):
        return self.max_qvals.flatten().tolist()