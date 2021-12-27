import numpy as np
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def movingAverage(array, n=5):
    array = np.cumsum(array)
    array[n:] -= array[:-n]
    return array/n