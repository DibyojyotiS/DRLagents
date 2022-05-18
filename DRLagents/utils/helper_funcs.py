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


def printDict(name, _dict:dict):
    print(name, ':')
    for k in _dict:
        if k =='self': continue
        if type(_dict[k]) == torch.nn.Module: continue
        print('\t',k,':',_dict[k])
    print('\n')