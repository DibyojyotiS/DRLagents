import numpy as np

def movingAverage(array, n=5):
    array = np.cumsum(array)
    array[n:] -= array[:-n]
    return array/n