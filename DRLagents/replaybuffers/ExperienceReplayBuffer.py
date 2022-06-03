import random

import torch

from DRLagents.replaybuffers.helper_funcs.prefetch_utils import threading_prefetch_decorator, threading_prefetch_wrapper

from . import ReplayBuffer
from ..utils import printDict


class ExperienceReplayBuffer(ReplayBuffer):
    """
    ## Experience Replay Buffer
    
        It stores a window of experiences and samples uniformly from the stored
        experiences.
    """
    def __init__(self, bufferSize, nprefetch=0, nthreads=5, print_args=False):
        """
        It stores a maximum of bufferSize number of experience tupples. And 
        allows uniform sampling among the stored. The older tupples are replaced
        for new ones once size reaches bufferSize.
        ### parameters
        1. bufferSize: int
                - the size of the memory buffer.
        2. nprefetch: int (default 0)
                - the number of batches to prefetch. 0 means no prefetching

        3. nthreads: int (default 5)
                - the number of threads to be used for prefetching
        """
        if print_args: printDict(self.__class__.__name__, locals())
        self.bufferSize = bufferSize
        self.nprefetch = nprefetch
        self.nthreads = nthreads

        self._tuppleDesc = ['state', 'action', 'reward', 'nextState', 'done']
        self.buffer = None
        self.nextidx = 0
        self.size = 0
        
        self.nprefetch = nprefetch
        self.nthreads = nthreads
        self.sample = threading_prefetch_wrapper(self.sample, nprefetch, nthreads)
    
    def __len__(self):
        return self.size

    def store(self, state:torch.tensor, action:torch.tensor, 
                reward:torch.tensor, nextState:torch.tensor, done:torch.tensor):
        
        # all should be tensors!!
        experienceTupple = [state, action, reward, nextState, done]

        # lazy init buffer
        if self.buffer is None: self.buffer = self._lazy_buffer_init(experienceTupple, self._tuppleDesc, self.bufferSize)

        idx = self.nextidx
        for i, key in enumerate(self._tuppleDesc):
            self.buffer[key][idx] = experienceTupple[i]
        self.nextidx = (idx + 1) % self.bufferSize
        self.size = min(self.bufferSize, self.size + 1)
    
    def sample(self, batchSize):
        indices = random.sample(range(self.size), batchSize)
        samples = {k:self.buffer[k][indices] for k in self._tuppleDesc}
        return samples

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        # do the prefetch init, 
        # prefetch_wrapper handles nprefetch==0
        self.sample = threading_prefetch_wrapper(self.sample, 
                        self.nprefetch,self.nthreads)