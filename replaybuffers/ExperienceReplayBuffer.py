import random

import torch

from replaybuffers import ReplayBuffer


class ExperienceReplayBuffer(ReplayBuffer):
    """
    ## Experience Replay Buffer
    
        It stores a window of experiences and samples uniformly from the stored
        experiences.
    """

    def __init__(self, bufferSize):
        """
        bufferSize: the size of the memory buffer.
        """
        self.bufferSize = bufferSize
        self._tuppleDesc = ['state', 'action', 'reward', 'nextState', 'done']
        self.buffer = None
        self.nextidx = 0
        self.size = 0


    def store(self, state:torch.tensor, action:torch.tensor, 
                reward:torch.tensor, nextState:torch.tensor, done:torch.tensor):
        
        # all should be tensors!!
        experienceTupple = [state, action, reward, nextState, done]

        # lazy init buffer
        if self.buffer is None: self.buffer = self._lazy_buffer_init(experienceTupple, self._tuppleDesc)

        idx = self.nextidx
        for i, key in enumerate(self._tuppleDesc):
            self.buffer[key][idx] = experienceTupple[i]
        self.nextidx = (idx + 1) % self.bufferSize
        self.size = min(self.bufferSize, self.size + 1)
        

    def sample(self, batchSize):
        indices = random.sample(range(self.size), batchSize)
        samples = {k:self.buffer[k][indices] for k in self._tuppleDesc}
        return samples

    
    def __len__(self):
        return self.size