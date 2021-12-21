# refer https://nn.labml.ai/rl/dqn/replay_buffer.html


import random
import numpy as np

import torch

from . import ReplayBuffer


class PrioritizedExperienceRelpayBuffer(ReplayBuffer):
    """## Prioritized Experince Replay Memory Buffer

        It stores the experieces along with priorities and samples among
        the experieces using the priorities. The experieces can be either
        stored as a windowed memory - the oldest expereicence is deleted 
        to make room for new. Or as a 'replace-min' pool where the experiece 
        with the least priority is deleted instead.

        NOTE: sampling and update have TC of O( batchSize*log2(bufferSize) )
    """

    def __init__(self, bufferSize:int, alpha:float, beta=0.2, beta_rate=0.0001, 
                bufferType='replace-min', beta_schedule=None, numpy_parallelized=False):
        """
        alpha: interpolates from uniform sampling (alpha = 0) to full prioritized sampling (alpha = 1).

        beta: compensates for the bias induced by priorited replay, fully compensates when beta = 1. 
                unbiased nature is more important towards end of training.

        beta_rate: the inverse time-constant for beta increase. By default the beta is updated as 
                    ''beta_ <- min(1, beta + episode*beta_rate)'' at every sampling step.
                    
        beta_schedule: function that takes the step number and returns the next beta
                        this is called at every time sample is produced. Overides beta & beta_rate.

        bufferType: either 'circular' or 'replace-min'
                    circular: buffer as priority sampling in a circular deque (windowed memory)
                    replace-min: replace the experience with min priority if full.

        numpy_parallelized: priority sampling using parallel array comparison with numpy 
                            -numpy_parallelized may be inefficient for large bufferSize

        Example for beta_schedule can be found at ReplayBuffers.helper_funcs.make_exponential_beta_schedule
        """

        assert bufferType in ['circular', 'replace-min']

        self.bufferSize = bufferSize
        self.alpha, self.beta_schedule = alpha, beta_schedule
        self.numpy_parallelized = numpy_parallelized
        self.replace_min =  bufferType == 'replace-min'

        # description of the experience tupple
        self._tuppleDesc = ['state', 'action', 'reward', 'nextState', 'done']
        self.buffer = None

        # sum trees for efficent queries
        self._priority_min = np.array([[float('inf'), -1] for _ in range(2*bufferSize+1)]) # val, idx
        self._priority_sum = np.array([0.0 for _ in range(2*bufferSize+1)])

        self.size = 0
        self.nextidx = 0
        self.max_priority = 1

        # define default beta schedule from beta and beta_rate if schedule not given
        if not beta_schedule:
            self.beta_schedule = lambda episode: min(1, beta + episode*beta_rate)

        # init the beta and episode counter
        self.episode = 0 
        self.beta = self.beta_schedule(0)

    
    def store(self, state:torch.tensor, action:torch.tensor, 
                reward:torch.tensor, nextState:torch.tensor, done:torch.tensor):
        
        # all should be tensors!!
        experienceTupple = [state, action, reward, nextState, done]

        # lazy init buffer
        if self.buffer is None: self.buffer = self._lazy_buffer_init(experienceTupple, self._tuppleDesc)

        idx = self.nextidx  

        # handle replace-min buffer, replace min makes sense only when the buffer is full
        if self.replace_min and self.size >= self.bufferSize:
            idx = self._get_min_idx()

        for i, key in enumerate(self._tuppleDesc):
            self.buffer[key][idx] = experienceTupple[i]
        self.nextidx = (idx + 1) % self.bufferSize
        self.size = min(self.bufferSize, self.size+1)

        priority_alpha = self.max_priority ** self.alpha
        self._update_priority_min(idx, priority_alpha)
        self._update_sum_priority(idx, priority_alpha)


    def sample(self, batchSize:int):

        # sample the buffer according to the priorities
        indices = self._numpy_parallized_weighted_sampling(batchSize) \
                    if self.numpy_parallelized else self._weighted_sampling(batchSize)

        # compute importance sampling weights - following the paper
        prob_min = self._get_min()/self._get_sum()
        max_weight = (self.size * prob_min) ** (-self.beta) # for normalization
        probs = self._priority_sum[indices + self.bufferSize] / self._get_sum()
        weights = (probs * self.size) ** (-self.beta)
        weights = weights / max_weight # normalized importance sampling weights

        # get the experiences
        samples = {k:self.buffer[k][indices] for k in self._tuppleDesc}

        return samples, indices, weights


    def update(self, indices, priorities:torch.Tensor):
        """priorities should be all positive"""
        priorities = priorities.cpu().numpy()
        for idx, priority in zip(indices, priorities):
            assert priority >= 0
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority**self.alpha + 0.00000001
            self._update_sum_priority(idx, priority_alpha)
            self._update_priority_min(idx, priority_alpha)


    def update_params(self):
        # increment episode counter
        self.episode += 1
        
        # update beta
        self.beta = self.beta_schedule(self.episode)


    def _weighted_sampling(self, batchSize):
        """ sample samples according to priorities """
        # ... maybe paralelize this? multiprocessing is not working
        indices = np.zeros(batchSize, dtype=int)
        for i in range(batchSize):
            P = random.uniform(0,1) # cumilative prob
            prefix_sum = P * self._get_sum()
            idx = self._find_predecessor(prefix_sum)
            indices[i] = idx

        return indices


    def _numpy_parallized_weighted_sampling(self, batchSize):
        """parallelization with numpy, not good tc for large buffers"""
        all_probs = self._priority_min[self.bufferSize:self.size+self.bufferSize][:,0]
        cum_probs = np.cumsum(all_probs)[:,None]
        p = np.random.rand(1,batchSize) * self._get_sum()
        indices = np.argmax(cum_probs > p, axis=0)
        return indices


    def _find_predecessor(self, prefix_sum):
        # find the largest idx such that the sum of probablities till idx is <= cumilative_prob
        idx = 1
        while idx < self.bufferSize:
            if self._priority_sum[2*idx] >= prefix_sum:
                idx = 2*idx
            else:
                prefix_sum = prefix_sum - self._priority_sum[2*idx]
                idx = 2*idx + 1

        return idx - self.bufferSize

    
    def _update_priority_min(self, idx, priority_alpha):
        # one indexed in idx -> the min is stored at self.min_priority[1]
        idx = idx + self.bufferSize
        self._priority_min[idx][0] = priority_alpha
        self._priority_min[idx][1] = idx - self.bufferSize
        while idx >= 2:
            idx//=2
            if self._priority_min[2*idx][0] < self._priority_min[2*idx+1][0]:
                self._priority_min[idx] = self._priority_min[2*idx]
            else:
                self._priority_min[idx] = self._priority_min[2*idx+1]


    def _update_sum_priority(self, idx, priority_alpha):
        # one indexed in idx
        idx = idx + self.bufferSize
        self._priority_sum[idx] = priority_alpha
        while idx >= 2:
            idx//=2
            self._priority_sum[idx] = self._priority_sum[2*idx] + self._priority_sum[2*idx+1]


    def _get_sum(self):
        return self._priority_sum[1]
    
    def _get_min(self):
        return self._priority_min[1][0]

    def _get_min_idx(self):
        return int(self._priority_min[1][1])

    def __len__(self):
        return self.size
