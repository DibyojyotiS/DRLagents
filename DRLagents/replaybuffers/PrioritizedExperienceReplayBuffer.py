# refer https://nn.labml.ai/rl/dqn/replay_buffer.html
from queue import Queue
import random
from threading import Thread
import time

import numpy as np
import torch
from torch import Tensor
from DRLagents.replaybuffers.helper_funcs.prefetch_utils import threading_prefetch_wrapper

from DRLagents.utils import printDict
from DRLagents.replaybuffers import ReplayBuffer


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
                alpha_rate=0.0, bufferType='replace-min', beta_schedule=None, 
                alpha_schedule=None, print_args=False, nprefetch=0, nthreads=1):
        """
        NOTE: the dtype for the buffer is infered from the first sample stored

        ### parameters
        1. alpha: float 
                - interpolates from uniform sampling (alpha = 0) to full prioritized sampling (alpha = 1).
                - the value of alpha would be different for different training environments.

        2. beta: float (default 0.2)
                - compensates for the bias induced by priorited replay.
                - fully compensates when beta = 1. unbiased nature is more important 
                towards end of training.

        3. beta_rate: float (default 0.0001)
                - anneals beta to 1.0
                - By default the beta is updated as 
                    ''beta_ <- min(1, beta + episode*beta_rate)'' at 
                    every new episode.

        4. alpha_rate: float (default 0.0)
                - anneals alpha to 0
                - By default the alpha is updated as 
                    ''alpha <- min(1, alpha - episode*alpha_rate)'' at 
                    every new episode.

        5. bufferType: str (default 'replace-min')
                - either 'circular' or 'replace-min'
                - circular: buffer as priority sampling in a circular deque (windowed memory)
                - replace-min: replace the experience with min priority if full.
    
        6. beta_schedule: function (default None)
                - that takes the episode number and returns the next beta
                this is called at every time sample is produced. 
                - Overides beta & beta_rate.
                - Example for beta_schedule can be found at \\
                    ReplayBuffers.helper_funcs.make_exponential_beta_schedule

        7. alpha_schedule: function (default None)
                - that takes the episode number and returns the next alpha
                this is called at every time sample is produced. 
                - Overides alpha & alpha_rate.

        8. nprefetch: int (default 0)
                - the number of batches to prefetch. 0 means no prefetching

        9. nthreads: int (default 5)
                - the number of threads to be used for prefetching
        """

        assert bufferType in ['circular', 'replace-min']
        if print_args: printDict(self.__class__.__name__, locals())

        self.bufferSize = bufferSize
        self.beta, self.beta_rate = beta, beta_rate
        self.alpha, self.alpha_rate = alpha, alpha_rate
        self.beta_schedule = beta_schedule
        self.alpha_schedule = alpha_schedule
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
            self.beta_schedule = self._default_beta_schedule
        if not alpha_schedule:
            self.alpha_schedule = self._default_alpha_schedule

        # init the beta and episode counter
        self.episode = 0 
        self.beta = self.beta_schedule(0)

        # init prefetching (prefetch_wrapper handles nprefetch==0)
        self.nprefetch = nprefetch
        self.nthreads = nthreads
        self.sample = threading_prefetch_wrapper(self.sample, nprefetch, nthreads)

        # finally
        self._announce_stuff()

    def __len__(self):
        return self.size

    def _announce_stuff(self):
        if self.replace_min: print('PrioritizedBuffer of type replace-min')
        else: print('PrioritizedBuffer of type circular') 

    def _weighted_sampling(self, batchSize):
        """ weighted sampling according to priorities
        Assumed that sumTree is 1-indexed. 
        
        parameters:
            - self.sumTree: array
                - represents the priorities
                - a complete binary tree in form of array
                - 1-indexed, i.e., sum is stored at sumTree[1]
            - batchSize: int
                - number of samples to draw

        returns: 
            - sampled indices: 
                    - idx retured in range [0, len(sumTree)//2-1]
                    - can be directly indexed into buffer """
        # ... maybe paralelize this? multiprocessing is not working
        indices = np.zeros(batchSize, dtype=int)
        for i in range(batchSize):
            P = random.uniform(0,1) # cumilative prob
            prefix_sum = P * self._get_sum()
            idx = self._find_predecessor(prefix_sum)
            indices[i] = idx

        return indices

    def _find_predecessor(self, prefix_sum):
        """ find the idx (leaf) such that the 
        idx has the smallest value >= prefix_sum
        ### parameters 
            - self.sumTree: array
                - represents the priorities
                - a complete binary tree in form of array
                - 1-indexed, i.e., sum is stored at sumTree[1]
            - prefix_sum: float
                - finds the leaf node just smaller than
                prefix_sum
        
        ### returns
            - predecessor: int
                - index of the predecessor among the leafs
        """
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

    def _default_beta_schedule(self, episode):
        return min(1, self.beta + episode*self.beta_rate)

    def _default_alpha_schedule(self, episode):
        return max(0, self.alpha - episode*self.alpha_rate)

    def store(self, state:Tensor, action:Tensor, 
                reward:Tensor, nextState:Tensor, done:Tensor):
        """ store the experience-tupple: state, action, reward, nextState, done
        in the priority-buffer at the index of the current minimum priority """
        # all should be tensors!!
        experienceTupple = [state, action, reward, nextState, done]

        # lazy init buffer
        if self.buffer is None: self.buffer = self._lazy_buffer_init(
                    experienceTupple, self._tuppleDesc, self.bufferSize)

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
        """ samples a batchSize number of experiences and returns
        a tupple (samples, indices, weights)

        ### parameters
        batchSize: int
        
        ### returns
        - sample: dict[str, Tensor]
                - {'state':tf.Tensor, 'action':tf.Tensor, 'reward':tf.Tensor, 
                    'nextState':tf.Tensor, 'done':tf.Tensor} 
        - indices: np.array
                - the sampled indices
        - weights: np.array
                - normalized importance sampling weights
         """  
        # sample the buffer according to the priorities
        indices = self._weighted_sampling(batchSize)

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
        self.episode += 1 # increment episode counter
        self.beta = self.beta_schedule(self.episode) # update beta
        self.alpha = self.alpha_schedule(self.episode)

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        # do the prefetch init, 
        # prefetch_wrapper handles nprefetch==0
        self.sample = threading_prefetch_wrapper(self.sample, 
                        self.nprefetch,self.nthreads)