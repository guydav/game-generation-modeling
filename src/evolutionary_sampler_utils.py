from collections import defaultdict
import typing

import numpy as np

class Selector():
    def update(self, key, reward):
        raise NotImplementedError
    
    def select(self, keys, rng):
        raise NotImplementedError

DEFAULT_EXPLORATION_CONSTANT = np.sqrt(0.5)
DEFAULT_BUFFER_SIZE = 100

class UCBSelector(Selector):
    '''
    Implements the Upper Confidence Bound (UCB) algorithm for selecting from a provided
    set of keys (arms). Takes in an exploration constant 'c' and an optional buffer size
    '''
    def __init__(self,
                 c: float = DEFAULT_EXPLORATION_CONSTANT,
                 buffer_size: typing.Optional[int] = DEFAULT_BUFFER_SIZE):
        

        self.c = c
        self.buffer_size = buffer_size

        self.reward_map = defaultdict(list)
        self.count_map = defaultdict(int)

        self.n_draws = 0

    def update(self, key, reward):
        '''
        Updates the buffer for the arm specified by the key with the provided reward. If the buffer
        is full, the oldest reward is removed
        '''
        self.reward_map[key].append(reward)
        if self.buffer_size is not None and len(self.reward_map[key]) > self.buffer_size:
            self.reward_map[key].pop(0)

    def select(self, keys, rng):
        '''
        Given a list of keys, returns the key with the highest UCB score and updates the internal
        counter for the selected key (and overall count)
        '''
        ucb_values = [sum(self.reward_map[key]) / self.count_map[key] + self.c * np.sqrt(np.log(self.n_draws) / self.count_map[key])
                      if key in self.count_map else float('inf') for key in keys]
        
        max_index = np.argmax(ucb_values)
        max_key = keys[max_index]

        self.count_map[max_key] += 1
        self.n_draws += 1

        return max_key
    
class ThompsonSamplingSelector(Selector):
    '''
    Implements the Thompson Sampling algorithm for selecting from a provided set of keys (arms). 
    Assumes that the rewards are Bernoulli distributed (i.e. 0 or 1)
    '''
    def __init__(self,
                 prior_alpha: int = 1,
                 prior_beta: int = 1,
                 buffer_size: typing.Optional[int] = DEFAULT_BUFFER_SIZE):
        
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.buffer_size = buffer_size

        self.reward_map = defaultdict(list)

    def update(self, key, reward):
        self.reward_map[key].append(reward)
        if self.buffer_size is not None and len(self.reward_map[key]) > self.buffer_size:
            self.reward_map[key].pop(0)

    def select(self, keys, rng):
        '''
        Given a list of keys, returns the key with the highest sampled mean
        '''
        thompson_values = [rng.beta(self.reward_map[key].count(1) + self.prior_alpha, self.reward_map[key].count(0) + self.prior_beta) for key in keys]
        max_index = np.argmax(thompson_values)
        max_key = keys[max_index]

        return max_key