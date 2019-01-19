# from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py

import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, reward, mask):
        self.memory.append(Transition(state, action, reward, mask))

    def sample(self):
        transitions = self.memory
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)