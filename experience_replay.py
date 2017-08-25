import random
import torch
from torch.autograd import Variable

class ReplayMemory(object):
    
    def __init__(self, capacity):
        
        # capacity represents the max number of transitions in the memory
        self.capacity = capacity
        
        # the list that will contain the last 100 events
        self.memory = []
    
    """The event represents: previous state, new state, previous action,
        previous reward
    """
    def push(self, event):
        
        self.memory.append(event)
        
        # if we go over the limit - delte the first transition 
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    """ Contains the sample of the memory, takes random sample to have a fixed
        batch size, reshapes the function to be pairs of lists, allow our 
        algorithm to have sample for state, action, and reward
    """
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        
        # put sample in pitorch variable
        return map(lambda x: Variable(torch.cat(x,0)),samples)
        