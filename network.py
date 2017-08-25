# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        
        # usethe tools from nn.Module
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        # connect the input to the 30 hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        
        # activitate the hidden neruon - x
        x = F.relu(self.fc1(state))
        
        # return output neruon
        q_values = self.fc2(x)
        return q_values