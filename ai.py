import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os

from network import Network
from experience_replay import ReplayMemory

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        
        # append the average of the rewards to reward_window
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        
        # take 100,000 transition for the model to learn
        self.memory = ReplayMemory(100000)
        
        """create an object using adam optimizer and connect 
        it to nerual network to make sure learning doesn't happen too fast"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # create fake dimension -- unsqueese, tensor wrapped into gradient
        self.prev_state = torch.Tensor(input_size).unsqueeze(0)
        self.prev_action = 0
        self.prev_reward = 0
    
    """Use softmax to select highest probablity of the q-value, then save 
        memory and improve performance by convertign state to gradient """
    def select_action(self,state):
        # T = 100, increasing the temperature makes it look more certain
        probs = F.softmax(self.model(
                Variable(state, volatile=True)) * 100)
        
        # draw randomly from probs
        action = probs.multinomial()
        
        return action.data[0,0]
    
    """ Implement Markov Decision Process """
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(
                1, batch_action.unsqueeze(1)).squeeze(1)
    
        next_outputs = self.model(batch_next_state).detach().max(1)[0]

        # the reward + the next output which is the max of the values of nex state        
        target = self.gamma * next_outputs + batch_reward
        
        # temporal difference lost (predictions- output, target- goal)
        td_loss = F.smooth_l1_loss(outputs,target)
        
        # re initialize optimizer
        self.optimizer.zero_grad()
        
        # backpropogate the temporal difference and free the memory(true)
        td_loss.backward(retain_variables=True)
        
        # update how much contripute to error (weight)
        self.optimizer.step()
        
    """ Update all the elements of our transition and select the action"""
    def update(self, reward, new_signal):
        
        # signal is the state, 3 signals plus orientation, -orientation
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        # update memory and convert the list to tensor
        self.memory.push((self.prev_state, new_state, torch.LongTensor(
                [int(self.prev_action)]), torch.Tensor([self.prev_reward])))
        
        # play an action
        action = self.select_action(new_state)
        
        # the ai starts learning after 100 transitions
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state,
                       batch_reward, batch_action)
        
        # update previous action
        self.prev_action = action
        self.prev_state = new_state
        self.prev_reward = reward
        
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action

    """ Take the sume of all rewards in stored reward and divide by the mean"""
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window)+1.)
    
    """ Save last model and optimizer"""
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, 'previous_brain.pth')
    
    """ Loads the saved file allows us to use that brain"""
    def load(self):
        
        if os.path.isfile('previous_brain.pth'):
            print("=> loading checkpoint...")
            
            checkpoint = torch.load('previous_brain.pth')
            
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print("Done.")
            
        else:
            print("File not found...")
        
        