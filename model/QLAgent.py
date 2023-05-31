import torch.nn as nn
from model.utils.model import *


class QLAgent(nn.Module):
    """
    General class for QL agents
    """

    def __init__(self, params):
        super(QLAgent, self).__init__()

        self.lr = params.lr
        self.gamma = params.gamma

        self.states = []
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        
        self.qtable = np.zeros((self.obs_dim*2, self.action_dim))

    def act(self, obs, expl, explore=False):
        state = obs.detach().cpu().numpy().tolist()
        if state not in self.states:
            self.states.append(state)
        aux = self.states.index(state)
        action = torch.Tensor(self.qtable[aux]).unsqueeze(dim=0)
        
        action = onehot_from_logits(action, eps=expl)
        action = onehot_to_number(action)

        return action.detach().cpu().numpy()
    
    def updateQ(self,current_state, next_state, action,reward,lr,gamma):
        curr = current_state.tolist()
        nxt = next_state.tolist()
        if curr not in self.states:
            self.states.append(curr)
        if nxt not in self.states:
            self.states.append(nxt)
            
        curr_index = self.states.index(curr)
        nxt_index = self.states.index(nxt)
        self.qtable[curr_index, action] = (1-lr)*self.qtable[curr_index, action] 
        self.qtable[curr_index, action] += lr*(reward + gamma*np.max(self.qtable[nxt_index,:]))
        

    def get_params(self):
        return {'qtable': self.qtable,}