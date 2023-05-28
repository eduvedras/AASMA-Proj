from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import tensorflow as tf
from model.network import BCNetwork
from model.utils.model import *
from model.utils.noise import OUNoise


class SARSAAgent(nn.Module):
    """
    General class for BC agents (policy, exploration noise)
    """

    def __init__(self, params):
        super(SARSAAgent, self).__init__()

        self.lr = params.lr
        self.gamma = params.gamma

        self.states = []
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.device = params.device
        self.discrete_action = params.discrete_action_space
        self.hidden_dim = params.hidden_dim

        constrain_out = not self.discrete_action
        
        self.qtable = np.zeros((self.obs_dim, self.action_dim))
        

        #self.policy = BCNetwork(self.obs_dim, self.action_dim, hidden_dim=self.hidden_dim)
        # self.target_policy = BCNetwork(self.obs_dim, self.action_dim, hidden_dim=self.hidden_dim)

        # hard_update(self.target_policy, self.policy)
        #self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=1e-08)

        # self.exploration = OUNoise(self.action_dim)

        self.num_heads = 100

    def act(self, obs, expl, explore=False):

        #if obs.dim() == 1:
        #    obs = obs.unsqueeze(dim=0)

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
        aux = self.states.index(curr)
        aux2 = self.states.index(nxt)
        
        next_action = np.random.randint(1, self.action_dim)
        self.qtable[aux, action] = self.qtable[aux, action] + lr*(reward + gamma*self.qtable[aux2, next_action] - self.qtable[aux, action])
        

    def get_params(self):
        return {'qtable': self.qtable,
                # 'critic': self.critic.state_dict(),
                # 'target_policy': self.target_policy.state_dict(),
                # 'target_critic': self.target_critic.state_dict(),
                #'policy_optimizer': self.policy_optimizer.state_dict(),
                # 'critic_optimizer': self.critic_optimizer.state_dict()
                }