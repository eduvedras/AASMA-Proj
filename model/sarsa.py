import torch
import numpy as np

from utils.misc import soft_update

from model.SARSAAgent import SARSAAgent
from model.utils.model import *


class SARSA(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        
        #self.qtable = np.zeros((params.obs_dim, params.action_dim))

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size // 2
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)
        
        
        #self.exploration_prob = params.exploration_prob
        #self.exploration_decreasing_decay = params.exploration_decreasing_decay
        #self.min_exploration_prob = params.min_exploration_prob
        #self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        params.critic.obs_dim = (self.obs_dim + self.action_dim)

        self.agents = [SARSAAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def act(self, observations, expl ,sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            agent.eval()
            actions.append(agent.act(obs, expl, explore=sample).squeeze())
            agent.train()
        return np.array(actions)


    def update(self, obses, next_obses, actions, rewards):
        #sample = replay_buffer.sample(self.batch_size, nth=self.agent_index)
        #obses, actions, rewards, next_obses, dones = sample
        
        #self.qtable[obses, actions] = (1 - self.lr) * self.qtable[obses, actions] + self.lr * (rewards + self.gamma * np.max(self.qtable[next_obses, :], axis=1) * (1 - dones))

        '''if self.discrete_action:
            actions = number_to_onehot(actions)
            actions = torch.max(actions.long(), 1)[1]'''

        for agent_i, agent in enumerate(self.agents):
            agent.updateQ(obses[agent_i],next_obses[agent_i],actions[agent_i],rewards[agent_i],self.lr,self.gamma)


    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError