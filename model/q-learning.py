import torch
import numpy as np

from model.QLAgent import QLAgent
from model.utils.model import *


class QLEARNING(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)
        

        params.critic.obs_dim = (self.obs_dim + self.action_dim)

        self.agents = [QLAgent(params) for _ in range(self.num_agents)]
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
        for agent_i, agent in enumerate(self.agents):
            agent.updateQ(obses[agent_i],next_obses[agent_i],actions[agent_i],rewards[agent_i],self.lr,self.gamma)


    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError