import torch
import numpy as np

from utils.misc import soft_update

from model.RANDOMAgent import RANDOMAgent
from model.utils.model import *


class RANDOM(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size // 2
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        params.critic.obs_dim = (self.obs_dim + self.action_dim)

        self.agents = [RANDOMAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            agent.eval()
            actions.append(agent.act(obs, explore=sample).squeeze())
            agent.train()
        return np.array(actions)


    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError