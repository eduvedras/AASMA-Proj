import time
import cv2

from recoder import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
from utils.train import set_seed_everywhere
from utils.environment import get_agent_types

from overcooked_ai_py.env import OverCookedEnv

from model.utils.model import *

from utils.agent import find_index

import hydra
from omegaconf import DictConfig


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space
        self.save_replay_buffer = cfg.save_replay_buffer
        # self.env = NormalizedEnv(make_env(cfg.env, discrete_action=self.discrete_action))
        self.env = OverCookedEnv(scenario=self.cfg.env, episode_length=self.cfg.episode_length)
        
        self.exploration_prob = cfg.exploration_prob
        self.exploration_decreasing_decay = cfg.exploration_decreasing_decay
        self.min_exploration_prob = cfg.min_exploration_prob

        self.env_agent_types = get_agent_types(self.env)
        self.agent_indexes = find_index(self.env_agent_types, 'ally')
        self.adversary_indexes = find_index(self.env_agent_types, 'adversary')

        if self.discrete_action:
            cfg.agent.params.obs_dim = self.env.observation_space.n
            cfg.agent.params.action_dim = self.env.action_space.n
            cfg.agent.params.action_range = list(range(cfg.agent.params.action_dim))
        else:
            print("CONTINUOUS ACTION SPACE")

        cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.params.critic.input_dim = cfg.agent.params.obs_dim + cfg.agent.params.action_dim

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.common_reward = cfg.common_reward
        obs_shape = [len(self.env_agent_types), cfg.agent.params.obs_dim]
        action_shape = [len(self.env_agent_types), cfg.agent.params.action_dim if not self.discrete_action else 1]
        reward_shape = [len(self.env_agent_types), 1]
        dones_shape = [len(self.env_agent_types), 1]
        self.replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          reward_shape=reward_shape,
                                          dones_shape=dones_shape,
                                          capacity=int(cfg.replay_buffer_capacity),
                                          device=self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                action = self.agent.act(obs, self.exploration_prob, sample=False)
                obs, rewards, done, info = self.env.step(action)
                rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

                self.video_recorder.record(self.env)

                episode_reward += sum(rewards)[0]
                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:

                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    start_time = time.time()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs = self.env.reset()
                self.exploration_prob = max(self.min_exploration_prob, np.exp(-self.exploration_decreasing_decay*episode))

                #self.ou_percentage = max(0, self.ou_exploration_steps - (self.step - self.num_seed_steps)) / self.ou_exploration_steps

                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
                if self.discrete_action: action = action.reshape(-1, 1)
            else:
                agent_observation = obs[self.agent_indexes]
                agent_actions = self.agent.act(agent_observation, self.exploration_prob,sample=True)
                action = agent_actions

            #if self.step >= self.cfg.num_seed_steps and self.step >= self.agent.batch_size:
                #self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, rewards, done, info = self.env.step(action)
            rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)
            
            self.agent.update(obs, next_obs, action, rewards)

            if episode_step + 1 == self.env.episode_length:
                done = True

            if self.cfg.render:
                cv2.imshow('Overcooked', self.env.render())
                cv2.waitKey(1)

            episode_reward += sum(rewards)[0]

            if self.discrete_action: action = action.reshape(-1, 1)

            dones = np.array([done for _ in self.env.agents]).reshape(-1, 1)

            self.replay_buffer.add(obs, action, rewards, next_obs, dones)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step % 5e2 == 0 and self.save_replay_buffer:
                print('Saving replay buffer...')
                self.replay_buffer.save(self.work_dir, self.step - 1)


@hydra.main(config_path='config', config_name='train_ql')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()