
import numpy as np
import random
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env

class CustomEnvWrapper(gym.Env):
    def __init__(self, env, representation_model):
        super().__init__()
        self.env = env
        self.representation_model = representation_model
        # Change the observation space to match the output of your representation model
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=self.env.observation_space.dtype)
        self.action_space = env.action_space

    # @property
    # def observation_space(self):
    #     return spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=self.env.observation_space.dtype)

    # @property
    # def action_space(self):
    #     return spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, dtype=self.env.action_space.dtype)

    def reset(self,  seed=None, options=None):
        #self.env.seed(random.randint(0, 4))
        obs, reset_info = self.env.reset()
        # print(obs)
        obs_tensor = torch.tensor(obs, dtype=torch.float)
        dim = len(obs_tensor)
        transformed_obs = self.representation_model(obs_tensor.view(1,dim))
        transformed_obs = transformed_obs.detach().numpy()
        return transformed_obs, reset_info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float)
        dim = len(obs_tensor)
        transformed_obs = self.representation_model(obs_tensor.view(1,dim))
        transformed_obs = transformed_obs.detach().numpy()
        return transformed_obs, reward, done, truncated, info
