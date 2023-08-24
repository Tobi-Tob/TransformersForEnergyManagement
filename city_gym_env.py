import itertools
from typing import List

import gym
from citylearn.citylearn import CityLearnEnv


def modify_obs(obs):
    return [j for i in obs for j in i]


def modify_action(action):
    return [action]


# environment wrapper for stable baselines
class CityGymEnv(gym.Env):

    def __init__(self, env: CityLearnEnv):
        self.env = env

        self.num_buildings = len(env.action_space)
        self.action_space = env.action_space[0]  # only in single agent setting
        self.observation_space = env.observation_space[0]  # only in single agent setting
        self.time_steps = env.time_steps
        # TO THINK : normalize or modify the observation space

    def reset(self, **kwargs) -> List[float]:
        obs = self.env.reset()
        return modify_obs(obs)

    def step(self, action: List[float]):
        # If `central_agent` is True, `actions` parameter should be a list of 1 list containing all buildings' actions and follows the ordering of
        # buildings in `buildings`.
        act = modify_action(action)
        obs, reward, done, info = self.env.step(act)
        return modify_obs(obs), sum(reward), done, info

    def render(self):
        return self.env.render()
