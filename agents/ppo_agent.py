from typing import Any, List

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from city_gym_env import modify_obs, modify_action


class PPOAgent(Agent):

    def __init__(self, env: CityLearnEnv,  **kwargs: Any):
        super().__init__(env, **kwargs)
        self.model = PPO.load("models/ppo1")

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        #  observation: (1x49)
        #  action: (1x9)
        obs = modify_obs(observations)
        action, _ = self.model.predict(obs)
        return modify_action(action)
