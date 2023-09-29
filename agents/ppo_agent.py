from typing import Any, List

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from env_wrapper import modify_obs, modify_action


class PPOAgent(Agent):

    def __init__(self, env: CityLearnEnv,  **kwargs: Any):
        super().__init__(env, **kwargs)
        self.model = PPO.load("models/ppo3")
        # print(self.model.policy)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        obs_modified = modify_obs(observations)
        actions = []
        for i in range(len(obs_modified)):
            action_i, _ = self.model.predict(obs_modified[i])
            actions.append(action_i)

        return modify_action(actions)

