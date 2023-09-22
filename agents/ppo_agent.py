from typing import Any, List

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from env_wrapper import modify_obs, modify_action, modify_obs2, modify_action2


class PPOAgent(Agent):

    def __init__(self, env: CityLearnEnv,  **kwargs: Any):
        super().__init__(env, **kwargs)
        self.model = PPO.load("models/ppo2")
        # print(self.model.policy)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        #  observation: (1x52)
        #  action: (1x9)
        obs = modify_obs(observations)
        action, _ = self.model.predict(obs)
        return modify_action(action)

    def predict2(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        obs = modify_obs2(observations)
        act = []
        #  observation: (3x21)
        #  action: (3x3)
        for i in range(len(obs)):
            action_i, _ = self.model.predict(obs[i])
            act.append(action_i)

        return modify_action2(act)
