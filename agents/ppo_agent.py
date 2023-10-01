from typing import Any, List

import numpy as np
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from env_wrapper import modify_obs, modify_action


class PPOAgent(Agent):

    def __init__(self, env: CityLearnEnv,  **kwargs: Any):
        super().__init__(env, **kwargs)
        model_id = 'PPO_test'
        self.models = []
        for n in [1, 2, 3]:
            model_n = PPO.load("models/" + model_id + "/m" + str(n))
            # model_n = PPO.load("models/ppo3")
            model_n.policy.set_training_mode(False)
            self.models.append(model_n)
        self.model_index = 0
        self.model_info = dict(
            model_id=model_id,
            num_timesteps=self.models[self.model_index].num_timesteps,
            learning_rate=self.models[self.model_index].learning_rate,
        )
        print(self.model_info)
        # print(self.model.policy)
        self.all_observations = []

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        obs_modified = modify_obs(observations)
        actions = []
        for i in range(len(obs_modified)):
            action_i, _ = self.models[self.model_index].predict(obs_modified[i], deterministic=True)
            actions.append(action_i)

            self.all_observations.append(obs_modified[i])

        return modify_action(actions)

    def set_model_index(self, idx):
        if idx < len(self.models):
            self.model_index = idx
        else:
            raise IndexError

    def print_normalizations(self):
        print('mean:')
        print(np.mean(self.all_observations, axis=0))
        print('std:')
        print(np.std(self.all_observations, axis=0))

