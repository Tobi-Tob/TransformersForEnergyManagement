from typing import Any, List

import numpy as np
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from env_wrapper import modify_obs, modify_action


class PPOAgent(Agent):

    def __init__(self, env: CityLearnEnv,  **kwargs: Any):
        super().__init__(env, **kwargs)
        model_id = 'PPO_1'
        self.ensemble = False  # if True: mean prediction over all 3 models
        self.model_index = 0   # else use model defined by model_index
        self.models = []
        for n in [1, 2, 3]:
            model_n = PPO.load("models/" + model_id + "/m" + str(n))
            # model_n = PPO.load("models/ppo3")
            model_n.policy.set_training_mode(False)
            self.models.append(model_n)
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
            if self.ensemble:
                action_m1, _ = self.models[0].predict(obs_modified[i], deterministic=True)
                action_m2, _ = self.models[1].predict(obs_modified[i], deterministic=True)
                action_m3, _ = self.models[2].predict(obs_modified[i], deterministic=True)
                action_i = (np.array(action_m1) + np.array(action_m2) + np.array(action_m3))/3
                action_i = action_i.tolist()
            else:
                action_i, _ = self.models[self.model_index].predict(obs_modified[i], deterministic=True)
            actions.append(action_i)

            self.all_observations.append(obs_modified[i])

        return modify_action(actions)

    def set_model_index(self, idx):
        if idx < len(self.models):
            self.model_index = idx
        else:
            raise IndexError

    def next_model_index(self):
        self.model_index += 1
        if self.model_index >= len(self.models):
            self.model_index = 0

    def print_normalizations(self):
        print('mean:')
        print(np.mean(self.all_observations, axis=0))
        print('std:')
        print(np.std(self.all_observations, axis=0))
        print('max:')
        print(np.max(self.all_observations, axis=0))
        print('min:')
        print(np.min(self.all_observations, axis=0))

