from typing import Any, List

import numpy as np
import torch
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import SAC
from stable_baselines3.common.utils import obs_as_tensor

from agents.forecaster import SolarGenerationForecaster, TemperatureForecaster
from env_wrapper import modify_obs, modify_action


class SACAgent(Agent):

    def __init__(self, env: CityLearnEnv, mode='submission', single_model=None, save_observations=False, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.mode = mode
        self.save_observations = save_observations
        self.models = []
        if mode in ['switch', 'ensemble']:
            model_id = 'SAC_1'
            self.model_index = 0
            for n in [1, 2, 3]:
                model_n = SAC.load("my_models/" + model_id + "/m" + str(n))
                model_n.policy.set_training_mode(False)
                self.models.append(model_n)
        elif mode in ['single']:
            if single_model is None:
                raise TypeError('A model has to be given by single_model, but is None')
            model_id = type(single_model).__name__
            self.models.append(single_model)
            self.models[0].policy.set_training_mode(False)
        else:
            model = SAC.load("my_models/submission_models/SAC_3_m1_4314.zip")
            model_id = type(model).__name__
            self.models.append(model)
            self.models[0].policy.set_training_mode(False)

        SGF = SolarGenerationForecaster()
        TF = TemperatureForecaster()
        self.forecaster = {
            type(SGF).__name__: SGF,
            type(TF).__name__: TF
        }
        names = []
        for forecaster_name in self.forecaster.keys():
            names.append(forecaster_name)

        self.model_info = dict(
            model_id=model_id,
            mode=self.mode,
            forecasters=names,
            num_timesteps=self.models[0].num_timesteps,
            learning_rate=self.models[0].learning_rate,
        )
        if self.save_observations:
            self.all_observations = []
        # print(self.models[0].policy)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        for forecaster in self.forecaster.values():
            forecaster.reset()

        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata)
        actions = []
        for i in range(len(obs_modified)):
            if self.mode is 'ensemble':
                action_m1, _ = self.models[0].predict(obs_modified[i], deterministic=True)
                action_m2, _ = self.models[1].predict(obs_modified[i], deterministic=True)
                action_m3, _ = self.models[2].predict(obs_modified[i], deterministic=True)
                action_i = (np.array(action_m1) + np.array(action_m2) + np.array(action_m3)) / 3
                action_i = action_i.tolist()
            elif self.mode is 'switch':
                action_i, _ = self.models[self.model_index].predict(obs_modified[i], deterministic=True)
            else:
                action_i, _ = self.models[0].predict(obs_modified[i], deterministic=True)

            actions.append(action_i)

            if self.save_observations:
                self.all_observations.append(obs_modified[i])

        return modify_action(actions, self.building_metadata)

    def predict_obs_value(self, observations):
        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata)
        obs_value = 0
        for obs in obs_modified:
            obs_t = obs_as_tensor(np.array(obs).reshape(1, -1), self.models[0].device)
            act_t = obs_as_tensor(np.array([0, 0, 0]).reshape(1, -1), self.models[0].device)
            values = torch.cat(self.models[0].critic(obs_t, act_t), dim=1)
            min_value, _ = torch.min(values, dim=1, keepdim=True)
            obs_value += min_value[0][0].detach().numpy()
        return obs_value / len(obs_modified)

    def set_model_index(self, idx):
        if self.mode is 'switch' or 'ensemble':
            if idx < len(self.models):
                self.model_index = idx
            else:
                raise IndexError

    def next_model_index(self):
        if self.mode is 'switch' or 'ensemble':
            self.model_index += 1
            if self.model_index >= len(self.models):
                self.model_index = 0

    def print_normalizations(self):
        if self.save_observations:
            print('sum:')
            print(np.sum(self.all_observations, axis=0))
            print('mean:')
            print(np.mean(self.all_observations, axis=0))
            print('std:')
            print(np.std(self.all_observations, axis=0))
            print('max:')
            print(np.max(self.all_observations, axis=0))
            print('min:')
            print(np.min(self.all_observations, axis=0))
