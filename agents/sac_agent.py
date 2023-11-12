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

    def __init__(self, env: CityLearnEnv, mode='ensemble', single_model=None, save_observations=False, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.mode = mode
        self.save_observations = save_observations
        self.current_timestep = 0
        self.models = []
        if mode in ['switch', 'ensemble']:
            self.model_index = 0
            model1 = SAC.load("my_models/submission_models/SAC_f3__24446.zip")
            model2 = SAC.load("my_models/submission_models/SAC_f4__20132.zip")
            model3 = SAC.load("my_models/submission_models/SAC_f5_28760.zip")
            model_id = type(model1).__name__
            self.models.append(model1)
            self.models.append(model2)
            self.models.append(model3)
            for model in self.models:
                model.policy.set_training_mode(False)
        elif mode in ['single']:
            if single_model is None:
                raise TypeError('A model has to be given by single_model, but is None')
            model_id = type(single_model).__name__
            self.models.append(single_model)
            self.models[0].policy.set_training_mode(False)
        else:
            model = SAC.load("my_models/submission_models/SAC_f5_28760.zip")
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
        # SACPolicy(
        #   (actor): Actor(
        #     (features_extractor): FlattenExtractor(
        #       (flatten): Flatten(start_dim=1, end_dim=-1)
        #     )
        #     (latent_pi): Sequential(
        #       (0): Linear(in_features=26, out_features=256, bias=True)
        #       (1): ReLU()
        #       (2): Linear(in_features=256, out_features=256, bias=True)
        #       (3): ReLU()
        #     )
        #     (mu): Linear(in_features=256, out_features=3, bias=True)
        #     (log_std): Linear(in_features=256, out_features=3, bias=True)
        #   )
        #   (critic): ContinuousCritic(
        #     (features_extractor): FlattenExtractor(
        #       (flatten): Flatten(start_dim=1, end_dim=-1)
        #     )
        #     (qf0): Sequential(
        #       (0): Linear(in_features=29, out_features=256, bias=True)
        #       (1): ReLU()
        #       (2): Linear(in_features=256, out_features=256, bias=True)
        #       (3): ReLU()
        #       (4): Linear(in_features=256, out_features=1, bias=True)
        #     )
        #     (qf1): Sequential(
        #       (0): Linear(in_features=29, out_features=256, bias=True)
        #       (1): ReLU()
        #       (2): Linear(in_features=256, out_features=256, bias=True)
        #       (3): ReLU()
        #       (4): Linear(in_features=256, out_features=1, bias=True)
        #     )
        #   )
        #   (critic_target): ContinuousCritic(
        #     (features_extractor): FlattenExtractor(
        #       (flatten): Flatten(start_dim=1, end_dim=-1)
        #     )
        #     (qf0): Sequential(
        #       (0): Linear(in_features=29, out_features=256, bias=True)
        #       (1): ReLU()
        #       (2): Linear(in_features=256, out_features=256, bias=True)
        #       (3): ReLU()
        #       (4): Linear(in_features=256, out_features=1, bias=True)
        #     )
        #     (qf1): Sequential(
        #       (0): Linear(in_features=29, out_features=256, bias=True)
        #       (1): ReLU()
        #       (2): Linear(in_features=256, out_features=256, bias=True)
        #       (3): ReLU()
        #       (4): Linear(in_features=256, out_features=1, bias=True)
        #     )
        #   )
        # )

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        self.current_timestep = 0
        for forecaster in self.forecaster.values():
            forecaster.reset()

        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata, self.current_timestep)
        actions = []
        for i in range(len(obs_modified)):
            if self.mode is 'ensemble':
                ensemble_actions = []
                for m, model in enumerate(self.models):
                    action_m, _ = self.models[m].predict(obs_modified[i], deterministic=True)
                    ensemble_actions.append(np.array(action_m))

                action_i = sum(ensemble_actions) / len(self.models)  # mean action of the ensemble
                action_i = action_i.tolist()
            elif self.mode is 'switch':
                action_i, _ = self.models[self.model_index].predict(obs_modified[i], deterministic=True)
            else:
                action_i, _ = self.models[0].predict(obs_modified[i], deterministic=True)

            actions.append(action_i)

            if self.save_observations:
                self.all_observations.append(obs_modified[i])

        self.current_timestep += 1

        return modify_action(actions, observations, self.building_metadata)

    def predict_obs_value(self, observations):
        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata, self.current_timestep)
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
