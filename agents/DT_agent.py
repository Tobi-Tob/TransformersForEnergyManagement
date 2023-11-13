from typing import List
import numpy as np

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv

from agents.forecaster import SolarGenerationForecaster, TemperatureForecaster
from env_wrapper import modify_obs, modify_action

from transformers import DecisionTransformerModel

class DTAgent(Agent):

    def __init__(self, env: CityLearnEnv, model_path):
        super().__init__(env)

        model = DecisionTransformerModel.from_pretrained(model_path, local_files_only=True)

        self.model = model
        self.mean = np.load(model_path + '/state_mean.npy')
        self.std = np.load(model_path + '/state_std.npy')
        self.current_timestep = 0
        self.context_length = self.model.config.max_length

        model_id = type(model).__name__


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
            model_path=model_path,
            forecasters=names
        )

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self._reset()
        self.current_timestep = 0
        for forecaster in self.forecaster.values():
            forecaster.reset()

        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata, self.current_timestep)
        actions = []
        for i in range(len(obs_modified)):
            state_preds_i, action_preds_i, return_preds_i = self.model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )
            actions.append(action_preds_i)

        self.current_timestep += 1

        return modify_action(actions, observations, self.building_metadata)

    def set_model_index(self, idx):
        pass

    def _reset(self):
        self.reset()
        # TODO
