from typing import List
import numpy as np
import torch

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv

from agents.forecaster import SolarGenerationForecaster, TemperatureForecaster
from env_wrapper import modify_obs, modify_action

from transformers import DecisionTransformerModel

from rewards.custom_reward import CombinedReward


class DTAgent(Agent):

    def __init__(self, env: CityLearnEnv, model_path):
        super().__init__(env)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = DecisionTransformerModel.from_pretrained(model_path, local_files_only=True)
        model = model.to(self.device)
        # print(model.eval())
        self.model = model

        Target_Return = -1
        self.scale = 1000
        self.state_dim = self.model.config.state_dim
        self.act_dim = self.model.config.act_dim
        self.context_length = self.model.config.max_length
        self.TR = Target_Return / self.scale

        self.reward_function = CombinedReward(env.get_metadata()['buildings'][0])

        self.mean = np.load(model_path + '/state_mean.npy')
        self.std = np.load(model_path + '/state_std.npy')
        self.current_timestep = 0

        self.state_history = None
        self.action_history = None
        self.return_to_go_history = None
        self.timestep_history = None

        self.model_id = type(model).__name__


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
            model_id=self.model_id,
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
        if self.current_timestep is 0:
            current_reward = np.zeros(len(self.env.get_metadata()['buildings']))
        else:
            current_reward = self.reward_function.calculate(observations=observations)
        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata, self.current_timestep)
        action_list = []
        for i in range(len(obs_modified)):
            state = np.array(obs_modified[i])
            states = torch.from_numpy(state).reshape(1, 1, self.state_dim).to(device=self.device, dtype=torch.float32)
            actions = torch.zeros((1, 1, self.act_dim), device=self.device, dtype=torch.float32)
            rewards = torch.zeros(1, 1, device=self.device, dtype=torch.float32)
            target_return = torch.tensor(self.TR, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
            attention_mask = torch.zeros(1, 1, device=self.device, dtype=torch.float32)

            with torch.no_grad():
                state_prediction, action_prediction, return_prediction = self.model(
                    states=states,  # TODO Norm
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=target_return,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
            action = action_prediction[0, -1]  # <class 'torch.Tensor'> tensor([-0.2093,  0.5136,  0.0752])
            actions[-1] = action
            action = action.detach().cpu().numpy()  # <class 'numpy.ndarray'> [-0.20933297  0.51364404  0.07517052]

            action_list.append(action)

        self.current_timestep += 1

        return modify_action(action_list, observations, self.building_metadata)

    def set_model_index(self, idx):
        pass

    def print_normalizations(self):
        print(f'{self.model_id} does not support print_normalizations')

    def _reset(self):
        self.reset()
        self.reward_function.reset()
