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

    def __init__(self, env: CityLearnEnv, model_path='my_models/Decision_Transformer/DT_e345_1'):
        super().__init__(env)

        Target_Return = -300
        self.scale = 1

        model = DecisionTransformerModel.from_pretrained(model_path, local_files_only=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        # print(model.eval())
        self.model = model

        self.state_dim = self.model.config.state_dim
        self.act_dim = self.model.config.act_dim
        self.context_length = self.model.config.max_length
        self.TR = Target_Return / self.scale

        self.reward_function = CombinedReward(env.get_metadata()['buildings'][0])

        self.mean = np.load(model_path + '/state_mean.npy')
        self.std = np.load(model_path + '/state_std.npy')
        self.current_timestep = 0
        self.n_buildings = len(env.get_metadata()['buildings'])

        self.state_history = [torch.zeros(1, self.context_length, self.state_dim, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.action_history = [torch.zeros(1, self.context_length, self.act_dim, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.reward_history = [torch.zeros(self.context_length, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.return_to_go_history = [self.TR * torch.ones(1, self.context_length, 1, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.timestep_history = torch.zeros(1, self.context_length, device=self.device, dtype=torch.int)

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
        self.reset()
        self.reward_function.reset()
        for forecaster in self.forecaster.values():
            forecaster.reset()

        self.current_timestep = 0
        self.state_history = [torch.zeros(1, self.context_length, self.state_dim, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.action_history = [torch.zeros(1, self.context_length, self.act_dim, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.reward_history = [torch.zeros(self.context_length, device=self.device, dtype=torch.float32)] * self.n_buildings
        print('Return-to-go at episode end:', [int(self.return_to_go_history[i][0, 0, -1].detach().item()) for i in range(self.n_buildings)])
        self.return_to_go_history = [self.TR * torch.ones(1, self.context_length, 1, device=self.device, dtype=torch.float32)] * self.n_buildings
        self.timestep_history = torch.zeros(1, self.context_length, device=self.device, dtype=torch.int)

        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        if self.current_timestep is 0:
            all_rewards = np.zeros(len(self.env.get_metadata()['buildings']))
        else:
            all_rewards = self.reward_function.calculate(observations=observations)

        obs_modified = modify_obs(observations, self.forecaster, self.building_metadata, self.current_timestep)
        action_list = []

        self.timestep_history = torch.cat([self.timestep_history[:, -self.context_length + 1:],
                                           torch.ones(1, 1, device=self.device, dtype=torch.int) * self.current_timestep], dim=1)

        for i in range(self.n_buildings):
            with ((torch.no_grad())):
                state = (np.array(obs_modified[i]) - self.mean) / self.std  # normalize state with mean and std from training
                state = torch.from_numpy(state).reshape(1, 1, self.state_dim).to(device=self.device, dtype=torch.float32)  # to tensor
                reward = torch.tensor(all_rewards[i], device=self.device, dtype=torch.float32).reshape(1)

                self.state_history[i] = \
                    torch.cat([self.state_history[i][:, -self.context_length + 1:], state], dim=1)  # append history before model call
                self.reward_history[i] = \
                    torch.cat([self.reward_history[i][-self.context_length + 1:], reward], dim=0)
                pred_return = (self.return_to_go_history[i][0, -1] - (reward / self.scale)).reshape(1, 1, 1)
                self.return_to_go_history[i] = \
                    torch.cat([self.return_to_go_history[i][:, -self.context_length + 1:], pred_return], dim=1)

                number_accessible_states = np.clip(self.current_timestep + 1, a_min=1, a_max=self.context_length)
                zero_padding = self.context_length - number_accessible_states
                assert number_accessible_states + zero_padding == self.context_length
                attention_mask = torch.cat([torch.zeros(zero_padding), torch.ones(number_accessible_states)]).to(dtype=torch.float32).reshape(1, -1)

                state_prediction, action_prediction, return_prediction = self.model(  # predicts the next K steps
                    states=self.state_history[i],
                    actions=self.action_history[i],
                    rewards=self.reward_history[i],
                    returns_to_go=self.return_to_go_history[i],
                    timesteps=self.timestep_history,
                    attention_mask=attention_mask,
                    return_dict=False)

                action = action_prediction[0, -1].detach().cpu().numpy()  # <class 'numpy.ndarray'>

                action[2] = np.clip(action[2], a_min=0, a_max=1)  # clip cooling_device_action to be in [0,1]

                action_tensor = torch.from_numpy(action).reshape(1, 1, self.act_dim).to(device=self.device, dtype=torch.float32)
                self.action_history[i] = \
                    torch.cat([self.action_history[i][:, -self.context_length + 1:], action_tensor], dim=1)  # append history after model call

                # save actions for every building
                action_list.append(action)

        self.current_timestep += 1

        modified_action = modify_action(action_list, observations, self.building_metadata)
        return modified_action

    def set_model_index(self, idx):
        pass

    def print_normalizations(self):
        print(f'{self.model_id} does not support print_normalizations')
