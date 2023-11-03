from random import randint
from typing import List
import numpy as np
from citylearn.citylearn import CityLearnEnv
from numpy import ndarray
from gym import Env, spaces

from agents.forecaster import SolarGenerationForecaster, TemperatureForecaster

'''
File to modify the observation and action space and wrap the citylearn environment
'''


def modify_obs(obs: List[List[float]], forecaster: dict, metadata, current_timestep) -> List[List[float]]:
    """
    Input: (1,52), Output: (3, 26)
    Modify the observation space to:
    [[relative_timestep, day_type, hour, outdoor_dry_bulb_temperature, outdoor_temperature_1h_predicted,
    carbon_intensity, mean_district_dhw_storage, mean_district_electrical_storage,
    indoor_dry_bulb_temperature, non_shiftable_load, solar_generation_1h_predicted, dhw_storage_soc,
    electrical_storage_soc, net_electricity_consumption, cooling_demand, dhw_demand, occupant_count,
    indoor_temperature_difference_to_set_point, power_outage, relative_non_shiftable_load,
    relative_solar_generation_1h_predicted, relative_dhw_storage_soc, relative_electrical_storage_soc,
    relative_net_electricity_consumption, relative_cooling_demand, relative_dhw_demand],...]
    """
    #  --> Delete unimportant observations like pricing, 12 and 24 h predictions
    #  --> Add usefully observation e.g. temperature_diff
    #  --> Pre-process observation with building specific info e.g. pv power, annual non-shiftable load estimate...
    #  --> Include info of other buildings e.g. mean storage levels
    #  --> Use historic weather forecast information
    #  --> Use building solar forecaster, temp forecaster and building power forecaster?
    #  --> Normalize observations when possible

    # Read metadata
    buildings = []
    pv_nominal_powers = []
    for building_metadata in metadata:
        buildings.append(building_metadata['name'])
        pv_nominal_powers.append(building_metadata['pv']['nominal_power'])

    relative_timestep = current_timestep / metadata[0]['simulation_time_steps']
    solar_generation_1h = forecaster['SolarGenerationForecaster'].forecast(obs, pv_nominal_powers[0])
    temperature_1h = forecaster['TemperatureForecaster'].forecast(obs)

    obs = obs[0]
    obs_buildings = obs[15:21] + obs[25:]  # all building level observations (#buildings * 11)

    # building-level observations:
    assert len(obs_buildings) % 11 == 0  # 11 observations per building
    obs_single_building = [obs_buildings[i:i + 11] for i in range(0, len(obs_buildings), 11)]
    assert len(obs_single_building) == len(buildings)
    dhw_storage_sum = 0
    electrical_storage_sum = 0
    for i, b in enumerate(obs_single_building):
        temperature_diff = b[0] - b[9]  # indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point
        b[9] = temperature_diff  # replace with temperature difference to set point
        b[2] = solar_generation_1h * pv_nominal_powers[i]  # replace with solar generation prediction

        dhw_storage_sum += b[3]  # to calculate mean_district_dhw_storage
        electrical_storage_sum += b[4]  # to calculate mean_district_electrical_storage

        b.append(b[1])  # relative_non_shiftable_load
        b.append(b[2])  # relative_solar_generation_1h_predicted
        b.append(b[3])  # relative_dhw_storage_soc
        b.append(b[4])  # relative_electrical_storage_soc
        b.append(b[5])  # relative_net_electricity_consumption
        b.append(b[6])  # relative_cooling_demand
        b.append(b[7])  # relative_dhw_demand
        # factor is calculated in _get_obs_normalization()
        assert len(b) == 18

    mean_district_dhw_storage = dhw_storage_sum / len(buildings)
    mean_district_electrical_storage = electrical_storage_sum / len(buildings)

    # all important district level observations: (8)
    obs_district = [relative_timestep, obs[0], obs[1], obs[2], temperature_1h, obs[14], mean_district_dhw_storage, mean_district_electrical_storage]

    obs_modified = []
    normalizations = _get_obs_normalization(metadata)
    for i, b in enumerate(obs_single_building):
        obs = obs_district + b
        for j in range(len(obs)):
            mean = normalizations[j][0][i] if isinstance(normalizations[j][0], list) else normalizations[j][0]
            std = normalizations[j][1][i] if isinstance(normalizations[j][1], list) else normalizations[j][1]

            obs[j] = (obs[j] - mean) / std

        assert len(obs) == 8 + 18
        obs_modified.append(obs)

    return obs_modified


def modify_action(action: List[ndarray], metadata) -> List[List[float]]:
    """
    Input: (3,3), Output: (1, 9), values are modified with corresponding building specific constants.
    """
    cooling_nominal_powers = []
    dhw_storage_capacity = []
    electrical_storage_capacity = []

    for building_metadata in metadata:
        cooling_nominal_powers.append(building_metadata['cooling_device']['nominal_power'])
        dhw_storage_capacity.append(building_metadata['dhw_storage']['capacity'])
        electrical_storage_capacity.append(building_metadata['electrical_storage']['capacity'])

    for i in range(len(metadata)):
        action[i][0] = action[i][0] / cooling_nominal_powers[i]
        action[i][1] = action[i][1] / dhw_storage_capacity[i]
        action[i][2] = action[i][2] / electrical_storage_capacity[i]

    return [np.concatenate(action).tolist()]


def _get_obs_normalization(metadata):
    dhw_storage_soc_capacity = []
    electrical_storage_soc_capacity = []
    cooling_demand_estimate = []
    dhw_demand_estimate = []
    non_shiftable_load_estimate = []
    solar_generation_estimate = []
    net_e_consumption_estimate = []
    for bm in metadata:
        dhw_storage_soc_capacity.append(1 / bm['dhw_storage']['capacity'])
        electrical_storage_soc_capacity.append(1 / bm['electrical_storage']['capacity'])
        cooling_demand = bm['annual_cooling_demand_estimate'] / bm['simulation_time_steps']
        cooling_demand_estimate.append(cooling_demand)
        # [2400.07568359375, 1256.208740234375, 1516.25146484375]
        dhw_demand = bm['annual_dhw_demand_estimate'] / bm['simulation_time_steps']
        dhw_demand_estimate.append(dhw_demand)
        # [153.8460235595703, 45.04438781738281, 109.74966430664062]
        non_shiftable_load = bm['annual_non_shiftable_load_estimate'] / bm['simulation_time_steps']
        non_shiftable_load_estimate.append(non_shiftable_load)
        # [450.445068359375, 323.14483642578125, 631.7621459960938]
        solar_generation = bm['annual_solar_generation_estimate'] / bm['simulation_time_steps']
        solar_generation_estimate.append(solar_generation)
        # [345.7142639160156, 172.8571319580078, 345.7142639160156]
        net_e_consumption_estimate.append(cooling_demand + dhw_demand + non_shiftable_load - solar_generation)

    normalizations = [
        #   -mean-      -std-
        [0.00000000, 1.00000000],  # relative_timestep (unchanged)
        [4.09861111, 1.97132168],  # day_type (normalized)
        [12.4680556, 6.92211284],  # hour (normalized)
        [24.2984569, 4.00000000],  # outdoor_dry_bulb_temperature (subtract the mean temp set point, divide by 4)
        [24.2984569, 4.00000000],  # outdoor_temperature_1h_prediction (subtract the mean temp set point)
        [0.45429827, 0.04875349],  # carbon_intensity (normalized)
        [0.00000000, 1.00000000],  # mean_district_dhw_storage (unchanged)
        [0.00000000, 1.00000000],  # mean_district_electrical_storage (unchanged)
        [24.2984569, 4.00000000],  # indoor_dry_bulb_temperature (subtract the mean temp set point)
        [0.00000000, 1.00000000],  # non_shiftable_load (unchanged)
        [0.00000000, 1.00000000],  # solar_generation_1h_predicted (unchanged)
        [0.00000000, dhw_storage_soc_capacity],  # dhw_storage_soc (fill level * capacity)
        [0.00000000, electrical_storage_soc_capacity],  # electrical_storage_soc (fill level * capacity)
        [0.00000000, 1.00000000],  # net_electricity_consumption (unchanged)
        [0.00000000, 1.00000000],  # cooling_demand (unchanged)
        [0.00000000, 1.00000000],  # dhw_demand (unchanged)
        [0.00000000, 1.00000000],  # occupant_count (unchanged)
        [0.00000000, 4.00000000],  # temperature_difference_to_set_point (divide by 4)
        [0.00000000, 1.00000000],  # power_outage (unchanged)
        [non_shiftable_load_estimate, non_shiftable_load_estimate],  # relative_non_shiftable_load (relative to buildings estimate)
        [solar_generation_estimate, solar_generation_estimate],  # relative_solar_generation_1h_predicted (relative to buildings estimate)
        [0.00000000, 1.00000000],  # relative_dhw_storage_soc (unchanged)
        [0.00000000, 1.00000000],  # relative_electrical_storage_soc (unchanged)
        [net_e_consumption_estimate, net_e_consumption_estimate],  # relative_net_electricity_consumption (relative to buildings estimate)
        [cooling_demand_estimate, cooling_demand_estimate],  # relative_cooling_demand (relative to buildings estimate)
        [dhw_demand_estimate, dhw_demand_estimate],  # relative_dhw_demand (relative to buildings estimate)(high std 2.9 and max 52)
    ]
    return normalizations


def get_modified_observation_space():
    observation_dim = 26
    low_limit = np.zeros(observation_dim)
    high_limit = np.zeros(observation_dim)
    low_limit[0], high_limit[0] = 0, 1  # relative_timestep
    low_limit[1], high_limit[1] = -1.58, 1.48  # day_type
    low_limit[2], high_limit[2] = -1.66, 1.67  # hour
    low_limit[3], high_limit[3] = -0.73, 4.01  # outdoor_dry_bulb_temperature
    low_limit[4], high_limit[4] = -0.73, 4.01  # outdoor_temperature_1h_prediction
    low_limit[5], high_limit[5] = -2.40, 2.09  # carbon_intensity
    low_limit[6], high_limit[6] = 0, 1  # mean_district_dhw_storage
    low_limit[7], high_limit[7] = 0, 1  # mean_district_electrical_storage
    low_limit[8], high_limit[8] = -5, 5  # indoor_dry_bulb_temperature
    low_limit[9], high_limit[9] = 0, 8.83  # non_shiftable_load
    low_limit[10], high_limit[10] = 0, 3  # solar_generation_1h_predicted
    low_limit[11], high_limit[11] = 0, 2.85  # dhw_storage_soc
    low_limit[12], high_limit[12] = 0, 4  # electrical_storage_soc
    low_limit[13], high_limit[13] = -5, 20  # net_electricity_consumption
    low_limit[14], high_limit[14] = 0, 10  # cooling_demand
    low_limit[15], high_limit[15] = 0, 10  # dhw_demand
    low_limit[16], high_limit[16] = 0, 3  # occupant_count
    low_limit[17], high_limit[17] = -10, 10  # temperature_difference to set point
    low_limit[18], high_limit[18] = 0, 1  # power_outage
    low_limit[19], high_limit[19] = -1, 20  # relative_non_shiftable_load
    low_limit[20], high_limit[20] = -1, 3  # relative_solar_generation_1h_predicted
    low_limit[21], high_limit[21] = 0, 1  # relative_dhw_storage_soc
    low_limit[22], high_limit[22] = 0, 1  # relative_electrical_storage_soc
    low_limit[23], high_limit[23] = -5, 20  # relative_net_electricity_consumption
    low_limit[24], high_limit[24] = -1, 100  # relative_cooling_demand
    low_limit[25], high_limit[25] = -1, 100  # relative_dhw_demand

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


def get_modified_action_space():
    action_dim = 3
    low_limit = np.zeros(action_dim)
    high_limit = np.zeros(action_dim)
    low_limit[0], high_limit[0] = -2.85, 2.85  # dhw_storage_action (max of all buildings)
    low_limit[1], high_limit[1] = -4, 4  # electrical_storage_action (max of all buildings)
    low_limit[2], high_limit[2] = 0, 4.11  # cooling_device_action (max of all buildings)

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


class CityEnvForTraining(Env):
    # EnvWrapper Class used for training, controlling one building interactions
    def __init__(self, env: CityLearnEnv):
        self.env = env
        self.metadata = env.get_metadata()['buildings']
        self.evaluation_model = None

        SGF = SolarGenerationForecaster()
        TF = TemperatureForecaster()
        self.forecaster = {
            type(SGF).__name__: SGF,
            type(TF).__name__: TF
        }

        self.num_buildings = len(env.buildings)
        self.active_building_ID = randint(0, 2)
        self.current_timestep = 0
        self.action_space = get_modified_action_space()
        self.observation_space = get_modified_observation_space()
        self.time_steps = env.time_steps

    def set_evaluation_model(self, model):
        """
        The evaluation model is used for generating actions for all buildings that are not actively included in the training process at the moment
        """
        self.evaluation_model = model

    def reset(self, **kwargs) -> List[float]:
        self.current_timestep = 0
        for forecaster in self.forecaster.values():
            forecaster.reset()
        self.active_building_ID += 1
        if self.active_building_ID >= self.num_buildings:
            self.active_building_ID = 0

        random_seed = randint(0, 99999)
        for b in self.env.buildings:
            b.stochastic_power_outage_model.random_seed = random_seed

        obs = modify_obs(self.env.reset(), self.forecaster, self.metadata, self.current_timestep)

        # if self.evaluation_model is not None:
        # test_obs = [1.4717988035316487, 0.36577623892071665, 0.4605091146284094, -0.6613356282905993, 1.6623099623810385, -0.8392258613092881,
        #             0.8102464956499016, -0.8746554498111345, 1.3965325787525062, 1.8631778991821282, 0.44612357020378113, 1.3038498163223267, 1.0,
        #             0.19288472831249237, -0.555773913860321, 0.0]
        # test_action, _ = self.evaluation_model.predict(test_obs, deterministic=True)
        # print(test_action)

        return obs[self.active_building_ID]

    def step(self, action_of_active_building: List[float]):
        # only return visible state of the active building
        # get observation of this timestep
        observations_of_all_buildings = modify_obs(self.env.observations, self.forecaster, self.metadata, self.current_timestep)
        actions_of_all_buildings = []
        # get remaining actions
        for i in range(self.num_buildings):
            if i == self.active_building_ID:
                actions_of_all_buildings.append(action_of_active_building)
            else:
                action_of_other_building, _ = self.evaluation_model.predict(observations_of_all_buildings[i], deterministic=True)
                # self.evaluation_model.set_training_mode(False) ?
                actions_of_all_buildings.append(action_of_other_building)

        actions = modify_action(actions_of_all_buildings, self.metadata)

        # do step in whole environment
        next_obs, reward, done, info = self.env.step(actions)

        self.current_timestep += 1
        # return only the next observation and reward for one building
        modify_obs_for_active_building = modify_obs(next_obs, self.forecaster, self.metadata, self.current_timestep)[self.active_building_ID]
        reward_for_active_building = reward[self.active_building_ID]

        return modify_obs_for_active_building, reward_for_active_building, done, info

    def render(self):
        return self.env.render()
