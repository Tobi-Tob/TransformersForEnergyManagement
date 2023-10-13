from random import randint
from typing import List
import numpy as np
from citylearn.citylearn import CityLearnEnv
from numpy import ndarray
from gym import Env, spaces

from agents.forecaster import SolarGenerationForecaster

'''
File to modify the observation and action space and wrap the citylearn environment
'''


def modify_obs(obs: List[List[float]], forecaster: dict, metadata) -> List[List[float]]:
    """
    Input: (1,52), Output: (3, 16)
    Modify the observation space to:
    [['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'carbon_intensity', 'indoor_dry_bulb_temperature',
    'non_shiftable_load', 'solar_generation_1h_predicted', 'dhw_storage_soc', 'electrical_storage_soc',
    'net_electricity_consumption', 'cooling_demand', 'dhw_demand', 'occupant_count',
    'indoor_dry_bulb_temperature_set_point', 'power_outage', 'indoor_temperature_difference'],...]
    """
    #  --> Delete unimportant observations like pricing, 12 and 24 h predictions
    #  --> Add usefully observation e.g. temperature_diff TODO next step temperature_1h?
    #  x   Include building specific info e.g. battery storage limit (or pre-process observation with this information)
    #  x   Include info of other buildings e.g. mean storage level or mean net energy consumption
    #  --> Use historic weather forecast information
    #  --> Use building solar forecaster or building power forecaster
    #  --> Normalize the observation

    # Read metadata
    buildings = []
    pv_nominal_powers = []
    annual_cooling_demand_estimate = []
    annual_dhw_demand_estimate = []
    annual_non_shiftable_load_estimate = []
    annual_solar_generation_estimate = []
    for building_metadata in metadata:
        buildings.append(building_metadata['name'])
        pv_nominal_powers.append(building_metadata['pv']['nominal_power'])
        annual_cooling_demand_estimate.append(building_metadata['annual_cooling_demand_estimate'])
        annual_dhw_demand_estimate.append(building_metadata['annual_dhw_demand_estimate'])
        annual_non_shiftable_load_estimate.append(building_metadata['annual_non_shiftable_load_estimate'])
        annual_solar_generation_estimate.append(building_metadata['annual_solar_generation_estimate'])

    solar_generation_1h = forecaster['SolarGenerationForecaster'].predict_solar_generation(obs)

    obs = obs[0]
    del obs[21:25]  # remove electricity pricing and predictions 6h, 12h, 24h
    obs_district = [obs[0], obs[1], obs[2], obs[14]]  # all important district observations (4)
    obs_buildings = obs[15:]  # all remaining building observations (#buildings * 11)

    # building-level observations
    assert len(obs_buildings) % 11 == 0  # 11 observations per building
    obs_single_building = [obs_buildings[i:i + 11] for i in range(0, len(obs_buildings), 11)]
    assert len(obs_single_building) == len(buildings)
    for i, b in enumerate(obs_single_building):
        temperature_diff = b[0] - b[9]  # indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point
        b[2] = solar_generation_1h * pv_nominal_powers[i]  # replace with solar generation prediction
        b.append(temperature_diff)  # add temperature difference
        assert len(b) == 12

    obs_modified = []
    normalizations = get_obs_normalization(metadata)
    for i, b in enumerate(obs_single_building):
        obs = obs_district + b
        for j in range(len(obs)):
            if isinstance(normalizations[j][0], list):
                obs[j] = (obs[j] - normalizations[j][0][i]) / normalizations[j][1][i]
            else:
                obs[j] = (obs[j] - normalizations[j][0]) / normalizations[j][1]
        assert len(obs) == 4 + 12
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


def get_modified_observation_space():
    observation_dim = 16
    low_limit = np.zeros(observation_dim)
    high_limit = np.zeros(observation_dim)
    low_limit[0], high_limit[0] = -1.58, 1.48  # day_type
    low_limit[1], high_limit[1] = -1.66, 1.67  # hour
    low_limit[2], high_limit[2] = -0.73, 4.01  # outdoor_dry_bulb_temperature
    low_limit[3], high_limit[3] = -2.40, 2.09  # carbon_intensity
    low_limit[4], high_limit[4] = -5, 5  # indoor_dry_bulb_temperature
    low_limit[5], high_limit[5] = -1, 20  # non_shiftable_load
    low_limit[6], high_limit[6] = -1, 3  # solar_generation_1h_predicted
    low_limit[7], high_limit[7] = 0, 1  # dhw_storage_soc
    low_limit[8], high_limit[8] = 0, 1  # electrical_storage_soc
    low_limit[9], high_limit[9] = -5, 10  # net_electricity_consumption
    low_limit[10], high_limit[10] = -1, 100  # cooling_demand
    low_limit[11], high_limit[11] = -1, 100  # dhw_demand
    low_limit[12], high_limit[12] = 0, 3  # occupant_count
    low_limit[13], high_limit[13] = -1.08, 0.74  # indoor_dry_bulb_temperature_set_point
    low_limit[14], high_limit[14] = 0, 1  # power_outage
    low_limit[15], high_limit[15] = -5, 5  # temperature_difference

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


def get_modified_action_space():
    action_dim = 3
    low_limit = np.zeros(action_dim)
    high_limit = np.zeros(action_dim)
    low_limit[0], high_limit[0] = -2.85, 2.85  # dhw_storage_action (max of all buildings)
    low_limit[1], high_limit[1] = -4, 4  # electrical_storage_action (max of all buildings)
    low_limit[2], high_limit[2] = 0, 4.11  # cooling_device_action (max of all buildings)

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


def get_obs_normalization(metadata):
    cooling_demand_estimate = []
    dhw_demand_estimate = []
    non_shiftable_load_estimate = []
    solar_generation_estimate = []
    net_e_consumption_estimate = []
    for bm in metadata:
        cooling_demand = bm['annual_cooling_demand_estimate']/bm['simulation_time_steps']
        cooling_demand_estimate.append(cooling_demand)
        # [2400.07568359375, 1256.208740234375, 1516.25146484375]
        dhw_demand = bm['annual_dhw_demand_estimate']/bm['simulation_time_steps']
        dhw_demand_estimate.append(dhw_demand)
        # [153.8460235595703, 45.04438781738281, 109.74966430664062]
        non_shiftable_load = bm['annual_non_shiftable_load_estimate']/bm['simulation_time_steps']
        non_shiftable_load_estimate.append(non_shiftable_load)
        # [450.445068359375, 323.14483642578125, 631.7621459960938]
        solar_generation = bm['annual_solar_generation_estimate']/bm['simulation_time_steps']
        solar_generation_estimate.append(solar_generation)
        # [345.7142639160156, 172.8571319580078, 345.7142639160156]
        net_e_consumption_estimate.append(cooling_demand + dhw_demand + non_shiftable_load - solar_generation)

    return [
        #   -mean-      -std-
        [4.09861111, 1.97132168],  # day_type
        [12.4680556, 6.92211284],  # hour
        [24.2984569, 4.00000000],  # outdoor_dry_bulb_temperature
        [0.45429827, 0.04875349],  # carbon_intensity
        [24.2984569, 4.00000000],  # indoor_dry_bulb_temperature
        [non_shiftable_load_estimate, non_shiftable_load_estimate],  # non_shiftable_load (high max 19)
        [solar_generation_estimate, solar_generation_estimate],      # solar_generation_1h_predicted
        [0.00000000, 1.00000000],  # dhw_storage_soc  TODO: normalisieren * capacity dafür zusatz feature batterie stand 0-1 hinzufuegen
        [0.00000000, 1.00000000],  # electrical_storage_soc
        [net_e_consumption_estimate, net_e_consumption_estimate],    # net_electricity_consumption
        [cooling_demand_estimate, cooling_demand_estimate],          # cooling_demand
        [dhw_demand_estimate, dhw_demand_estimate],                  # dhw_demand (high std 2.9 and max 52)
        [0.00000000, 1.00000000],  # occupant_count
        [24.2984569, 4.00000000],  # indoor_dry_bulb_temperature_set_point
        [0.00000000, 1.00000000],  # power_outage
        [0.00000000, 4.00000000],  # temperature_difference_to_set_point
    ]


class CityEnvForTraining(Env):
    # EnvWrapper Class used for training, controlling one building interactions
    def __init__(self, env: CityLearnEnv):
        self.env = env
        self.metadata = env.get_metadata()['buildings']
        self.evaluation_model = None

        SGF = SolarGenerationForecaster()
        self.forecaster = {
            type(SGF).__name__: SGF
        }

        self.num_buildings = len(env.buildings)
        self.active_building_ID = 0
        self.action_space = get_modified_action_space()
        self.observation_space = get_modified_observation_space()
        self.time_steps = env.time_steps

    def set_evaluation_model(self, model):
        """
        The evaluation model is used for generating actions for all buildings that are not actively included in the training process at the moment
        """
        self.evaluation_model = model

    def reset(self, **kwargs) -> List[float]:
        for forecaster in self.forecaster.values():
            forecaster.reset()
        self.active_building_ID += 1
        if self.active_building_ID >= self.num_buildings:
            self.active_building_ID = 0

        random_seed = randint(0, 99999)
        for b in self.env.buildings:
            b.stochastic_power_outage_model.random_seed = random_seed

        obs = modify_obs(self.env.reset(), self.forecaster, self.metadata)

        # if self.evaluation_model is not None:
        #     test_obs = [1.4717988035316487, 0.36577623892071665, 0.4605091146284094, -0.6613356282905993, 1.6623099623810385, -0.8392258613092881, 0.8102464956499016, -0.8746554498111345, 1.3965325787525062, 1.8631778991821282, 0.44612357020378113, 1.3038498163223267, 1.0, 0.19288472831249237, -0.555773913860321, 0.0, 0.16414415836334229, 0.5525509585834176, 0.04521620637206958, 0.0, 1.8179616928100586]
        #     test_action, _ = self.evaluation_model.predict(test_obs, deterministic=True)
        #     print(test_action)

        return obs[self.active_building_ID]

    def step(self, action_of_active_building: List[float]):
        # only return visible state of the active building
        observations_of_all_buildings = modify_obs(self.env.observations, self.forecaster, self.metadata)
        actions_of_all_buildings = []
        for i in range(self.num_buildings):
            if i == self.active_building_ID:
                actions_of_all_buildings.append(action_of_active_building)
            else:
                action_of_other_building, _ = self.evaluation_model.predict(observations_of_all_buildings[i], deterministic=True)
                actions_of_all_buildings.append(action_of_other_building)

        actions = modify_action(actions_of_all_buildings, self.metadata)

        next_obs, reward, done, info = self.env.step(actions)

        return modify_obs(next_obs, self.forecaster, self.metadata)[self.active_building_ID], sum(reward), done, info

    def render(self):
        return self.env.render()
