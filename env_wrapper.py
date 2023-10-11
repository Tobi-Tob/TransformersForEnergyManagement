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
    Input: (1,52), Output: (3, 17)
    Modify the observation space to:
    [['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'carbon_intensity', 'indoor_dry_bulb_temperature',
    'non_shiftable_load', 'solar_generation', 'dhw_storage_soc', 'electrical_storage_soc', 'net_electricity_consumption',
    'cooling_demand', 'dhw_demand', 'occupant_count', 'indoor_dry_bulb_temperature_set_point', 'power_outage',
    'indoor_temperature_difference', 'solar_generation_1h_predicted'],...]
    """
    #  --> Delete unimportant observations like pricing, 12 and 24 h predictions
    #  --> Add usefully observation e.g. temperature_diff
    #  x   Include building specific info e.g. battery storage limit (or pre-process observation with this information)
    #  x   Include info of other buildings e.g. mean storage level or mean net energy consumption
    #  --> Use historic weather forecast information
    #  --> Use building solar forecaster or building power forecaster
    #  --> Normalize the observation TODO use estimates in metadata

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
        b.append(temperature_diff)  # add temperature difference
        b.append(solar_generation_1h * pv_nominal_powers[i])  # add solar generation prediction
        assert len(b) == 13

    obs_modified = []
    normalizations = get_obs_normalization(metadata)
    for b in obs_single_building:
        obs = obs_district + b
        for i in range(len(obs)):
            obs[i] = (obs[i] - normalizations[i][0]) / normalizations[i][1]
        assert len(obs) == 4 + 13
        obs_modified.append(obs)

    return obs_modified


def modify_action(action: List[ndarray], metadata) -> List[List[float]]:
    """
    Input: (3,3), Output: (1, 9)
    """
    return [np.concatenate(action).tolist()]


def get_modified_observation_space():
    observation_dim = 17
    low_limit = np.zeros(observation_dim)
    high_limit = np.zeros(observation_dim)
    low_limit[0], high_limit[0] = 1, 7  # day_type
    low_limit[1], high_limit[1] = 1, 24  # hour
    low_limit[2], high_limit[2] = 21.37, 40.32  # outdoor_dry_bulb_temperature
    low_limit[3], high_limit[3] = 0.3375, 0.5561  # carbon_intensity
    low_limit[4], high_limit[4] = 9.99, 37.23  # indoor_dry_bulb_temperature
    low_limit[5], high_limit[5] = 0.1686, 8.8252  # non_shiftable_load
    low_limit[6], high_limit[6] = 0, 703.63  # solar_generation
    low_limit[7], high_limit[7] = 0, 1  # dhw_storage_soc
    low_limit[8], high_limit[8] = 0, 1  # electrical_storage_soc
    low_limit[9], high_limit[9] = -706.45, 19.75  # net_electricity_consumption
    low_limit[10], high_limit[10] = 0, 12.1999  # cooling_demand
    low_limit[11], high_limit[11] = 0, 6.5373  # dhw_demand
    low_limit[12], high_limit[12] = 0, 3  # occupant_count
    low_limit[13], high_limit[13] = 20, 27.23  # indoor_dry_bulb_temperature_set_point
    low_limit[14], high_limit[14] = 0, 1  # power_outage
    low_limit[15], high_limit[15] = -17.23, 17.23  # temperature_difference
    low_limit[16], high_limit[16] = 0, 703.63  # solar_generation_1h_predicted

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


def get_modified_action_space():
    action_dim = 3
    low_limit = np.zeros(action_dim)
    high_limit = np.zeros(action_dim)
    low_limit[0], high_limit[0] = -1, 1  # dhw_storage_action
    low_limit[1], high_limit[1] = -0.83, 0.83  # electrical_storage_action
    low_limit[2], high_limit[2] = 0, 1  # cooling_device_action

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


def get_obs_normalization(metadata):
    return np.array([
        #  -mean-      -std-
        [4.09861111, 1.97132168],  # day_type
        [12.4680556, 6.92211284],  # hour
        [29.5859027, 4.78622061],  # outdoor_dry_bulb_temperature
        [0.45429827, 0.04875349],  # carbon_intensity
        [24.2984569, 2.00000000],  # indoor_dry_bulb_temperature
        [0.00000000, 1.00000000],  # non_shiftable_load
        [0.00000000, 1.00000000],  # solar_generation
        [0.00000000, 1.00000000],  # dhw_storage_soc
        [0.00000000, 1.00000000],  # electrical_storage_soc
        [0.00000000, 1.00000000],  # net_electricity_consumption
        [0.00000000, 1.00000000],  # cooling_demand
        [0.00000000, 1.00000000],  # dhw_demand
        [0.00000000, 1.00000000],  # occupant_count
        [24.2984569, 2.00000000],  # indoor_dry_bulb_temperature_set_point
        [0.00000000, 1.00000000],  # power_outage
        [0.00000000, 2.00000000],  # temperature_difference
        [0.00000000, 1.00000000],  # solar_generation_1h_predicted
    ])


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
