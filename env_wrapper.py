from typing import List
import numpy as np
from numpy import ndarray
from gym import Env, spaces

'''
File to modify the observation and action space and wrap the citylearn environment
'''


def modify_obs(obs: List[List[float]]) -> List[List[float]]:
    """
    Input: (1,52), Output: (3, 21)
    Modify the observation space to:
    [['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
    'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h', 'direct_solar_irradiance',
    'direct_solar_irradiance_predicted_6h', 'carbon_intensity', 'indoor_dry_bulb_temperature', 'non_shiftable_load',
    'solar_generation','dhw_storage_soc','electrical_storage_soc','net_electricity_consumption', 'cooling_demand',
    'dhw_demand', 'occupant_count', 'indoor_dry_bulb_temperature_set_point', 'power_outage', 'temperature_diff'],...]
    """
    #  --> Delete unimportant observations like pricing, 12 and 24 h predictions
    #  --> Add usefully observation e.g. temperature_diff
    #  x   Include building specific info e.g. battery storage limit
    #  x   Include info of other buildings e.g. mean storage level or mean net energy consumption
    #  x   Use historic weather forecast information in observation
    #  x   Use building solar forecaster or building power forecaster (siehe Challenge 22 Platz 2, siehe Platz 3 SolarModul/DemandModul)
    #  x   Normalize the observation

    obs = obs[0]
    del obs[21:25]  # remove electricity pricing and predictions 6h, 12h, 24h
    obs_district = obs[:15]  # all remaining district observations (15)
    obs_buildings = obs[15:]  # all remaining building observations (#buildings * 11)

    # district observations
    del obs_district[12:14]  # remove direct solar irradiance prediction 12h, 24h
    del obs_district[8:10]  # remove diffuse solar irradiance prediction 12h, 24h
    del obs_district[4:6]  # remove outdoor dry bulb temperature prediction 12h, 24h
    assert len(obs_district) == 9

    # building-level observations
    assert len(obs_buildings) % 11 == 0  # 11 observations per building
    obs_single_building = [obs_buildings[i:i + 11] for i in range(0, len(obs_buildings), 11)]
    # add temperature difference: indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point
    for b in obs_single_building:
        temperature_diff = b[0] - b[9]  # indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point
        b.append(temperature_diff)
        assert len(b) == 12

    obs_modified = []
    for i in range(len(obs_single_building)):
        obs_modified.append(obs_district + obs_single_building[i])

    return obs_modified


def modify_action(action: List[ndarray]) -> List[List[float]]:
    """
    Input: (3,3), Output: (1, 9)
    """
    return [np.concatenate(action).tolist()]


def get_modified_observation_space():
    observation_dim = 21
    low_limit = np.zeros(observation_dim)
    high_limit = np.zeros(observation_dim)
    low_limit[0], high_limit[0] = 1, 7  # day_type
    low_limit[1], high_limit[1] = 1, 24  # hour
    low_limit[2], high_limit[2] = 21.37, 40.32  # outdoor_dry_bulb_temperature
    low_limit[3], high_limit[3] = 21.38, 40.45  # outdoor_dry_bulb_temperature_predicted_6h
    low_limit[4], high_limit[4] = 0, 466.61  # diffuse_solar_irradiance
    low_limit[5], high_limit[5] = 0, 461.26  # diffuse_solar_irradiance_predicted_6h
    low_limit[6], high_limit[6] = 0, 908.49  # direct_solar_irradiance
    low_limit[7], high_limit[7] = 0, 1056.34  # direct_solar_irradiance_predicted_6h
    low_limit[8], high_limit[8] = 0.3375, 0.5561  # carbon_intensity
    low_limit[9], high_limit[9] = 9.99, 37.23  # indoor_dry_bulb_temperature
    low_limit[10], high_limit[10] = 0.1686, 8.8252  # non_shiftable_load
    low_limit[11], high_limit[11] = 0, 703.63  # solar_generation
    low_limit[12], high_limit[12] = 0, 1  # dhw_storage_soc
    low_limit[13], high_limit[13] = 0, 1  # electrical_storage_soc
    low_limit[14], high_limit[14] = -706.45, 19.75  # net_electricity_consumption
    low_limit[15], high_limit[15] = 0, 12.1999  # cooling_demand
    low_limit[16], high_limit[16] = 0, 6.5373  # dhw_demand
    low_limit[17], high_limit[17] = 0, 3  # occupant_count
    low_limit[18], high_limit[18] = 20, 27.23  # indoor_dry_bulb_temperature_set_point
    low_limit[19], high_limit[19] = 0, 1  # power_outage
    low_limit[20], high_limit[20] = -17.23, 17.23  # temperature_difference

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


def get_modified_action_space():
    action_dim = 3
    low_limit = np.zeros(action_dim)
    high_limit = np.zeros(action_dim)
    low_limit[0], high_limit[0] = -1, 1  # dhw_storage_action
    low_limit[1], high_limit[1] = -0.83, 0.83  # electrical_storage_action
    low_limit[2], high_limit[2] = 0, 1  # cooling_device_action

    return spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)


class CityEnvForTraining(Env):
    # EnvWrapper Class used for training, controlling one building interactions
    def __init__(self, env):
        self.city_env = env
        self.agent = None

        self.num_buildings = len(env.buildings)
        self.active_building_ID = 0
        self.action_space = get_modified_action_space()
        self.observation_space = get_modified_observation_space()
        self.time_steps = env.time_steps

    def set_agent(self, agent):
        self.agent = agent

    def reset(self, **kwargs) -> List[float]:
        self.active_building_ID += 1
        if self.active_building_ID >= self.num_buildings:
            self.active_building_ID = 0

        obs = modify_obs(self.city_env.reset())
        return obs[self.active_building_ID]

    def step(self, action_of_active_building: List[float]):
        # only return visible state of the active building
        observations_of_all_buildings = modify_obs(self.city_env.observations)
        actions_of_all_buildings = []
        for i in range(self.num_buildings):
            if i == self.active_building_ID:
                actions_of_all_buildings.append(action_of_active_building)
            else:
                action_of_other_building, _ = self.agent.predict(observations_of_all_buildings[i])
                actions_of_all_buildings.append(action_of_other_building)

        actions = modify_action(actions_of_all_buildings)

        next_obs, reward, done, info = self.city_env.step(actions)

        return modify_obs(next_obs)[self.active_building_ID], sum(reward), done, info

    def render(self):
        return self.city_env.render()
