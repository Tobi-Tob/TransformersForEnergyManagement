from typing import List
import numpy as np
from numpy import ndarray
import gym
from citylearn.citylearn import CityLearnEnv

'''
File to modify the observation and action space and wrap the citylearn environment
'''


def modify_obs(obs: List[List[float]]) -> List[float]:
    """
    Modify the observation space 2D -> 1D for compatibility with stable baselines
    """
    return obs[0]


def modify_action(action: List[float]) -> List[List[float]]:
    """
    Modify the action space 1D -> 2D to meet the requirements of the citylearn central agent
    """
    return [action]


def modify_obs2(obs: List[List[float]]) -> List[List[float]]:
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


def modify_action2(action: List[ndarray]) -> List[List[float]]:
    """
    Input: (3,3), Output: (1, 9)
    """
    return [np.concatenate(action).tolist()]


# environment wrapper for stable baselines
class CityGymEnv(gym.Env):

    def __init__(self, env: CityLearnEnv):
        self.env = env

        self.num_buildings = len(env.action_space)
        self.action_space = env.action_space[0]  # only in single agent setting
        self.observation_space = env.observation_space[0]  # only in single agent setting
        self.time_steps = env.time_steps
        # TO THINK : normalize or modify the observation space

    def reset(self, **kwargs) -> List[float]:
        obs = self.env.reset()
        return modify_obs(obs)

    def step(self, action: List[float]):
        # If `central_agent` is True, `actions` parameter should be a list of 1 list containing all buildings' actions and follows the ordering of
        # buildings in `buildings`.
        act = modify_action(action)
        obs, reward, done, info = self.env.step(act)
        return modify_obs(obs), sum(reward), done, info

    def render(self):
        return self.env.render()


class CityEnvForTraining(gym.Env):
    # EnvWrapper Class used for training, controlling one building interactions, not used yet
    def __init__(self, env: CityLearnEnv):
        self.env = env

        self.num_buildings = len(env.action_space)
        self.action_space = env.action_space[0]  # TODO modify
        self.observation_space = env.observation_space[0]  # TODO modify
        self.time_steps = env.time_steps

    def reset(self, **kwargs) -> List[float]:
        obs = self.env.reset()
        return modify_obs(obs)

    def step(self, action: List[float]):
        # only return visible state for one building
        # calc reward based on only one building?
        act = modify_action(action)
        obs, reward, done, info = self.env.step(act)
        return modify_obs(obs), sum(reward), done, info

    def render(self):
        return self.env.render()
