from typing import List
import gym
from citylearn.citylearn import CityLearnEnv

'''
File to modify the observation and action space and wrap the citylearn environment
'''
act_mapping = {
    "dhw_storage_action": 0,  # action to control the hot water storage tank
    "electrical_storage_action": 1,  # action to control the electrical storage
    "cooling_device_action": 2  # action to control the heat pump
}

obs_mapping = {
    "day_type": 0,
    "hour": 1,
    "outdoor_temperature": 2,  # current outdoor dry-bulb temperature (C)
    "outdoor_temperature_predicted_6h": 3,
    "outdoor_temperature_predicted_12h": 4,
    "outdoor_temperature_predicted_24h": 5,
    "diffuse_solar_radiation": 6,  # solar radiation (from different directions scattered by fog/clouds) (W/m2)
    "diffuse_solar_radiation_predicted_6h": 7,
    "diffuse_solar_radiation_predicted_12h": 8,
    "diffuse_solar_radiation_predicted_24h": 9,
    "direct_solar_radiation": 10,  # direct beam solar radiation (W/m2)
    "direct_solar_radiation_predicted_6h": 11,
    "direct_solar_radiation_predicted_12h": 12,
    "direct_solar_radiation_predicted_24h": 13,
    "carbon_intensity": 14,  # current carbon intensity of the power grid (kg_CO2/kWh)

    "indoor_temperature": 16,  # current indoor temperature (C)
    "equipment_electric_power": 17,  # current electricity consumption of electrical devices (kWh)

    "electricity_pricing": 22,  # current electricity pricing (USD/kWh)
    "electricity_pricing_predicted_6h": 23,
    "electricity_pricing_predicted_12h": 24,
    "electricity_pricing_predicted_24h": 25,
    "cooling_load": 26,  # current cooling load (kWh)
    "dhw_heating": 27,  # current hot water load (kWh)
    "occupant_count": 28,  # current number of occupants (people)
    "temperature_set_point": 29,  # current temperature set point (C)

    # Building 2
    "indoor_temperature_2": 30,
    "equipment_electric_power_2": 31,

    "cooling_load_2": 36,
    "dhw_heating_2": 37,
    "occupant_count_2": 38,
    "temperature_set_point_2": 39,

    # Building 3
    "indoor_temperature_3": 40,
    "equipment_electric_power_3": 41,

    "cooling_load_3": 46,
    "dhw_heating_3": 47,
    "occupant_count_3": 48,
    "temperature_set_point_3": 49,
}


def modify_obs(obs: List[List[float]]) -> List[float]:
    return [j for i in obs for j in i]


def modify_action(action: List[float]) -> List[List[float]]:
    return [action]


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
