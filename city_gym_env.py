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
    # District
    "day_type": 0,
    "hour": 1,
    "outdoor_dry_bulb_temperature": 2,  # current outdoor dry-bulb temperature (C)
    "outdoor_dry_bulb_temperature_predicted_6h": 3,
    "outdoor_dry_bulb_temperature_predicted_12h": 4,
    "outdoor_dry_bulb_temperature_predicted_24h": 5,
    "diffuse_solar_irradiance": 6,  # solar radiation (from different directions scattered by fog/clouds) (W/m2)
    "diffuse_solar_irradiance_predicted_6h": 7,
    "diffuse_solar_irradiance_predicted_12h": 8,
    "diffuse_solar_irradiance_predicted_24h": 9,
    "direct_solar_irradiance": 10,  # direct beam solar radiation (W/m2)
    "direct_solar_irradiance_predicted_6h": 11,
    "direct_solar_irradiance_predicted_12h": 12,
    "direct_solar_irradiance_predicted_24h": 13,
    "carbon_intensity": 14,  # current carbon intensity of the power grid (kg_CO2/kWh)

    # Building 1
    "indoor_dry_bulb_temperature": 15,  # current indoor temperature (C)
    "non_shiftable_load": 16,  # current electricity consumption of electrical devices (kWh)
    "solar_generation": 17,  # current solar generation (kWh)
    "dhw_storage_soc": 18,  # current hot water storage state of charge (kWh)
    "electrical_storage_soc": 19,  # current electrical storage state of charge (kWh)
    "net_electricity_consumption": 20,  # current buildings net electricity demand to the grid (kWh)

    # District
    "electricity_pricing": 21,  # current electricity pricing (USD/kWh)
    "electricity_pricing_predicted_6h": 22,
    "electricity_pricing_predicted_12h": 23,
    "electricity_pricing_predicted_24h": 24,

    # Building 1
    "cooling_demand": 25,  # current cooling demand (kWh)
    "dhw_demand": 26,  # current hot water demand (kWh)
    "occupant_count": 27,  # current number of occupants (people)
    "indoor_dry_bulb_temperature_set_point": 28,  # current temperature set point (C)
    "power_outage": 29,  # current power outage (0 or 1?)

    # Building 2
    "indoor_dry_bulb_temperature2": 30,
    "non_shiftable_load2": 31,
    "solar_generation2": 32,
    "dhw_storage_soc2": 33,
    "electrical_storage_soc2": 34,
    "net_electricity_consumption2": 35,
    "cooling_demand2": 36,
    "dhw_demand2": 37,
    "occupant_count2": 38,
    "indoor_dry_bulb_temperature_set_point2": 39,
    "power_outage2": 40,

    # Building 3
    "indoor_dry_bulb_temperature3": 41,
    "non_shiftable_load3": 42,
    "solar_generation3": 43,
    "dhw_storage_soc3": 44,
    "electrical_storage_soc3": 45,
    "net_electricity_consumption3": 46,
    "cooling_demand3": 47,
    "dhw_demand3": 48,
    "occupant_count3": 49,
    "indoor_dry_bulb_temperature_set_point3": 50,
    "power_outage3": 51,
}


def modify_obs(obs: List[List[float]]) -> List[float]:
    """
    Modify the observation space 2D -> 1D for compatibility with stable baselines
    """
    return [j for i in obs for j in i]


# Input:
# [['dhw_storage', 'electrical_storage', 'cooling_device',
# 'dhw_storage', 'electrical_storage', 'cooling_device',
# 'dhw_storage', 'electrical_storage', 'cooling_device']]
def modify_action(action: List[float]) -> List[List[float]]:
    """
    Modify the action space 1D -> 2D to meet the requirements of the citylearn central agent
    """
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
