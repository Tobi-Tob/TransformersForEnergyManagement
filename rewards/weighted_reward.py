import numpy as np
from citylearn.reward_function import RewardFunction


class WeightedRewardFunction(RewardFunction):
    """ Simple passthrough example of comfort reward from Citylearn env """
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
        self.district_electricity_consumption_history = []
    
    def calculate(self, observations):
        if not self.central_agent:
            raise NotImplementedError("WeightedRewardFunction only supports central agent")
        net_electricity_consumption = np.array([o['net_electricity_consumption'] for o in observations])
        district_electricity_consumption = np.sum(net_electricity_consumption)
        num_buildings = len(net_electricity_consumption)
        self.district_electricity_consumption_history.append(district_electricity_consumption)
        self.district_electricity_consumption_history = self.district_electricity_consumption_history[-24:]  # keep last 24 hours

        carbon_intensity = np.array([o['carbon_intensity'] for o in observations])
        indoor_dry_bulb_temperature = np.array([o['indoor_dry_bulb_temperature'] for o in observations])
        indoor_dry_bulb_temperature_set_point = np.array([o['indoor_dry_bulb_temperature_set_point'] for o in observations])
        temperature_diff = np.abs(indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point)
        occupant_count = np.array([o['occupant_count'] for o in observations])
        power_outage = np.array([o['power_outage'] for o in observations])

        # Carbon emissions
        carbon_emissions_cost = 0.60 * min(sum(net_electricity_consumption * carbon_intensity) * -1, 0.0)

        # Discomfort
        # Thermal resilience (why not compare to temperature set point of previous timestep?)
        discomfort_cost = 0
        thermal_resilience_cost = 0
        for i in range(num_buildings):
            discomfort_cost -= 1.17 * 1/num_buildings if occupant_count[i] > 0.5 and temperature_diff[i] > 1 else 0
            thermal_resilience_cost -= 73 * 1/num_buildings if occupant_count[i] > 0.5 and temperature_diff[i] > 1 and power_outage[i] > 0.5 else 0

        # Ramping
        try:
            ramping_cost = 0.75 * -np.abs(district_electricity_consumption - self.district_electricity_consumption_history[-2])
        except IndexError:
            ramping_cost = 0

        # Load factor (ratio of running average consumption to peak consumption in the last 24 hours)
        load_factor_cost = 0.84 * (np.mean(self.district_electricity_consumption_history)/np.max(self.district_electricity_consumption_history) - 1)

        # Daily peak
        daily_peak_cost = 0.11 * -np.max(self.district_electricity_consumption_history)

        # All-time peak (approximation)
        all_time_peak_cost =\
            1.7 * -np.clip(np.max(self.district_electricity_consumption_history[:22]) - district_electricity_consumption, float('-inf'), 0)**2

        # TODO Unserved energy (proportion of unmet demand due to supply shortage)

        return [0.3*discomfort_cost + 0.1*carbon_emissions_cost + 0.3*(ramping_cost + load_factor_cost
                + daily_peak_cost + all_time_peak_cost) + 0.3*thermal_resilience_cost]
