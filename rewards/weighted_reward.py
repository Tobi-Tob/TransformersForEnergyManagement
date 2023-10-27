import numpy as np
from citylearn.reward_function import RewardFunction


class WeightedRewardFunction(RewardFunction):
    """ Simple passthrough example of comfort reward from Citylearn env """
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
        self.district_electricity_consumption_history = []
        self.max_district_electricity_consumption = 0
        self.simulation_time_steps = None
        self.current_time_step = 1
    
    def calculate(self, observations):
        if not self.central_agent:
            raise NotImplementedError("WeightedRewardFunction only supports central agent")
        if self.simulation_time_steps is None:
            self.simulation_time_steps = self.env_metadata['simulation_time_steps']
        if self.current_time_step >= self.simulation_time_steps:
            self.reset()
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

        # Carbon emissions 0.60, 0.25, 0.63, 0.30, 0.45, 0.62, 0.62, 0.36, 0.30, 0.22
        carbon_emissions_cost = 0.435 * min(sum(net_electricity_consumption * carbon_intensity) * -1, 0.0)

        # Discomfort 1.17, 1.17, 1.16, 1.17, 1.16, 1.18, 1.17, 1.17
        # Thermal resilience 73 49 49 49 49 49 49
        # (why not compare to temperature set point of previous timestep?)
        discomfort_cost = 0
        thermal_resilience_cost = 0
        for i in range(num_buildings):
            discomfort_cost -= 1.17 * 1/num_buildings if occupant_count[i] > 0.5 and temperature_diff[i] > 1 else 0
            thermal_resilience_cost -= 49 * 1/num_buildings if occupant_count[i] > 0.5 and temperature_diff[i] > 1 and power_outage[i] > 0.5 else 0

        # Ramping 0.75 0.48 0.55 0.65 0.71 0.56 0.76 0.75 0.74 0.75 0.49 0.75 0.75 0.53 0.51 0.54 0.74 0.76 0.76 0.73
        try:
            ramping_cost = 0.66 * -np.abs(district_electricity_consumption - self.district_electricity_consumption_history[-2])
        except IndexError:
            ramping_cost = 0

        # Load factor (0.84) 1.70 1.58 1.55 1.63 1.59 1.58 1.53 1.70 1.67
        # (ratio of running average consumption to peak consumption in the last 24 hours)
        load_factor_cost = 1.614 * (np.mean(self.district_electricity_consumption_history)/np.max(self.district_electricity_consumption_history) - 1)

        # Daily peak 0.11 0.108 0.108 0.110 0.066 0.97 0.109 0.107... 0.106 0.107
        daily_peak_cost = 0.102 * -np.max(self.district_electricity_consumption_history)

        # All-time peak 36.47 33.42 36.57 33.11 26.57 36.28 36.35 36.40
        if district_electricity_consumption > self.max_district_electricity_consumption:
            all_time_peak_cost = 34.4 * (self.max_district_electricity_consumption - district_electricity_consumption)
            self.max_district_electricity_consumption = district_electricity_consumption  # keep the max
        else:
            all_time_peak_cost = 0

        self.current_time_step += 1
        return [0.3*discomfort_cost + 0.1*carbon_emissions_cost
                + 0.075*(ramping_cost + load_factor_cost + daily_peak_cost + all_time_peak_cost) + 0.3*thermal_resilience_cost]

    def reset(self):
        """
        Resets reward function
        """
        self.district_electricity_consumption_history = []
        self.max_district_electricity_consumption = 0
        self.simulation_time_steps = None
        self.current_time_step = 1
