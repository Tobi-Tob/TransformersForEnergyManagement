import numpy as np
from citylearn.reward_function import RewardFunction


class CombinedReward(RewardFunction):
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
        self.current_time_step = 1
        self.simulation_time_steps = None
        self.previous_electrical_storage = None
        self.previous_dhw_storage = None
        self.electricity_consumption_history = []
        self.max_electricity_consumption = 0

    def calculate(self, observations):
        """
        v.1 temp_diff_reward + 0.1 * emission_reward
        2_discomfort: 0.1, 1_carbon_emission: 1.0
        v.2 temp_diff_reward + 0.3 * emission_reward
        2_discomfort: 0.1, 1_carbon_emission: 1.0
        v.3 temp_diff_reward + emission_reward
        2_discomfort: 0.2, 1_carbon_emission: 0.95

        v.4 temp_diff_reward + grid_reward
        2_discomfort: 0.15, 1_carbon_emission: 0.95, 3_ramping: 1.0 4_load: 0.98, 56_peak: 0.9 0.85
        v.5 temp_diff_reward + 0.5 * grid_reward
        2_discomfort: 0.12, 1_carbon_emission: 0.95, 3_ramping: 1.1 4_load: 0.98, 56_peak: 0.92 0.9
        v.6 temp_diff_reward + 0.5 * grid_reward
        2_discomfort: 0.25, 1_carbon_emission: 0.95, 3_ramping: 1.0 4_load: 0.98, 56_peak: 0.87 0.85
        v.7 temp_diff_reward + grid_reward
        2_discomfort: 0.18, 1_carbon_emission: 0.95, 3_ramping: 1.2 4_load: 0.98, 56_peak: 0.90 0.80 good 8_unserved 0.37
        """
        if not self.central_agent:
            raise NotImplementedError("RewardFunction only supports central agent")
        if self.simulation_time_steps is None:
            self.simulation_time_steps = self.env_metadata['simulation_time_steps']
        if self.current_time_step >= self.simulation_time_steps:
            self.reset()

        temp_diff_reward = np.array(self._get_temp_diff_reward(observations))
        unserved_energy_reward = np.array(self._get_unserved_energy_reward(observations))
        emission_reward = np.array(self._get_emission_reward(observations))
        grid_reward = np.array(self._get_grid_reward(observations))

        return temp_diff_reward + grid_reward

    def _get_temp_diff_reward(self, observations):
        """
        Best 2_discomfort: 0.1, 8_unserved_energy: 0.5, 7_thermal_resilience: 0.45 (all in combination with unserved_energy_reward)
        """
        indoor_dry_bulb_temperature = np.array([o['indoor_dry_bulb_temperature'] for o in observations])
        indoor_dry_bulb_temperature_set_point = np.array([o['indoor_dry_bulb_temperature_set_point'] for o in observations])
        temperature_diff = np.abs(indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point)
        power_outage = np.array([o['power_outage'] for o in observations])

        reward = []
        for i in range(len(observations)):
            unmet_hours_cost = -np.clip(temperature_diff[i] - 1, a_min=0, a_max=np.inf)
            # unmet_cost = -temperature_diff[i] # linear also promising
            # thermal_resilience_cost = -temperature_diff[i]+1 if power_outage[i] == 1 else 0  # does not benefit thermal resilience

            reward.append(unmet_hours_cost)

        return reward

    def _get_unserved_energy_reward(self, observations):
        """
        Funktion 4
        Seems not to be relevant
        Best 8_unserved_energy: 0.65 (worse than with emission_reward)
        """
        if self.previous_electrical_storage is None:
            self.previous_electrical_storage = [o['electrical_storage_soc'] for o in observations]
        if self.previous_dhw_storage is None:
            self.previous_dhw_storage = [o['dhw_storage_soc'] for o in observations]

        electrical_storage = [o['electrical_storage_soc'] for o in observations]
        dhw_storage = [o['dhw_storage_soc'] for o in observations]
        num_buildings = len(electrical_storage)
        cooling_demand = [o['cooling_demand'] for o in observations]
        dhw_demand = [o['dhw_demand'] for o in observations]
        non_shiftable_load = [o['non_shiftable_load'] for o in observations]
        solar_generation = [o['solar_generation'] for o in observations]
        power_outage = [o['power_outage'] for o in observations]

        reward = []

        for i in range(num_buildings):
            unserved_energy_cost = 0
            if power_outage[i] > 0:
                building_metadata = self.env_metadata['buildings'][i]
                ec = building_metadata['electrical_storage']['capacity']
                dc = building_metadata['dhw_storage']['capacity']
                e_efficiency = building_metadata['electrical_storage']['efficiency']
                e_loss_coef = building_metadata['electrical_storage']['loss_coefficient']
                dhw_loss_coef = building_metadata['dhw_storage']['loss_coefficient']

                energy_from_electrical_storage = np.round(np.clip((1 - e_loss_coef) * self.previous_electrical_storage[i] - electrical_storage[i],
                                                                  a_min=0, a_max=np.inf) * ec * e_efficiency, decimals=4)
                energy_from_dhw_storage = np.round(np.clip((1 - dhw_loss_coef) * self.previous_dhw_storage[i] - dhw_storage[i],
                                                           a_min=0, a_max=np.inf) * dc, decimals=4)

                expected_energy = cooling_demand[i] + dhw_demand[i] + non_shiftable_load[i]
                served_energy = energy_from_electrical_storage + energy_from_dhw_storage + solar_generation[i]  # info vllt als feature?

                unserved_energy_cost = - np.clip(expected_energy - served_energy, a_min=0, a_max=np.inf) / expected_energy

            reward.append(unserved_energy_cost)

        self.previous_electrical_storage = electrical_storage
        self.previous_dhw_storage = dhw_storage
        return reward

    def _get_emission_reward(self, observations):
        """
        Version 6 clip, single, shared obs
        Best 1_carbon_emission: 0.55, best 8_unserved energy: 0.45 (high variance)
        """
        emissions = [o['carbon_intensity'] * o['net_electricity_consumption'] for o in observations]
        reward = []
        for i in range(len(observations)):
            emission_cost = -np.clip(emissions[i], a_min=0, a_max=np.inf)
            reward.append(emission_cost)
        return reward

    def _get_grid_reward(self, observations):
        """
        ramping_cost v.2 single --- 3_ramping: 0.8 (starts at 0.8 and did not learn further)

        peak_cost v.1 citylearn MARL --- 3_ramping up to 1.2, 4_load_factor up to 1, 5_daily_peak down to 0.85, 6_annual_peak down to 0.8

        load_factor_cost v.1 --- 4_load_factor constant 0.7
        """
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        district_electricity_consumption = sum(net_electricity_consumption)
        self.electricity_consumption_history.append(net_electricity_consumption)
        self.electricity_consumption_history = self.electricity_consumption_history[-24:]  # keep last 24 hours

        ramping_cost = []
        for i in range(len(observations)):
            try:
                ramping = -np.abs(net_electricity_consumption[i] - self.electricity_consumption_history[-2][i])
            except IndexError:
                ramping = 0
            ramping_cost.append(ramping)
        ramping_cost = np.array(ramping_cost)

        building_electricity_consumption = np.array(net_electricity_consumption, dtype=float) * -1
        peak_cost = np.sign(building_electricity_consumption) * 0.01 * building_electricity_consumption ** 2 * np.nanmax(
            [0, district_electricity_consumption])

        load_factor_cost = np.mean(self.electricity_consumption_history, axis=0) / np.max(self.electricity_consumption_history, axis=0) - 1
        # breaks markov property, bad?

        return peak_cost

    def reset(self):
        self.current_time_step = 1
        self.simulation_time_steps = None
        self.previous_electrical_storage = None
        self.previous_dhw_storage = None
        self.electricity_consumption_history = []
        self.max_electricity_consumption = 0
