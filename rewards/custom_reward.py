import numpy as np
from citylearn.reward_function import RewardFunction


class TempDiffReward(RewardFunction):
    def __init__(self, env_metadata):
        super().__init__(env_metadata)

    def calculate(self, observations):
        if not self.central_agent:
            raise NotImplementedError("RewardFunction only supports central agent")

        indoor_dry_bulb_temperature = np.array([o['indoor_dry_bulb_temperature'] for o in observations])
        indoor_dry_bulb_temperature_set_point = np.array([o['indoor_dry_bulb_temperature_set_point'] for o in observations])
        temperature_diff = np.abs(indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point)
        power_outage = np.array([o['power_outage'] for o in observations])

        cost = []
        for i in range(len(observations)):
            unmet_hours_cost = -np.clip(temperature_diff[i] - 1, a_min=0, a_max=np.inf)
            # unmet_cost = -temperature_diff[i] # linear also promising
            # thermal_resilience_cost = -temperature_diff[i]+1 if power_outage[i] == 1 else 0  # does not benefit thermal resilience

            cost.append(unmet_hours_cost)

        return cost

    def reset(self):
        pass

    class TempDiffReward(RewardFunction):
        def __init__(self, env_metadata):
            super().__init__(env_metadata)

        def calculate(self, observations):
            if not self.central_agent:
                raise NotImplementedError("RewardFunction only supports central agent")

            indoor_dry_bulb_temperature = np.array([o['indoor_dry_bulb_temperature'] for o in observations])
            indoor_dry_bulb_temperature_set_point = np.array([o['indoor_dry_bulb_temperature_set_point'] for o in observations])
            temperature_diff = np.abs(indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point)
            power_outage = np.array([o['power_outage'] for o in observations])

            cost = []
            for i in range(len(observations)):
                unmet_hours_cost = -np.clip(temperature_diff[i] - 1, a_min=0, a_max=np.inf)
                # unmet_cost = -temperature_diff[i] # linear also promising
                # thermal_resilience_cost = -temperature_diff[i]+1 if power_outage[i] == 1 else 0  # does not benefit thermal resilience

                cost.append(unmet_hours_cost)

            return cost

        def reset(self):
            pass


class UnservedEnergyReward(RewardFunction):
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
        self.current_time_step = 1
        self.simulation_time_steps = None
        self.previous_electrical_storage = None
        self.previous_dhw_storage = None

    def calculate(self, observations):
        """
        Funktion 1
        """
        if self.simulation_time_steps is None:
            self.simulation_time_steps = self.env_metadata['simulation_time_steps']
        if self.current_time_step >= self.simulation_time_steps:
            self.reset()
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
                # TODO vllt ohne clip
                expected_energy = cooling_demand[i] + dhw_demand[i] + non_shiftable_load[i]
                served_energy = energy_from_electrical_storage + energy_from_dhw_storage + solar_generation[i]  # only correct for outages
                # TODO vllt reward signal von solar_generation loesen
                unserved_energy_cost = - np.clip(expected_energy - served_energy, a_min=0, a_max=np.inf)  # TODO /expected_energy, vllt ohne clip?

            reward.append(unserved_energy_cost)

        self.previous_electrical_storage = electrical_storage
        self.previous_dhw_storage = dhw_storage

        return reward

    def reset(self):
        """
        Resets reward function
        """
        self.current_time_step = 1
        self.simulation_time_steps = None
        self.previous_electrical_storage = None
        self.previous_dhw_storage = None
