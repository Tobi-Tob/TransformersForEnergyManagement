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
