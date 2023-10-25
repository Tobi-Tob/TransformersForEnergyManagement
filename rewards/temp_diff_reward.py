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

        comfort_cost = []
        for i in range(len(observations)):
            cost = -np.clip(temperature_diff[i] - 1, a_min=0, a_max=np.inf)
            comfort_cost.append(cost)

        return comfort_cost

    def reset(self):
        pass
