from citylearn.reward_function import ComfortReward
from typing import List, Mapping, Union
import numpy as np


class MixReward(ComfortReward):
    """
    Reference: https://github.com/Roberock/citylearn2023_kit
    """

    def __init__(self, env_metadata):
        super().__init__(env_metadata)
        self.levels = []
        self.levels = np.zeros((3, 100))
        self.q_values = []
        self.net_el_con_old = np.array([0, 0, 0])
        self.net_el_unc_old = np.array([0, 0, 0])

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        rew_com = self._get_reward_comfort(observations)
        rew_cons = self._get_reward_eimssions(observations)
        rew_peak = self._get_reward_ramp(observations)
        weights_list = [.3, .3, .4]
        if self.central_agent:
            reward_list = rew_com + rew_cons + rew_peak
            reward = [sum([r * w for r, w in zip(reward_list, weights_list)])]
        else:
            reward = []
            w = weights_list
            for r0, r1, r2 in zip(rew_com, rew_cons, rew_peak):
                reward.append(w[0] * r0 + w[1] * r1 + w[2] * r2)
        return reward

    def _get_reward_comfort(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o in observations:
            heating_demand = o.get('heating_demand', 0.0)
            cooling_demand = o.get('cooling_demand', 0.0)
            heating = heating_demand > cooling_demand
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - self.band
            upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + self.band
            delta = abs(indoor_dry_bulb_temperature - set_point)

            if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                exponent = self.lower_exponent if heating else self.higher_exponent
                reward = -(delta ** exponent)

            elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                reward = 0.0 if heating else -delta

            elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                reward = -delta if heating else 0.0

            else:
                exponent = self.higher_exponent if heating else self.lower_exponent
                reward = -(delta ** exponent)

            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward

    def _get_reward_consumption(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        if self.central_agent:
            reward = [min(sum(net_electricity_consumption) * -1, 0.0)]
        else:
            reward = [min(v * -1, 0.0) for v in net_electricity_consumption]
        return reward

    def _get_reward_eimssions(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        emissions = [o['carbon_intensity'] * o['net_electricity_consumption'] for o in observations]

        if self.central_agent:
            reward = [min(sum(emissions) * -1, 0.0)]
        else:
            reward = [min(v * -1, 0.0) for v in emissions]
        return reward

    def _get_reward_ramp(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        net_el_con_con = [o['net_electricity_consumption'] for o in observations]
        ramp_controlled = abs(np.array(net_el_con_con) - self.net_el_con_old)
        ramp_controlled = ramp_controlled * -1
        self.net_el_con_old = np.array(net_el_con_con)
        if self.central_agent:
            reward = [min(ramp_controlled)]
        else:
            reward = [ramp_controlled]
        return reward
