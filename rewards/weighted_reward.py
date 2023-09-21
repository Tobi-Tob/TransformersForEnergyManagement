from citylearn.reward_function import RewardFunction


class WeightedRewardFunction(RewardFunction):
    """ Simple passthrough example of comfort reward from Citylearn env """
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
    
    def calculate(self, observations):
        if not self.central_agent:
            raise NotImplementedError("WeightedRewardFunction only supports central agent")
        net_electricity_consumption_list = [o['net_electricity_consumption'] for o in observations]
        carbon_intensity = [o['carbon_intensity'] for o in observations][0]
        carbon_emissions_cost = min(sum(net_electricity_consumption_list) * -1, 0.0) * carbon_intensity
        return [carbon_emissions_cost]
