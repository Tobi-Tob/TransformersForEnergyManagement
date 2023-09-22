import numpy as np

act_mapping = {
    "dhw_storage_action": [0, 3, 6],  # action to control the hot water storage tank
    "electrical_storage_action": [1, 4, 7],  # action to control the electrical storage
    "cooling_device_action": [2, 5, 8]  # action to control the heat pump
}

obs_mapping = {
    # District
    "day_type": [0],
    "hour": [1],
    "outdoor_dry_bulb_temperature": [2],  # current outdoor dry-bulb temperature (C)
    "outdoor_dry_bulb_temperature_predicted_6h": [3],
    "outdoor_dry_bulb_temperature_predicted_12h": [4],
    "outdoor_dry_bulb_temperature_predicted_24h": [5],
    "diffuse_solar_irradiance": [6],  # solar radiation (from different directions scattered by fog/clouds) (W/m2)
    "diffuse_solar_irradiance_predicted_6h": [7],
    "diffuse_solar_irradiance_predicted_12h": [8],
    "diffuse_solar_irradiance_predicted_24h": [9],
    "direct_solar_irradiance": [10],  # direct beam solar radiation (W/m2)
    "direct_solar_irradiance_predicted_6h": [11],
    "direct_solar_irradiance_predicted_12h": [12],
    "direct_solar_irradiance_predicted_24h": [13],
    "carbon_intensity": [14],  # current carbon intensity of the power grid (kg_CO2/kWh)
    "electricity_pricing": [21],  # current electricity pricing (USD/kWh)
    "electricity_pricing_predicted_6h": [22],
    "electricity_pricing_predicted_12h": [23],
    "electricity_pricing_predicted_24h": [24],

    # Building 1-3
    "indoor_dry_bulb_temperature": [15, 30, 41],  # current indoor temperature (C)
    "non_shiftable_load": [16, 31, 42],  # current electricity consumption of electrical devices (kWh)
    "solar_generation": [17, 32, 43],  # current solar generation (kWh)
    "dhw_storage_soc": [18, 33, 44],  # current hot water storage state of charge (%)
    "electrical_storage_soc": [19, 34, 45],  # current electrical storage state of charge (%)
    "net_electricity_consumption": [20, 35, 46],  # current buildings net electricity demand to the grid (kWh)
    "cooling_demand": [25, 36, 47],  # current cooling energy demand (kWh)
    "dhw_demand": [26, 37, 48],  # current domestic hot water energy demand (kWh)
    "occupant_count": [27, 38, 49],  # current number of occupants (people)
    "indoor_dry_bulb_temperature_set_point": [28, 39, 50],  # current temperature set point (C)
    "power_outage": [29, 40, 51],  # current power outage (0 or 1)
}


def print_interactions(action, reward, next_observation):
    if False:
        def get_act(str_act):
            data = [action[0][i] for i in act_mapping[str_act]]
            return data

        def get_obs(str_obs):
            data = [next_observation[0][i] for i in obs_mapping[str_obs]]
            return data

        # print(get_act("electrical_storage_action"), "electrical_storage_action")
        print(reward, "reward")
        # print()
        # print(get_obs("hour"), "hour")
        print(get_obs("indoor_dry_bulb_temperature"), get_obs("indoor_dry_bulb_temperature_set_point"), get_obs("occupant_count"))


def print_metrics(episode_metrics):
    if len(episode_metrics) > 0:
        # print all episode_metrics values
        for metric in episode_metrics[0].keys():
            display_name = episode_metrics[0][metric]['display_name']
            value = np.mean([e[metric]['value'] for e in episode_metrics])
            if metric == "average_score":
                print('\033[92m' + f"{'====>':<6} {'Score:':<18} {value}")
            else:
                weight = str(episode_metrics[0][metric]['weight']) + "x"
                print(f"{weight:<6} {display_name:<18} {np.round(value, decimals=4)}")
