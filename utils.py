from random import randint

import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json
from stable_baselines3.common.callbacks import BaseCallback

from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from rewards.user_reward import SubmissionReward

act_mapping = {
    "dhw_storage_action": [0, 3, 6],  # Action to control the hot water storage tank. Fraction of `dhw_storage` `capacity` to charge/discharge by.
    "electrical_storage_action": [1, 4, 7],  # Action to control the electrical storage. Fraction of `electrical_storage` `capacity`.
    "cooling_device_action": [2, 5, 8]  # Action to control the heat pump,
    # fraction of `cooling_device` `nominal_power` to make available for space cooling.
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
    #  = cooling_device/heating_device/dhw_device/non_shiftable_load_device/electrical_storage.electricity_consumption + solar_generation
    "solar_generation": [17, 32, 43],  # current solar generation (kWh)
    "dhw_storage_soc": [18, 33, 44],  # current hot water storage state of charge (%)
    "electrical_storage_soc": [19, 34, 45],  # current electrical storage state of charge (%)
    "net_electricity_consumption": [20, 35, 46],  # current buildings net electricity demand to the grid (kWh)
    "cooling_demand": [25, 36, 47],  # cooling demand (kWh) is a function of controller cooling_device_action as the controller
    # sets the input energy to the cooling device that outputs some cooling energy that is set as cooling demand.
    "dhw_demand": [26, 37, 48],  # current domestic hot water energy demand (kWh)
    "occupant_count": [27, 38, 49],  # current number of occupants (people)
    "indoor_dry_bulb_temperature_set_point": [28, 39, 50],  # current temperature set point (C)
    "power_outage": [29, 40, 51],  # current power outage (0 or 1)
}


def print_interactions(action, reward, next_observation):
    do_print = False
    if do_print:
        def get_act(str_act):
            data = [action[0][i] for i in act_mapping[str_act]]
            return data

        def get_obs(str_obs):
            data = [next_observation[0][i] for i in obs_mapping[str_obs]]
            return data

        # print(reward, "dhw_demand")
        if get_obs('power_outage')[0] == 1:
            # print(get_act("cooling_device_action"), "cooling_device_action")
            print(reward, "reward")
            print(get_obs('day_type'), get_obs('hour'), 'outage:', get_obs('power_outage'))
            print(get_obs('dhw_storage_soc'), get_obs('electrical_storage_soc'))


def print_metrics(episode_metrics):
    if len(episode_metrics) > 0:
        # print all episode_metrics values
        score = 0  # own score (special computation for nan values)
        weight_correction = 0
        for metric in episode_metrics[0].keys():
            display_name = episode_metrics[0][metric]['display_name']
            values = [e[metric]['value'] for e in episode_metrics]
            number_of_nan = sum(np.isnan(x) for x in values)
            if metric == "power_outage_normalized_unserved_energy_total":
                n_episodes_with_outage = len(values) - number_of_nan
            if number_of_nan == len(values):
                value = None
            else:
                value = np.nanmean(values)
            if metric is not "average_score":
                weight = episode_metrics[0][metric]['weight']
                if value is not None:
                    print(f"{str(weight):<6} {display_name:<18} {np.round(value, decimals=4)}")
                    score += weight * value
                else:
                    print(f"{str(weight):<6} {display_name:<18} None")
                    weight_correction += weight
        score = score / (1 - weight_correction)  # only has an effect if there was no power outage during the evaluation
        print('\033[92m' + f"{'====>':<6} {'Score:':<18} {score}")
        print('\033[0m' + f"Number of episodes with power outage: {n_episodes_with_outage} / {len(values)}")


def init_environment(buildings_to_use, simulation_start_end=None, reward_function=SubmissionReward, **kwargs) -> CityLearnEnv:
    r"""Initialize `CityLearnEnv` and returns the environment

        Parameters
        ----------
        buildings_to_use: list[int]
            List to define which buildings are used in the environment, example: [1,2,4,17].
        reward_function: RewardFunction
            Reward function to use
        simulation_start_end: list[int]
            List to define start and end time step, example: [0,8759].

        """
    schema_path = 'data/schemas/warm_up/schema.json'
    schema = read_json(schema_path)
    if simulation_start_end is not None:
        schema['simulation_start_time_step'] = simulation_start_end[0]
        schema['simulation_end_time_step'] = simulation_start_end[1]
    dict_buildings = schema['buildings']

    # set all buildings to include=false
    for building_schema in dict_buildings.values():
        building_schema['include'] = False

    # include=true for buildings to use
    for building_number in buildings_to_use:
        building_id = 'Building_' + str(building_number)
        if building_id in dict_buildings:
            dict_buildings[building_id]['include'] = True

    env = CityLearnEnv(schema, reward_function=reward_function)
    return env


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    Performs an evaluation on the validation environment logging: validation_scores, validation_reward,
    value_estimates, mean_dhw_storage_action, mean_electrical_storage_action, mean_cooling_device_action

    """

    def __init__(self, eval_interval, n_eval_episodes, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback (they are defined in the base class):

        # self.model type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env type: Union[gym.Env, VecEnv, None]

        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.eval_interval = eval_interval  # type: int
        self.n_eval_episodes = n_eval_episodes

        # The logger object, used to report things in the terminal
        # self.logger stable_baselines3.common.logger

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.eval_env = CityLearnEnv('./data/schemas/warm_up/schema.json', reward_function=SubmissionReward)
        self.n_buildings = len(self.eval_env.buildings)
        if type(self.model).__name__ is 'SAC':
            self.eval_agent = SACAgent(self.eval_env, mode='single', single_model=self.model)
        elif type(self.model).__name__ is 'PPO':
            self.eval_agent = PPOAgent(self.eval_env, mode='single', single_model=self.model)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.n_calls % self.eval_interval == 0 and self.n_calls > 719 * 3:  # call every n steps and perform evaluation
            seed = 73055  # seed 1
            for b in self.eval_env.buildings:
                b.stochastic_power_outage_model.random_seed = seed
            observations = self.eval_env.reset()
            actions = self.eval_agent.register_reset(observations)

            value_at_initial_state = self.eval_agent.predict_obs_value(observations)
            self.logger.record("train/value_estimate_t0", value_at_initial_state)

            J = 0
            t = 0
            action_sum = np.zeros(self.n_buildings * 3)
            episodes_completed = 0
            episode_metrics = []

            while True:  # run n episodes in eval_env with eval_agent
                observations, reward, done, _ = self.eval_env.step(actions)
                J += sum(reward) / self.n_buildings
                action_sum += np.abs(np.array(actions[0]))
                t += 1

                if t == 700:
                    value_estimate_t700 = self.eval_agent.predict_obs_value(observations)
                    self.logger.record("train/value_estimate_t700", value_estimate_t700)

                if not done:
                    actions = self.eval_agent.predict(observations)
                else:
                    episodes_completed += 1
                    metrics_df = self.eval_env.evaluate_citylearn_challenge()
                    episode_metrics.append(metrics_df)
                    print(f"Evaluation Episodes complete: {episodes_completed} | running J: {np.round(J / episodes_completed, decimals=2)} | "
                          f"{metrics_df}")

                    if episodes_completed == 1:
                        seed = 1  # seed 2
                    elif episodes_completed == 2:
                        seed = 41000  # seed 3
                    elif episodes_completed == 3:
                        seed = 13700  # seed 4
                    elif episodes_completed == 4:
                        seed = 1404  # seed 5
                    elif episodes_completed == 5:
                        seed = 5000  # seed 6
                    else:
                        seed = randint(0, 99999)  # seed >= 7 is random
                    for b in self.eval_env.buildings:
                        b.stochastic_power_outage_model.random_seed = seed

                    observations = self.eval_env.reset()
                    actions = self.eval_agent.register_reset(observations)

                if episodes_completed >= self.n_eval_episodes:
                    break

            carbon_emissions = np.nanmean([m['carbon_emissions_total']['value'] for m in episode_metrics])
            discomfort = np.nanmean([m['discomfort_proportion']['value'] for m in episode_metrics])
            ramping = np.nanmean([m['ramping_average']['value'] for m in episode_metrics])
            load_factor = np.nanmean([m['daily_one_minus_load_factor_average']['value'] for m in episode_metrics])
            daily_peak = np.nanmean([m['daily_peak_average']['value'] for m in episode_metrics])
            annual_peak = np.nanmean([m['annual_peak_average']['value'] for m in episode_metrics])
            thermal_resilience = np.nanmean([m['one_minus_thermal_resilience_proportion']['value'] for m in episode_metrics])
            unserved_energy = np.nanmean([m['power_outage_normalized_unserved_energy_total']['value'] for m in episode_metrics])
            eval_score = 0.1 * carbon_emissions + 0.3 * discomfort + 0.075 * ramping + 0.075 * load_factor + 0.075 * daily_peak + \
                         0.075 * annual_peak + 0.15 * thermal_resilience + 0.15 * unserved_energy

            mean_dhw_storage_action = (action_sum[0] + action_sum[3] + action_sum[6]) / (3 * self.eval_env.episode_time_steps)
            mean_electrical_storage_action = (action_sum[1] + action_sum[4] + action_sum[7]) / (3 * self.eval_env.episode_time_steps)
            mean_cooling_device_action = (action_sum[2] + action_sum[5] + action_sum[8]) / (3 * self.eval_env.episode_time_steps)

            self.logger.record("rollout/validation_score", eval_score)
            self.logger.record("rollout/validation_reward", J / self.n_eval_episodes)

            self.logger.record("metric/1_carbon_emissions", carbon_emissions)
            self.logger.record("metric/2_discomfort", discomfort)
            self.logger.record("metric/3_ramping", ramping)
            self.logger.record("metric/4_load_factor", load_factor)
            self.logger.record("metric/5_daily_peak", daily_peak)
            self.logger.record("metric/6_annual_peak", annual_peak)
            self.logger.record("metric/7_thermal_resilience", thermal_resilience)
            self.logger.record("metric/8_unserved_energy", unserved_energy)

            self.logger.record("train/mean_dhw_storage_action", mean_dhw_storage_action / self.n_eval_episodes)
            self.logger.record("train/mean_electrical_storage_action", mean_electrical_storage_action / self.n_eval_episodes)
            self.logger.record("train/mean_cooling_device_action", mean_cooling_device_action / self.n_eval_episodes)

            self.model.policy.set_training_mode(True)

        return True
