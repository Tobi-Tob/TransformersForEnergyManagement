import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json
from stable_baselines3.common.callbacks import BaseCallback

from agents.ppo_agent import PPOAgent
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
    do_print = True
    if do_print:
        def get_act(str_act):
            data = [action[0][i] for i in act_mapping[str_act]]
            return data

        def get_obs(str_obs):
            data = [next_observation[0][i] for i in obs_mapping[str_obs]]
            return data

        print(get_act("cooling_device_action"), "cooling_device_action")
        print(reward, "reward")
        print()
        print(get_obs('indoor_dry_bulb_temperature'), get_obs('indoor_dry_bulb_temperature_set_point'))


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


def init_environment(buildings_to_use, simulation_start_end=None, **kwargs) -> CityLearnEnv:
    r"""Initialize `CityLearnEnv` and returns the environment

        Parameters
        ----------
        buildings_to_use: list[int]
            List to define which buildings are used in the environment, example: [1,2,4,17].
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

    env = CityLearnEnv(schema, reward_function=SubmissionReward)
    return env


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    Performs an evaluation on the validation environment logging: validation_score, validation_reward,
    value_estimates, mean_dhw_storage_action, mean_electrical_storage_action, mean_cooling_device_action

    """

    def __init__(self, eval_interval, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback (they are defined in the base class):

        # The RL model
        # self.model = None  # type: BaseAlgorithm

        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.eval_interval = eval_interval  # type: int
        # self.num_timesteps = 0  # type: int

        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]

        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger

        # Sometimes, for event callback, it is useful to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.n_calls % self.eval_interval == 0:  # call every n steps and perform evaluation

            eval_env = CityLearnEnv('./data/schemas/warm_up/schema.json', reward_function=SubmissionReward)
            eval_agent = PPOAgent(eval_env, mode='single', single_model=self.model)

            observations = eval_env.reset()

            value_at_initial_state = eval_agent.predict_obs_value(observations)
            self.logger.record("train/value_estimate_t0", value_at_initial_state)

            actions = eval_agent.register_reset(observations)

            J = 0
            t = 0
            action_sum = np.zeros(len(eval_env.buildings) * 3)

            while True:  # run one episode in eval_env with eval_agent
                observations, reward, done, _ = eval_env.step(actions)
                J += reward[0]
                action_sum += np.abs(np.array(actions[0]))
                t += 1

                if t == 400:
                    value_estimate_t400 = eval_agent.predict_obs_value(observations)
                    self.logger.record("train/value_estimate_t400", value_estimate_t400)

                if not done:
                    actions = eval_agent.predict(observations)
                else:
                    metrics_df = eval_env.evaluate_citylearn_challenge()
                    break

            eval_score = metrics_df['average_score']['value']
            mean_dhw_storage_action = (action_sum[0] + action_sum[3] + action_sum[6]) / (3 * eval_env.episode_time_steps)
            mean_electrical_storage_action = (action_sum[1] + action_sum[4] + action_sum[7]) / (3 * eval_env.episode_time_steps)
            mean_cooling_device_action = (action_sum[2] + action_sum[5] + action_sum[8]) / (3 * eval_env.episode_time_steps)

            self.logger.record("rollout/validation_score", eval_score)
            self.logger.record("rollout/validation_reward", J)
            self.logger.record("train/mean_dhw_storage_action", mean_dhw_storage_action)
            self.logger.record("train/mean_electrical_storage_action", mean_electrical_storage_action)
            self.logger.record("train/mean_cooling_device_action", mean_cooling_device_action)

            self.model.policy.set_training_mode(True)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
