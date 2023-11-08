import numpy as np
import pandas as pd
import time
import os
from citylearn.citylearn import CityLearnEnv

"""
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.user_agent import SubmissionAgent
from rewards.user_reward import SubmissionReward

class WrapperEnv:
    """
    Env to wrap provide Citylearn Env data without providing full env
    Preventing attribute access outside of the available functions
    """
    def __init__(self, env_data):
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.time_steps = env_data['time_steps']
        self.seconds_per_time_step = env_data['seconds_per_time_step']
        self.random_seed = env_data['random_seed']
        self.buildings_metadata = env_data['buildings_metadata']
        self.episode_tracker = env_data['episode_tracker']
    
    def get_metadata(self):
        return {'buildings': self.buildings_metadata}

def create_citylearn_env(config, reward_function):
    env = CityLearnEnv(config.SCHEMA, reward_function=reward_function)

    env_data = dict(
        observation_names = env.observation_names,
        action_names = env.action_names,
        observation_space = env.observation_space,
        action_space = env.action_space,
        time_steps = env.time_steps,
        random_seed = None,
        episode_tracker = None,
        seconds_per_time_step = None,
        buildings_metadata = env.get_metadata()['buildings']
    )

    wrapper_env = WrapperEnv(env_data)
    return env, wrapper_env

def update_power_outage_random_seed(env: CityLearnEnv, random_seed: int) -> CityLearnEnv:
    """Update random seed used in generating power outage signals.
    
    Used to optionally update random seed for stochastic power outage model in all buildings.
    Random seeds should be updated before calling :py:meth:`citylearn.citylearn.CityLearnEnv.reset`.
    """

    for b in env.buildings:
        b.stochastic_power_outage_model.random_seed = random_seed

    return env

def evaluate(config):
    print("Starting local evaluation")
    
    env, wrapper_env = create_citylearn_env(config, SubmissionReward)
    print("Env Created")

    agent = SubmissionAgent(wrapper_env)

    observations = env.reset()

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(observations)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        env_metadata = env.get_metadata()
        previous_dhw_storage_soc_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'dhw_storage_soc']
        previous_electrical_storage_soc_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'electrical_storage_soc']
        check_data = []
        
        while True:
            
            ### This is only a reference script provided to allow you 
            ### to do local evaluation. The evaluator **DOES NOT** 
            ### use this script for orchestrating the evaluations. 

            observations, _, done, _ = env.step(actions)
            net_electricity_consumption_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'net_electricity_consumption']
            non_shiftable_load_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'non_shiftable_load']
            cooling_demand_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'cooling_demand']
            dhw_demand_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'dhw_demand']
            cooling_device_cop_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'cooling_device_cop']
            solar_generation_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'solar_generation']
            dhw_storage_soc_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'dhw_storage_soc']
            electrical_storage_soc_observations = [observations[0][i] for i in range(len(observations[0])) if env.observation_names[0][i] == 'electrical_storage_soc']
            print(env.time_step)
            
            for i, b in enumerate(env.buildings):
                # cooling
                cooling_electricity_consumption = cooling_demand_observations[i]/cooling_device_cop_observations[i]

                # dhw
                ## dhw demand
                dhw_demand_electricity_consumption = dhw_demand_observations[i]/env_metadata['buildings'][i]['dhw_device']['efficiency']

                ## dhw storage
                dhw_soc_init = previous_dhw_storage_soc_observations[i]*(1 - env_metadata['buildings'][i]['dhw_storage']['loss_coefficient'])
                dhw_soc_init = max(0.0, dhw_soc_init)
                dhw_storage_energy_balance = (dhw_storage_soc_observations[i] - dhw_soc_init)*env_metadata['buildings'][i]['dhw_storage']['capacity']
                previous_dhw_storage_soc_observations[i] = dhw_storage_soc_observations[i]

                if dhw_storage_energy_balance >= 0:
                    dhw_storage_energy_balance /= env_metadata['buildings'][i]['dhw_storage']['round_trip_efficiency']
                else:
                    dhw_storage_energy_balance *= env_metadata['buildings'][i]['dhw_storage']['round_trip_efficiency']

                dhw_storage_electricity_consumption = dhw_storage_energy_balance/env_metadata['buildings'][i]['dhw_device']['efficiency']
                dhw_electricity_consumption = dhw_demand_electricity_consumption + dhw_storage_electricity_consumption

                # electrical storage
                electrical_soc_init = previous_electrical_storage_soc_observations[i]*(1 - env_metadata['buildings'][i]['electrical_storage']['loss_coefficient'])
                electrical_soc_init = max(0.0, electrical_soc_init)
                electrical_storage_energy_balance = (electrical_storage_soc_observations[i] - electrical_soc_init)*env_metadata['buildings'][i]['electrical_storage']['capacity']
                previous_electrical_storage_soc_observations[i] = electrical_storage_soc_observations[i]

                # the round trip efficiency for electrical storage is an estimate 
                # as what is stored in env.metadata is the round trip efficiency as at the time of environment reset.
                # efficiency in the battery model is a function of nominal power and charged/discharged energy.
                # see: 
                # https://www.citylearn.net/api/citylearn.energy_model.html#citylearn.energy_model.Battery.get_current_efficiency and how it is used in
                # https://www.citylearn.net/api/citylearn.energy_model.html#citylearn.energy_model.Battery.charge
                if electrical_storage_energy_balance >= 0:
                    electrical_storage_energy_balance /= env_metadata['buildings'][i]['electrical_storage']['round_trip_efficiency']
                else:
                    electrical_storage_energy_balance *= env_metadata['buildings'][i]['electrical_storage']['round_trip_efficiency']

                calculated_electrical_storage_electricity_consumption = electrical_storage_energy_balance
                internal_electrical_storage_electricity_consumption = b.electrical_storage.energy_balance[b.time_step]
                
                # net electricity consumption
                calculated_net_electricity_consumption = cooling_electricity_consumption\
                    + dhw_electricity_consumption\
                        + calculated_electrical_storage_electricity_consumption\
                            + non_shiftable_load_observations[i]\
                                - solar_generation_observations[i]
                
                internal_net_electricity_consumption = cooling_electricity_consumption\
                    + dhw_electricity_consumption\
                        + internal_electrical_storage_electricity_consumption\
                            + non_shiftable_load_observations[i]\
                                - solar_generation_observations[i]
                
                check_data += [
                    (b.name, b.time_step, b.power_outage, net_electricity_consumption_observations[i], calculated_net_electricity_consumption, internal_net_electricity_consumption),
                ]
            
            if not done:
                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                episodes_completed += 1
                metrics_df = env.evaluate_citylearn_challenge()
                episode_metrics.append(metrics_df)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics_df}", )
                
                # Optional: Uncomment line below to update power outage random seed 
                # from what was initially defined in schema
                env = update_power_outage_random_seed(env, 90000)

                observations = env.reset()

                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= config.num_episodes:
                break

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")

    check_data = pd.DataFrame(check_data, columns=['building', 'time_step', 'power_outage', 'observation', 'calculated_with_calculated_electrical_storage_consumption', 'calculated_with_internal_electrical_storage_consumption'])
    check_data['observation_and_calculated_with_calculated_electrical_storage_consumption_are_equal'] = check_data.apply(lambda x: abs(x['observation'] - x['calculated_with_calculated_electrical_storage_consumption']) < 0.00001, axis=1)
    check_data['observation_and_calculated_with_internal_electrical_storage_consumption_are_equal'] = check_data.apply(lambda x: abs(x['observation'] - x['calculated_with_internal_electrical_storage_consumption']) < 0.00001, axis=1)
    check_data.to_csv('electricity_consumption_check.csv', index=False)

if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1
    
    config = Config()

    evaluate(config)
