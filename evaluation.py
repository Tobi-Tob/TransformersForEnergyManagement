from datetime import datetime

import datasets
import numpy as np
import time
import os
from random import randint

import pandas as pd

import utils
from citylearn.citylearn import CityLearnEnv

from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.zero_agent import ZeroAgent
from rewards.user_reward import SubmissionReward
"""
This is an edited version of local_evaluation.py provided by the challenge. 
"""


class WrapperEnv:
    """
    Env to wrap provide Citylearn Env data without providing full env
    Preventing attribute access outside the available functions
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
        observation_names=env.observation_names,
        action_names=env.action_names,
        observation_space=env.observation_space,
        action_space=env.action_space,
        time_steps=env.time_steps,
        random_seed=None,
        episode_tracker=None,
        seconds_per_time_step=None,
        buildings_metadata=env.get_metadata()['buildings']
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
    print("========================= Starting Evaluation =========================")
    generate_data = True
    if generate_data:
        print('Collecting Data...')

    env, wrapper_env = create_citylearn_env(config, SubmissionReward)

    # model = SAC.load("my_models/SAC_test\m0_1438_steps.zip")
    # agent = SACAgent(wrapper_env, mode='single', single_model=model, save_observations=False)
    agent = SACAgent(wrapper_env, save_observations=True if generate_data else False)

    agent.set_model_index(0)

    env = update_power_outage_random_seed(env, randint(0, 99999))
    observations = env.reset()

    agent_time_elapsed = 0
    step_start = time.perf_counter()
    actions = agent.register_reset(observations)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    J = 0
    action_sum = np.zeros(len(env.buildings) * 3)

    dataset = []
    observation_data = []
    next_observation_data = []
    action_data = []
    reward_data = []
    done_data = []
    try:
        while True:
            if generate_data:
                observation_data.append(observations)

            observations, reward, done, _ = env.step(actions)

            if generate_data:
                reward_data.append(reward)
                done_data.append(done)

            J += sum(reward)
            action_sum += np.abs(np.array(actions[0]))
            utils.print_interactions(actions, reward, observations)

            if not done:
                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                episodes_completed += 1
                metrics_df = env.evaluate_citylearn_challenge()
                episode_metrics.append(metrics_df)
                print(f"Episode complete: {episodes_completed} | Reward: {np.round(J, decimals=2)} "
                      f"| Average Action: {np.round(action_sum / env.episode_time_steps, decimals=4)}")
                print(f"Latest episode metrics: {metrics_df}")
                J = 0
                action_sum = np.zeros(len(env.buildings) * 3)

                if generate_data:
                    observation_data, action_data = agent.get_obs_and_action_data()
                    for b in range(len(env.buildings)):
                        building_obs_data = [np.array(all_buildings_obs[b]) for all_buildings_obs in observation_data]
                        building_next_obs_data = building_obs_data[1:]
                        building_next_obs_data.append(building_obs_data[-1])
                        building_obs_data = np.array(building_obs_data)
                        building_next_obs_data = np.array(building_next_obs_data)
                        building_act_data = np.array([all_buildings_act[b] for all_buildings_act in action_data])
                        building_rew_data = np.array([all_buildings_rew[b] for all_buildings_rew in reward_data])
                        dict_building_i = {
                            "observations": building_obs_data,
                            "next_observations": building_next_obs_data,
                            "actions": building_act_data,
                            "rewards": building_rew_data,
                            "dones": np.array(done_data)
                        }
                        dataset.append(dict_building_i)

                env = update_power_outage_random_seed(env, randint(0, 99999))
                observations = env.reset()

                step_start = time.perf_counter()
                actions = agent.register_reset(observations)
                agent_time_elapsed += time.perf_counter() - step_start

            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= config.num_episodes:
                break

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True

    if not interrupted:
        dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        print(f"======================= Completed: {dt_string} =======================")

    print(agent.model_info, SubmissionReward.__name__)
    print(f"Total agent time: {np.round(agent_time_elapsed, decimals=2)}s")
    utils.print_metrics(episode_metrics)

    agent.print_normalizations()

    if generate_data:
        print("Amount Of Sequences: ", len(dataset))
        total_values = (2 * len(dataset[0]['observations'][0]) + len(dataset[0]['actions'][0]) + 2) * len(dataset[0]['actions']) * len(dataset)
        print("Total values to store: ", total_values) # 123120

        file_name = 'test'
        file_info = f"_{len(dataset)}"
        file_extension = ".pkl"
        file_path = "./data/DT_data/" + file_name + file_info + file_extension

        # create or overwrite pickle file
        # with open(file_path, "wb") as f:
        #    pickle.dump(dataset, f)

        # ds = datasets.Dataset.from_dict(dataset)

        #datasets.Dataset.from_pandas(pd.DataFrame(data=dataset)).save_to_disk(file_path)

        datasets.Dataset.from_dict({k: [s[k] for s in dataset] for k in dataset[0].keys()}).save_to_disk(file_path)

        print("========================= Writing Completed ============================")
        print("==> Data saved in", file_path, utils.get_string_file_size(file_path))

        # utils.check_data_structure(file_path)


if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1

        # Power outage probability:
        # p(outage|day) = 0.393% (modified to 1.97%)
        # p(outage>=1|month) = 11.15% (modified to 44.90%)
        # To have at least one outage in the evaluation with 95% probability: episodes >= 26 (modified to >=6)


    config_data = Config()

    evaluate(config_data)



















