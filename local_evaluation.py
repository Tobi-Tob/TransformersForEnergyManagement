import numpy as np
import time
import os

from citylearn.citylearn import CityLearnEnv

from city_gym_env import CityGymEnv

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
        while True:

            # This is only a reference script provided to allow you
            # to do local evaluation. The evaluator **DOES NOT**
            # use this script for orchestrating the evaluations.

            observations, _, done, _ = env.step(actions)
            if not done:
                step_start = time.perf_counter()
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                episodes_completed += 1
                metrics_df = env.evaluate_citylearn_challenge()
                episode_metrics.append(metrics_df)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics_df}", )

                observations = env.reset()

                step_start = time.perf_counter()
                actions = agent.predict(observations)
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
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")
    if len(episode_metrics) > 0:
        carbon_emission = np.mean([e['carbon_emissions_total']['value'] for e in episode_metrics])
        unmet_hours = np.mean([e['discomfort_proportion']['value'] for e in episode_metrics])
        ramping = np.mean([e['ramping_average']['value'] for e in episode_metrics])
        load_factor = np.mean([e['daily_one_minus_load_factor_average']['value'] for e in episode_metrics])
        daily_peak = np.mean([e['daily_peak_average']['value'] for e in episode_metrics])
        annual_peak = np.mean([e['annual_peak_average']['value'] for e in episode_metrics])
        average_score = np.mean([e['average_score']['value'] for e in episode_metrics])
        print(episode_metrics[0]['carbon_emissions_total']['display_name'], carbon_emission, episode_metrics[0]['carbon_emissions_total']['weight'])
        print(episode_metrics[0]['discomfort_proportion']['display_name'], unmet_hours, episode_metrics[0]['discomfort_proportion']['weight'])
        print(episode_metrics[0]['ramping_average']['display_name'], ramping, episode_metrics[0]['ramping_average']['weight'])
        print(episode_metrics[0]['daily_one_minus_load_factor_average']['display_name'], load_factor,
              episode_metrics[0]['daily_one_minus_load_factor_average']['weight'])
        print(episode_metrics[0]['daily_peak_average']['display_name'], daily_peak, episode_metrics[0]['daily_peak_average']['weight'])
        print(episode_metrics[0]['annual_peak_average']['display_name'], annual_peak, episode_metrics[0]['annual_peak_average']['weight'])
        print("==> Score:", average_score)


if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1

    config = Config()

    evaluate(config)
