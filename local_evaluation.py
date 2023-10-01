import numpy as np
import time
import os
import utils
from citylearn.citylearn import CityLearnEnv
from random import randint

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
    print("Starting local evaluation")

    env, wrapper_env = create_citylearn_env(config, SubmissionReward)
    print("Env Created")

    agent = SubmissionAgent(wrapper_env)
    agent.set_model_index(0)

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
    try:
        while True:

            observations, reward, done, _ = env.step(actions)

            J += reward[0]
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

                # Optional: Uncomment line below to update power outage random seed 
                # from what was initially defined in schema
                env = update_power_outage_random_seed(env, randint(0, 9999))

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

    print(f"Total agent time: {np.round(agent_time_elapsed, decimals=2)}s")
    utils.print_metrics(episode_metrics)

    # agent.print_normalizations()


if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1  # TODO epidose 2+ hat keine Stromausfälle?


    config_data = Config()

    evaluate(config_data)



















