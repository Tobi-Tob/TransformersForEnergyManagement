import numpy as np
import time
import os
from tqdm.auto import tqdm
import json

from citylearn.citylearn import CityLearnEnv

from agents.forecaster import SolarGenerationForecaster, TemperatureForecaster, NoneShiftableLoadForecaster
from agents.zero_agent import ZeroAgent

"""
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from my_models.user_model import SubmissionModel


def rmse(prediction, actual):
    """Root Mean Squared error metric."""
    return np.sqrt(np.mean(np.power(prediction - actual, 2)))


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


def create_citylearn_env(config):
    env = CityLearnEnv(config.SCHEMA)

    env_data = dict(
        observation_names=env.observation_names,
        action_names=env.action_names,
        observation_space=env.observation_space,
        action_space=env.action_space,
        time_steps=env.time_steps,
        buildings_metadata=env.get_metadata()['buildings'],
        num_buildings=len(env.buildings),
        building_names=[b.name for b in env.buildings],
        b0_pv_capacity=env.buildings[0].pv.nominal_power,
    )

    # Turn off actions for all buildings and do not simulate power outage (forecasting only).
    for b in env.buildings:
        b.ignore_dynamics = True
        b.simulate_power_outage = False

    return env, env_data


def evaluate(config):
    print("Starting forecaster evaluation/data collection")
    collect_data = False
    evaluate_forecaster = True

    env, env_data = create_citylearn_env(config)
    model = ZeroAgent(env)

    # Init forecaster:

    # forecaster = SolarGenerationForecaster()
    # forecaster = TemperatureForecaster()

    non_shiftable_load_estimate = []
    for bm in env.get_metadata()['buildings']:
        non_shiftable_load = bm['annual_non_shiftable_load_estimate'] / bm['simulation_time_steps']
        non_shiftable_load_estimate.append(non_shiftable_load)

    forecaster = NoneShiftableLoadForecaster(non_shiftable_load_estimate)

    X = []
    y = []
    error_b1 = []
    error_b2 = []
    error_b3 = []

    observations = env.reset()
    for _ in tqdm(range(env.time_steps)):

        if collect_data:
            obs_modified_b1 = forecaster.modify_obs(observations)[0]
            obs_modified_b2 = forecaster.modify_obs(observations)[1]
            obs_modified_b3 = forecaster.modify_obs(observations)[2]
            X.append(obs_modified_b1)
            X.append(obs_modified_b2)
            X.append(obs_modified_b3)

        if evaluate_forecaster:
            predictions = forecaster.forecast(observations)

        # step in environment
        actions = model.predict(observations)
        observations, _, done, _ = env.step(actions)

        if evaluate_forecaster:
            # target_temperature = observations[0][2]
            target_load_b1 = observations[0][16]
            target_load_b2 = observations[0][31]
            target_load_b3 = observations[0][42]
            error_b1.append(np.abs(predictions[0] - target_load_b1))
            error_b2.append(np.abs(predictions[1] - target_load_b2))
            error_b3.append(np.abs(predictions[2] - target_load_b3))
            print('target', target_load_b1, 'prediction', predictions[0])
            print('target', target_load_b2, 'prediction', predictions[1])
            print('target', target_load_b3, 'prediction', predictions[2])

        if collect_data:
            next_value_b1 = observations[0][16] / non_shiftable_load_estimate[0]
            next_value_b2 = observations[0][31] / non_shiftable_load_estimate[1]
            next_value_b3 = observations[0][42] / non_shiftable_load_estimate[2]
            y.append(next_value_b1)
            y.append(next_value_b2)
            y.append(next_value_b3)

        if done:
            break

    if collect_data:
        print('Data collected:')
        print('X:', X)
        print('len(X):', len(X))
        print('y:', y)
        print('len(y):', len(y))
        np.save("data/load_forecast/X", X)
        np.save("data/load_forecast/y", y)
    if evaluate_forecaster:
        print('Forecaster evaluated:')
        print('mean:')
        print(np.mean(error_b1 + error_b2 + error_b3))
        print('std:')
        print(np.std(error_b1 + error_b2 + error_b3))
        print('max:')
        print(np.max(error_b1 + error_b2 + error_b3))
        print('min:')
        print(np.min(error_b1 + error_b2 + error_b3))


if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')


    config = Config()

    evaluate(config)
