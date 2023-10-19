import numpy as np
import time
import os
from tqdm.auto import tqdm
import json

from citylearn.citylearn import CityLearnEnv

from agents.forecaster import SolarGenerationForecaster, TemperatureForecaster

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
    print("Starting local evaluation")

    env, env_data = create_citylearn_env(config)
    tau = 1  # 48
    print("Env Created")

    model = SubmissionModel(env_data=env_data, tau=tau)

    model_time_elapsed = 0

    num_steps = 0
    interrupted = False

    forecast_quailty_scores = []

    collect_data = False
    evaluate_forecaster = True
    # forecaster = SolarGenerationForecaster()
    forecaster = TemperatureForecaster()
    X = []
    y = []
    error = []

    try:
        observations = env.reset()
        for _ in tqdm(range(env.time_steps)):

            ### This is only a reference script provided to allow you
            ### to do local evaluation. The evaluator **DOES NOT**
            ### use this script for orchestrating the evaluations.

            if collect_data:
                obs_modified = forecaster.modify_obs(observations)[0]
                X.append(obs_modified)

            step_start = time.perf_counter()
            forecasts_dict = model.compute_forecast(observations)
            model_time_elapsed += time.perf_counter() - step_start

            # Perform logging.
            # ========================================================================
            ground_truth_vals = {
                **{b.name: {
                    'Equipment_Eletric_Power': b.energy_simulation.non_shiftable_load[env.time_step + 1:env.time_step + 1 + tau],
                    'DHW_Heating': b.energy_simulation.dhw_demand[env.time_step + 1:env.time_step + 1 + tau],
                    'Cooling_Load': b.energy_simulation.cooling_demand[env.time_step + 1:env.time_step + 1 + tau]
                } for b in env.buildings},
                'Solar_Generation': env.buildings[0].energy_simulation.solar_generation[env.time_step + 1:env.time_step + 1 + tau],
                'Carbon_Intensity': env.buildings[0].carbon_intensity.carbon_intensity[env.time_step + 1:env.time_step + 1 + tau]
            }
            gt_length = len(ground_truth_vals['Carbon_Intensity'])  # length of ground truth data

            forecast_scores = {
                **{b.name: {
                    load_type: rmse(np.array(forecasts_dict[b.name][load_type])[:gt_length], np.array(ground_truth_vals[b.name][load_type]))
                    for load_type in ['Equipment_Eletric_Power', 'DHW_Heating', 'Cooling_Load']}
                    for b in env.buildings},
                **{param: rmse(np.array(forecasts_dict[param])[:gt_length], np.array(ground_truth_vals[param]))
                   for param in ['Solar_Generation', 'Carbon_Intensity']}
            }
            forecast_quailty_scores.append(forecast_scores)  # store time step score for post-processing

            # Step environment.
            # ========================================================================
            actions = np.zeros((1, len(env.buildings) * 3))

            if evaluate_forecaster:
                prediction = forecaster.forecast(observations)

            observations, _, done, _ = env.step(actions)

            if evaluate_forecaster:
                target = observations[0][2]
                error.append(np.abs(prediction - target))

            if collect_data:
                next_value = observations[0][2]
                y.append(next_value)

            if done:
                break

            num_steps += 1

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True

    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {model_time_elapsed}s")

    # Compute forecast quality metrics.
    # ===================================================================================
    mean_param_values = {
        **{b.name: {
            'Equipment_Eletric_Power': np.mean(b.energy_simulation.non_shiftable_load),
            'DHW_Heating': np.mean(b.energy_simulation.dhw_demand),
            'Cooling_Load': np.mean(b.energy_simulation.cooling_demand)
        } for b in env.buildings},
        'Solar_Generation': np.mean(env.buildings[0].energy_simulation.solar_generation),
        'Carbon_Intensity': np.mean(env.buildings[0].carbon_intensity.carbon_intensity)
    }
    results = {
        **{b.name: {
            load_type: np.mean([d[b.name][load_type] for d in forecast_quailty_scores]) / mean_param_values[b.name][load_type]
            for load_type in ['Equipment_Eletric_Power', 'DHW_Heating', 'Cooling_Load']}
            for b in env.buildings},
        **{param: np.mean([d[param] for d in forecast_quailty_scores]) / mean_param_values[param]
           for param in ['Solar_Generation', 'Carbon_Intensity']}
    }

    all_preds = []
    for b in env.buildings:
        all_preds.append(results[b.name]['Equipment_Eletric_Power'])
        all_preds.append(results[b.name]['DHW_Heating'])
        all_preds.append(results[b.name]['Cooling_Load'])
    all_preds.append(results['Solar_Generation'])
    all_preds.append(results['Carbon_Intensity'])
    results['Mean_Forecast_NRMSE'] = np.mean(all_preds)

    results['Mean_Equipment_Eletric_Power'] = np.mean([results[b.name]['Equipment_Eletric_Power'] for b in env.buildings])
    results['Mean_DHW_Heating'] = np.mean([results[b.name]['DHW_Heating'] for b in env.buildings])
    results['Mean_Cooling_Load'] = np.mean([results[b.name]['Cooling_Load'] for b in env.buildings])

    print("=========================Forecast Quality Results=========================")
    print(json.dumps(results, indent=4))

    if collect_data:
        print(X)
        print(y)
        np.save("data/temperature_forecast/X", X)
        np.save("data/temperature_forecast/y", y)
    if evaluate_forecaster:
        print('mean:')
        print(np.mean(error))
        print('std:')
        print(np.std(error))
        print('max:')
        print(np.max(error))
        print('min:')
        print(np.min(error))


if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')


    config = Config()

    evaluate(config)
