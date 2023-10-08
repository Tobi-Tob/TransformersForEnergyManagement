"""
Implementation of example prediction method.
Note, this forecasting method is illustratively *terrible*, but
provides a reference on how to format your forecasting method.

The forecasting/prediction model is implemented as a class.

This class must have the following methods:
    - __init__(self, ...), which initialises the Predictor object and
        performs any initial setup you might want to do.
    - compute_forecast(observation), which executes your prediction method,
        creating timeseries forecasts for [building electrical loads,
        normalise solar pv generation powers, grid carbon intensity]
        given the current observation.

You may wish to implement additional methods to make your model code neater.
"""

import numpy as np

from my_models.base_predictor_model import BasePredictorModel


class ExamplePredictor(BasePredictorModel):

    def __init__(self, env_data, tau):
        """Initialise Prediction object and perform setup.

        Args:
            env_data : Dictionary containing data about the environment
                    observation_names = env.observation_names,
                    observation_space = env.observation_space,
                    action_space = env.action_space,
                    time_steps = env.time_steps,
                    buildings_metadata = env.get_metadata()['buildings'],
                    num_buildings = len(env.buildings),
                    building_names = [b.name for b in env.buildings],
                    b0_pv_capacity = env.buildings[0].pv.nominal_power,

            tau (int): length of planning horizon (number of time instances
                into the future to forecast).
        """

        # Check local evaluation
        self.num_buildings = env_data['num_buildings']
        self.building_names = env_data['building_names']
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.tau = tau

        # Load in pre-computed prediction model.
        self.load()
        # ====================================================================
        # insert your loading code here
        # ====================================================================

        # Create buffer/tracking attributes
        self.prev_observations = None
        self.buffer = {'key': []}
        # ====================================================================
        print("=========================Available Observations=========================")
        print(self.observation_names)  # available observations

        # dummy forecaster buffer - delete for your implementation
        # ====================================================================
        self.prev_vals = {
            **{b_name: {
                'Equipment_Eletric_Power': None,
                'DHW_Heating': None,
                'Cooling_Load': None
            } for b_name in self.building_names},
            'Solar_Generation': None,
            'Carbon_Intensity': None
        }
        self.b0_pv_capacity = env_data['b0_pv_capacity']
        # ====================================================================

    def load(self):
        """No loading required for trivial example model."""
        pass

    def compute_forecast(self, observations):
        """Compute forecasts for each variable given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation. The structure of this list can
                be viewed via CityLearnEnv.observation_names.

        Returns:
            predictions_dict (dict): dictionary containing forecasts for each
                variable. Format is as follows:
                {
                    'Building_1': { # this is env.buildings[0].name
                        'Equipment_Eletric_Power': [ list of 48 floats - predicted equipment electric power for Building_1 ],
                        'DHW_Heating': [ list of 48 floats - predicted DHW heating for Building_1 ],
                        'Cooling_Load': [ list of 48 floats - predicted cooling load for Building_1 ]
                        },
                    'Building_2': ... (as above),
                    'Building_3': ... (as above),
                    'Solar_Generation': [ list of 48 floats - predicted solar generation ],
                    'Carbon_Intensity': [ list of 48 floats - predicted carbon intensity ]
                }
        """

        # ====================================================================
        # insert your forecasting code here
        # ====================================================================

        # dummy forecaster for illustration - delete for your implementation
        # ====================================================================
        # NOTE: this observation parsing only works of central agent setups
        # which is the case for the default schema provided.
        # View observations formatting via CityLearnEnv.observation_names
        current_vals = {
            **{b_name: {
                'Equipment_Eletric_Power': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'non_shiftable_load')[0][i]],
                'DHW_Heating': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'dhw_demand')[0][i]],
                'Cooling_Load': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'cooling_demand')[0][i]]
            } for i, b_name in enumerate(self.building_names)},
            'Solar_Generation': np.array(observations)[0][
                                    np.where(np.array(self.observation_names)[0] == 'solar_generation')[0][0]] / self.b0_pv_capacity * 1000,
            # Note: the `solar_generation` observations are the kWh produced by the panels on each building in the preceeding hour.
            # As we want to predict the normalised solar generation [W/kW] common to all buildings, we back-calculate this value using the
            # first building; normalised solar generation [W/kW] = (building solar generation [kWh] / (building solar capacity [kW] * 1hr)) * 1000
            'Carbon_Intensity': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'carbon_intensity')[0][0]]
        }

        if self.prev_vals['Carbon_Intensity'] is None:
            predictions_dict = {
                **{b_name: {
                    'Equipment_Eletric_Power': [current_vals[b_name]['Equipment_Eletric_Power'] for _ in range(self.tau)],
                    'DHW_Heating': [current_vals[b_name]['DHW_Heating'] for _ in range(self.tau)],
                    'Cooling_Load': [current_vals[b_name]['Cooling_Load'] for _ in range(self.tau)]
                } for i, b_name in enumerate(self.building_names)},
                'Solar_Generation': [current_vals['Solar_Generation'] for _ in range(self.tau)],
                'Carbon_Intensity': [current_vals['Carbon_Intensity'] for _ in range(self.tau)]
            }

        else:
            predictions_dict = {}
            predict_inds = [t + 1 for t in range(self.tau)]

            for b_name in self.building_names:
                predictions_dict[b_name] = {}
                for load_type in ['Equipment_Eletric_Power', 'DHW_Heating', 'Cooling_Load']:
                    predictions_dict[b_name][load_type] = np.poly1d(
                        np.polyfit([-1, 0], [self.prev_vals[b_name][load_type], current_vals[b_name][load_type]], deg=1))(predict_inds)

            predictions_dict['Solar_Generation'] = np.poly1d(
                np.polyfit([-1, 0], [self.prev_vals['Solar_Generation'], current_vals['Solar_Generation']], deg=1))(predict_inds)
            predictions_dict['Carbon_Intensity'] = np.poly1d(
                np.polyfit([-1, 0], [self.prev_vals['Carbon_Intensity'], current_vals['Carbon_Intensity']], deg=1))(predict_inds)

        self.prev_vals = current_vals
        # ====================================================================

        return predictions_dict
