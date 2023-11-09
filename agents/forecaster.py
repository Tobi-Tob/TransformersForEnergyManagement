from typing import List
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


class SolarGenerationForecaster:
    def __init__(self):
        self.model = torch.load('my_models/Forecaster/solar_forecaster')
        self.scaler = load('my_models/Forecaster/solar_obs_scaler.bin')
        self.input_dim = 7
        self.epsilon_clip = 8e-3  # clip predictions smaller than epsilon to 0
        self._diffuse_solar_irradiance_predictions = np.full(6, np.nan)
        self._direct_solar_irradiance_predictions = np.full(6, np.nan)

    def get_diffuse_solar_irradiance(self):
        return self._diffuse_solar_irradiance_predictions

    def reset(self):
        self._diffuse_solar_irradiance_predictions = np.full(6, np.nan)
        self._direct_solar_irradiance_predictions = np.full(6, np.nan)

    def forecast(self, observation, pv_nominal_power_building_0):
        """
        Input observations and the first buildings pv nominal power.
        Returns normalized solar generation of next time step given the weather.
        Prediction has to be multiplied by the PV nominal power of the building to be adequate.
        """
        X = self.modify_obs(observation, pv_nominal_power_building_0)
        X = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        next_solar_prediction = self.model(X).detach().numpy()[0][0]
        if next_solar_prediction < self.epsilon_clip:
            next_solar_prediction = 0

        return next_solar_prediction

    def modify_obs(self, obs: List[List[float]], pv_nominal_power_building_0) -> np.array:
        """
        Input: (1,52), Output: (7)
        Modify the observation space to:
        [['hour', 'outdoor_dry_bulb_temperature', 'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_1h',
        'direct_solar_irradiance', 'direct_solar_irradiance_predicted_1h', 'solar_generation']]
        """
        #  --> Delete unimportant observations
        #  --> Add usefully observation (prediction_1h)
        #  --> Solar generation normalize with pv nominal power

        obs = obs[0]

        hour = obs[1]
        temperature = obs[2]
        diffuse_solar = obs[6]
        diffuse_solar_6h = obs[7]
        direct_solar = obs[10]
        direct_solar_6h = obs[11]
        building_solar_generation = obs[17] / pv_nominal_power_building_0

        # update prediction history
        self._diffuse_solar_irradiance_predictions[0:5] = self._diffuse_solar_irradiance_predictions[1:6]
        self._diffuse_solar_irradiance_predictions[5] = diffuse_solar_6h
        self._direct_solar_irradiance_predictions[0:5] = self._direct_solar_irradiance_predictions[1:6]
        self._direct_solar_irradiance_predictions[5] = direct_solar_6h

        diffuse_solar_1h = self._diffuse_solar_irradiance_predictions[0] if not np.isnan(
            self._diffuse_solar_irradiance_predictions[0]) else diffuse_solar
        direct_solar_1h = self._direct_solar_irradiance_predictions[0] if not np.isnan(self._direct_solar_irradiance_predictions[0]) else direct_solar

        obs_modified = np.array([[hour, temperature, diffuse_solar, diffuse_solar_1h, direct_solar, direct_solar_1h, building_solar_generation]])

        return obs_modified


def train_solar_forecaster(save_model: bool):
    # hyperparameters
    lr = 0.005
    n_epochs = 500
    batch_size = 100

    # load data set
    X = np.load('../data/solar_forecast/X.npy')
    y = np.load('../data/solar_forecast/y.npy')

    # train-test split for model evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Define the model
    model = nn.Sequential(
        nn.Linear(len(X[0]), 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )

    # loss function and optimizer
    loss_function = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_test_loss = np.inf  # init to infinity
    best_weights = None
    training_loss_history = []
    test_loss_history = []

    for epoch in range(n_epochs):
        if epoch == 100:  # lower learning rate
            optimizer = optim.Adam(model.parameters(), lr=lr / 10)
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(training_loss=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        # eval training_loss
        y_pred = model(X_train)
        training_loss = loss_function(y_pred, y_train)
        training_loss = float(training_loss)
        training_loss_history.append(training_loss)
        # eval test_loss
        y_pred = model(X_test)
        test_loss = loss_function(y_pred, y_test)
        test_loss = float(test_loss)
        test_loss_history.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    if save_model:
        torch.save(model, '../my_models/Forecaster/solar_forecaster')
        dump(scaler, '../my_models/Forecaster/solar_obs_scaler.bin', compress=False)

    print("Test MSE: %.6f" % best_test_loss)
    print("Test RMSE: %.6f" % np.sqrt(best_test_loss))
    plt.plot(training_loss_history, label="Training MSE")
    plt.plot(test_loss_history, label="Test MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()


class TemperatureForecaster:
    def __init__(self):
        self.model = torch.load('my_models/Forecaster/temperature_forecaster')
        self.scaler = load('my_models/Forecaster/temperature_obs_scaler.bin')
        self.input_dim = 5
        self._temperature_predictions = np.full(6, np.nan)
        self._direct_solar_irradiance_predictions = np.full(6, np.nan)

    def reset(self):
        self._temperature_predictions = np.full(6, np.nan)
        self._direct_solar_irradiance_predictions = np.full(6, np.nan)

    def forecast(self, observation):
        """
        Returns the predicted temperature of next time step given the weather.
        """
        X = self.modify_obs(observation)
        given_prediction = X[0][2]
        X = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        model_prediction = self.model(X).detach().numpy()[0][0]

        def parametric_sigmoid(x):
            a = 7
            b = 1.3
            return 1 / (1 + np.exp(-a * (x - b)))

        diff = np.abs(model_prediction - given_prediction)
        model_prediction = parametric_sigmoid(diff) * given_prediction + (1-parametric_sigmoid(diff)) * model_prediction

        return model_prediction

    def modify_obs(self, obs: List[List[float]]) -> np.array:
        """
        Input: (1,52), Output: (5)
        Modify the observation space to:
        [['hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_1h',
        'direct_solar_irradiance', 'direct_solar_irradiance_predicted_1h']]
        """
        #  --> Delete unimportant observations
        #  --> Add usefully observation (prediction_1h by looking at the 6h prediction from 5h ago)

        obs = obs[0]

        hour = obs[1]
        temperature = obs[2]
        temperature_6h = obs[3]
        direct_solar = obs[10]
        direct_solar_6h = obs[11]

        # update prediction history
        self._temperature_predictions[0:5] = self._temperature_predictions[1:6]
        self._temperature_predictions[5] = temperature_6h
        self._direct_solar_irradiance_predictions[0:5] = self._direct_solar_irradiance_predictions[1:6]
        self._direct_solar_irradiance_predictions[5] = direct_solar_6h

        temperature_1h = self._temperature_predictions[0] if not np.isnan(self._temperature_predictions[0]) else temperature
        direct_solar_1h = self._direct_solar_irradiance_predictions[0] if not np.isnan(self._direct_solar_irradiance_predictions[0]) \
            else direct_solar

        obs_modified = np.array([[hour, temperature, temperature_1h, direct_solar, direct_solar_1h]])

        return obs_modified


def train_temperature_forecaster(save_model: bool):
    # hyperparameters
    lr = 0.005
    n_epochs = 800
    batch_size = 100

    # load data set
    X = np.load('../data/temperature_forecast/X.npy')
    y = np.load('../data/temperature_forecast/y.npy')

    # train-test split for model evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Define the model
    model = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.ReLU(),
        nn.Linear(64, 12),
        nn.ReLU(),
        nn.Linear(12, 1),
    )

    # loss function and optimizer
    loss_function = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_test_loss = np.inf  # init to infinity
    best_weights = None
    training_loss_history = []
    test_loss_history = []

    for epoch in range(n_epochs):
        if epoch == 200:  # lower learning rate
            optimizer = optim.Adam(model.parameters(), lr=lr / 2)
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(training_loss=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        # eval training_loss
        y_pred = model(X_train)
        training_loss = loss_function(y_pred, y_train)
        training_loss = float(training_loss)
        training_loss_history.append(training_loss)
        # eval test_loss
        y_pred = model(X_test)
        test_loss = loss_function(y_pred, y_test)
        test_loss = float(test_loss)
        test_loss_history.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    if save_model:
        torch.save(model, '../my_models/Forecaster/temperature_forecaster')
        dump(scaler, '../my_models/Forecaster/temperature_obs_scaler.bin', compress=False)

    print("Test MSE: %.6f" % best_test_loss)
    print("Test RMSE: %.6f" % np.sqrt(best_test_loss))  # good: 0.263619
    if not save_model:
        print('Did not save the model')
    plt.plot(training_loss_history, label="Training MSE")
    plt.plot(test_loss_history, label="Test MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()


class NoneShiftableLoadForecaster:
    def __init__(self, non_shiftable_load_estimate: List[float]):
        self.model = torch.load('my_models/Forecaster/load_forecaster')
        self.scaler = load('my_models/Forecaster/load_obs_scaler.bin')
        self.non_shiftable_load_estimate = non_shiftable_load_estimate
        self.input_dim = 4

    def reset(self, non_shiftable_load_estimate):
        self.non_shiftable_load_estimate = non_shiftable_load_estimate

    def forecast(self, observation) -> List[float]:
        """
        Input observations and the non-shiftable load estimates of each building.
        Returns a list of predicted non_shiftable_loads for every building.
        """
        modified_obs = self.modify_obs(observation)
        next_load_prediction = []
        for i, building_obs in enumerate(modified_obs):
            building_obs = [building_obs]
            X = torch.tensor(self.scaler.transform(building_obs), dtype=torch.float32)
            y = self.model(X).detach().numpy()[0][0]

            if building_obs[0][3] == -1:  # TODO not active
                y = 0.5 * (y + building_obs[0][2])
                print('outage', building_obs[0][3], 'load', building_obs[0][2], 'y_mean', y)
            next_load_prediction.append(y * self.non_shiftable_load_estimate[i])

        return next_load_prediction

    def modify_obs(self, obs: List[List[float]]) -> np.array:
        """
        Input: (1,52), Output: (4)
        Modify the observation space to:
        [['day_type', 'hour', 'current_non_shiftable_load_normalized', 'occupant_count'],...]
        """
        #  --> Delete unimportant observations
        #  --> Normalize current_non_shiftable_load with non_shiftable_load_estimate

        obs = obs[0]

        day_type = obs[0]
        hour = obs[1]

        obs_buildings = obs[15:21] + obs[25:]  # all building level observations (#buildings * 11)

        # building-level observations:
        assert len(obs_buildings) % 11 == 0  # 11 observations per building
        obs_single_building = [obs_buildings[i:i + 11] for i in range(0, len(obs_buildings), 11)]
        assert len(obs_single_building) == len(self.non_shiftable_load_estimate)
        modified_obs = []
        for i, b in enumerate(obs_single_building):
            current_non_shiftable_load_normalized = b[1] / self.non_shiftable_load_estimate[i]
            occupant_count = b[8]
            obs_building_i = [day_type, hour, current_non_shiftable_load_normalized, occupant_count]
            modified_obs.append(obs_building_i)
            assert len(obs_building_i) == 4

        return modified_obs


def train_load_forecaster(save_model: bool):
    # hyperparameters
    lr = 0.001
    n_epochs = 5000
    batch_size = 1000

    # load data set
    X = np.load('../data/load_forecast/X.npy')
    y = np.load('../data/load_forecast/y.npy')

    # train-test split for model evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Define the model
    model = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    # loss function and optimizer
    loss_function = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_test_loss = np.inf  # init to infinity
    best_weights = None
    training_loss_history = []
    test_loss_history = []

    for epoch in range(n_epochs):
        if epoch == 100:  # lower learning rate
            optimizer = optim.Adam(model.parameters(), lr=lr / 10)
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(training_loss=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        # eval training_loss
        y_pred = model(X_train)
        training_loss = loss_function(y_pred, y_train)
        training_loss = float(training_loss)
        training_loss_history.append(training_loss)
        # eval test_loss
        y_pred = model(X_test)
        test_loss = loss_function(y_pred, y_test)
        test_loss = float(test_loss)
        test_loss_history.append(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    if save_model:
        torch.save(model, '../my_models/Forecaster/load_forecaster')
        dump(scaler, '../my_models/Forecaster/load_obs_scaler.bin', compress=False)

    print("Test MSE: %.6f" % best_test_loss)  # best 0.658855
    print("Test RMSE: %.6f" % np.sqrt(best_test_loss))
    plt.plot(training_loss_history, label="Training MSE")
    plt.plot(test_loss_history, label="Test MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_load_forecaster(save_model=True)
