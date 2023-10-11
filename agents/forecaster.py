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
        self._pv_nominal_power = 2.4  # value of building 1 which was used for training
        self._diffuse_solar_irradiance_predictions = np.full(6, np.nan)
        self._direct_solar_irradiance_predictions = np.full(6, np.nan)

    def get_diffuse_solar_irradiance(self):
        return self._diffuse_solar_irradiance_predictions

    def reset(self):
        self._diffuse_solar_irradiance_predictions = np.full(6, np.nan)
        self._direct_solar_irradiance_predictions = np.full(6, np.nan)

    def predict_solar_generation(self, observation):
        """
        Returns normalized solar generation of next time step given the weather.
        Prediction has to be multiplied by the PV nominal power of the building to be adequate.
        """
        X = self.modify_obs(observation)
        X = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        next_solar_prediction = self.model(X).detach().numpy()[0][0]
        if next_solar_prediction < self.epsilon_clip:
            next_solar_prediction = 0

        return next_solar_prediction

    def modify_obs(self, obs: List[List[float]]) -> np.array:
        """
        Input: (1,52), Output: (7)
        Modify the observation space to:
        [['hour', 'outdoor_dry_bulb_temperature', 'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_1h',
        'direct_solar_irradiance', 'direct_solar_irradiance_predicted_1h', 'solar_generation'],...]
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
        building_solar_generation = obs[17] / self._pv_nominal_power

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


def train_solar_forcaster(save_model: bool):
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
            optimizer = optim.Adam(model.parameters(), lr=lr/10)
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


if __name__ == '__main__':
    train_solar_forcaster(save_model=False)
