import argparse
import datetime
import os

import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from env_wrapper import CityEnvForTraining
from stable_baselines3 import SAC

from rewards.user_reward import SubmissionReward
from utils import init_environment, CustomCallback


class AttentionBasedFeatureExtractor(BaseFeaturesExtractor):
    """
    Not used
    :param observation_space: (gym.Space)
    """

    def __init__(self, observation_space, features_dim=None, num_heads = 1, dropout = 0.0):
        self.input_dim = get_flattened_obs_dim(observation_space)
        if features_dim is None:
            features_dim = self.input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        super().__init__(observation_space, features_dim=features_dim)

        # Self Attention block
        self.self_attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads, dropout=self.dropout)

        # linear block
        # self.linear_net = nn.Sequential(
        #     nn.Linear(self.input_dim, self.features_dim),
        #     nn.Dropout(self.dropout))

        # Layers to apply in between the main layers
        self.flatten = nn.Flatten()
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(self.features_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Flatten input: 26 ---> 26
        x = self.flatten(observations)

        # Attention block: 26 ---> 26
        attention_out, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attention_out)
        x = self.norm1(x)

        # Linear block:  26 ---> 8
        # linear_out = self.linear_net(x)
        # x = self.dropout(linear_out)
        # x = self.norm2(x)

        return x

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    args = parser.parse_args()
    # Model name
    model_id = 'test' if args.id is None else args.id
    model_id = "SAC_" + str(model_id)
    model_dir = f"my_models/{model_id}"
    log_dir = f"logs/" + datetime.datetime.now().strftime("%m%d")

    # Hyperparameters
    policy = 'MlpPolicy'  # Multi Layer Perceptron Policy
    learning_rate = 3e-4  # good between 3e-4 and 3e-3
    pi_network = [256, 256]
    q_network = [256, 256]
    activation_fn = torch.nn.ReLU  # LeakyReLU
    # features_dim = None
    # num_heads = 2
    # dropout = 0.0
    buffer_size = 10_000
    batch_size = 256
    gamma = 1

    total_timesteps = 20_000  # total timesteps to run in the environment
    eval_interval = 1438  # how frequent to do a validation run in the complete environment
    n_eval_episodes = 4  # do n episodes for each validation run
    save_interval = 1438  # save model every n timesteps
    buildings_to_remove = 0  # 0 to use all 3 buildings for training

    init_with_given_model_params = False
    continue_with_given_model = False
    model_to_continue = 'my_models/submission_models/SAC_c8__11504.zip'

    # Initialize the training environment
    training_buildings = [1, 2, 3]
    if buildings_to_remove is not 0:
        training_buildings.remove(buildings_to_remove)

    env = init_environment(training_buildings, [0, 719], reward_function=SubmissionReward)
    env = CityEnvForTraining(env)  # Environment only for training
    env.reset()

    policy_kwargs = dict(activation_fn=activation_fn,
                         net_arch=dict(pi=pi_network, qf=q_network), # Initially shared then diverging: [128, dict(vf=[256], pi=[16])]
                         # features_extractor_class=AttentionBasedFeatureExtractor,
                         # features_extractor_kwargs=dict(features_dim=features_dim, num_heads = num_heads, dropout = dropout)
                         )

    # Initialize the agent
    agent = SAC(policy=policy,
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=719 * len(training_buildings),
                batch_size=batch_size,
                tau=0.005,  # soft update coefficient
                gamma=gamma,
                ent_coef='auto',  # Entropy regularization coefficient, 'auto'
                # action_noise=NormalActionNoise(0.0, 0.1), look at common.noise, helps for hard exploration problem
                use_sde=False,
                stats_window_size=1,  # Window size for the rollout logging, specifying the number of episodes to average
                tensorboard_log=log_dir,
                verbose=2)

    assert os.path.exists(model_to_continue)
    assert not init_with_given_model_params or not continue_with_given_model

    if init_with_given_model_params:  # init from model checkpoint
        agent.set_parameters(model_to_continue)

    if continue_with_given_model:  # continue training from checkpoint with saved parameters
        agent = SAC.load(path=model_to_continue, env=env)

    env.set_evaluation_model(agent)  # allow CityEnvForTraining access to the model

    sub_id = 'm' + str(buildings_to_remove)

    custom_callback = CustomCallback(eval_interval=eval_interval, n_eval_episodes=n_eval_episodes)
    checkpoint_callback = CheckpointCallback(save_path=model_dir, save_freq=save_interval, name_prefix=sub_id, save_vecnormalize=True, verbose=2)

    # Train the agent
    agent.learn(total_timesteps=total_timesteps, callback=[custom_callback, checkpoint_callback], log_interval=1,
                tb_log_name=model_id, reset_num_timesteps=True, progress_bar=True)

    agent.save(f"{model_dir}/{sub_id}_complete")


if __name__ == '__main__':
    train()

# Start training in terminal:
# python train_SAC.py --id 4 --continue_with_model 'my_models/submission_models/SAC_c8__11504.zip'
# tensorboard --logdir=logs
