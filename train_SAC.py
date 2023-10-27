import argparse
import datetime
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

from env_wrapper import CityEnvForTraining
from stable_baselines3 import SAC

from rewards.custom_reward import UnservedEnergyReward, TempDiffReward
from rewards.weighted_reward import WeightedRewardFunction
from utils import init_environment, CustomCallback


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
    learning_rate = 3e-5  # good between 3e-4 and 3e-3
    pi_network = [256, 256]
    q_network = [256, 256]
    activation_fn = torch.nn.ReLU  # LeakyReLU
    buffer_size = 20_000
    batch_size = 256
    gamma = 1

    total_timesteps = 20_000  # total timesteps to run in the environment
    eval_interval = 1438  # how frequent to do a validation run in the complete environment
    n_eval_episodes = 1  # do n episodes for each validation run
    save_interval = 1438  # save model every n timesteps
    buildings_to_remove = 0  # 0 to use all 3 buildings for training

    # Initialize the training environment
    training_buildings = [1, 2, 3]
    if buildings_to_remove is not 0:
        training_buildings.remove(buildings_to_remove)

    env = init_environment(training_buildings, [0, 719], reward_function=UnservedEnergyReward)
    env = CityEnvForTraining(env)  # Environment only for training
    env.reset()

    # Initialize the agent
    agent = SAC(policy=policy,
                policy_kwargs=dict(activation_fn=activation_fn, net_arch=dict(pi=pi_network, qf=q_network)),
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=719,
                batch_size=batch_size,
                tau=0.005,  # soft update coefficient
                gamma=gamma,
                ent_coef='auto',  # Entropy regularization coefficient, 'auto'
                # action_noise=NormalActionNoise(0.0, 0.1),
                stats_window_size=1,  # Window size for the rollout logging, specifying the number of episodes to average
                tensorboard_log=log_dir,
                verbose=2)

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
# python train_SAC.py --id 4
# tensorboard --logdir=logs
