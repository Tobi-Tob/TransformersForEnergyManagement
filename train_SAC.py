import argparse
import datetime
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

from env_wrapper import CityEnvForTraining
from stable_baselines3 import SAC

from rewards.temp_diff_reward import TempDiffReward
from utils import init_environment, CustomCallback


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    args = parser.parse_args()
    # Model name
    model_id = 4 if args.id is None else args.id
    model_id = "SAC_" + str(model_id)
    model_dir = f"my_models/{model_id}"
    log_dir = f"logs/" + datetime.datetime.now().strftime("%m%d")

    # Hyperparameters
    policy = 'MlpPolicy'  # Multi Layer Perceptron Policy
    learning_rate = 3e-4  # good between 3e-4 and 3e-3
    pi_network = [250, 250]
    q_network = [250, 250]
    activation_fn = torch.nn.ReLU  # LeakyReLU
    buffer_size = 100_000
    batch_size = 256

    total_timesteps = 100_000  # total timesteps to run in the environment
    eval_interval = 1438  # doing a validation run in the complete env
    save_interval = 1438  # save model every n timesteps

    for i in [1]:
        # Initialize the training environment
        training_buildings = [1, 2, 3]
        training_buildings.remove(i)
        env = init_environment(training_buildings, [0, 719], reward_function=TempDiffReward)
        env = CityEnvForTraining(env)  # Environment only for training
        env.reset()

        # Initialize the agent
        agent = SAC(policy=policy,
                    policy_kwargs=dict(activation_fn=activation_fn, net_arch=dict(pi=pi_network, qf=q_network)),
                    env=env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=100,
                    batch_size=batch_size,
                    tau=0.005,  # soft update coefficient
                    gamma=1,
                    # ent_coef='auto',  # Entropy regularization coefficient
                    # action_noise=NormalActionNoise(0.0, 0.1),
                    stats_window_size=1,  # Window size for the rollout logging, specifying the number of episodes to average
                    tensorboard_log=log_dir,
                    verbose=2)

        env.set_evaluation_model(agent)  # allow CityEnvForTraining access to the model

        sub_id = 'm' + str(i)
        model_sub_id = model_id + '_' + sub_id

        custom_callback = CustomCallback(eval_interval=eval_interval)
        checkpoint_callback = CheckpointCallback(save_path=model_dir, save_freq=save_interval, name_prefix=sub_id, save_vecnormalize=True, verbose=2)

        # Train the agent
        agent.learn(total_timesteps=total_timesteps, callback=[custom_callback, checkpoint_callback], log_interval=1,
                    tb_log_name=model_sub_id, reset_num_timesteps=False, progress_bar=True)

        agent.save(f"{model_dir}/{sub_id}_complete")


if __name__ == '__main__':
    train()

# Start training in terminal:
# python train_SAC.py --id 4
# tensorboard --logdir=logs
