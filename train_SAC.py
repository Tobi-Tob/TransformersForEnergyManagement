import datetime
import torch
from citylearn.citylearn import CityLearnEnv
from stable_baselines3.common.noise import NormalActionNoise

from rewards.user_reward import SubmissionReward
from env_wrapper import CityEnvForTraining
from stable_baselines3 import SAC
from utils import init_environment, CustomCallback


def train():
    # Model name
    model_id = 1
    model_id = "SAC_" + str(model_id)
    model_dir = f"my_models/{model_id}"
    log_dir = f"logs/" + datetime.datetime.now().strftime("%m%d")

    # Hyperparameters
    policy = 'MlpPolicy'  # Multi Layer Perceptron Policy
    learning_rate = 3e-4  # 3e-3, 5e-4 later lower lr, value net higher lr?
    pi_network = [250, 250]  # [250, 250]
    v_network = [250, 250]  # [250, 250]
    activation_fn = torch.nn.ReLU  # LeakyReLU
    buffer_size = 100_000
    batch_size = 256  # 72, 200, 500

    total_timesteps = 1_000_000  # total timesteps to run in the environment
    eval_interval = 1440  # doing a validation run in the complete env
    save_interval = 3600  # save model every n timesteps

    for i in [1]:
        # Initialize the training environment
        training_buildings = [1, 2, 3]
        training_buildings.remove(i)
        env = init_environment(training_buildings, [0, 719])
        env = CityEnvForTraining(env)  # Environment only for training
        env.reset()

        # Initialize the agent
        agent = SAC(policy=policy,
                    # policy_kwargs=dict(activation_fn=activation_fn, net_arch=dict(pi=pi_network, vf=v_network)),
                    env=env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=100,
                    batch_size=batch_size,
                    tau=0.005,  # soft update coefficient
                    gamma=1,
                    # ent_coef='auto',  # Entropy regularization coefficient
                    # action_noise=NormalActionNoise(0.0, 0.1),
                    stats_window_size=5,  # Window size for the rollout logging, specifying the number of episodes to average
                    tensorboard_log=log_dir,
                    verbose=2)

        env.set_evaluation_model(agent)  # allow CityEnvForTraining access to the model
        custom_callback = CustomCallback(eval_interval=eval_interval)

        sub_id = 'm' + str(i)
        model_sub_id = model_id + '_' + sub_id

        # Train the agent
        for interval in range(1, int(total_timesteps/save_interval)):
            agent.learn(total_timesteps=save_interval, log_interval=1, callback=custom_callback,
                        tb_log_name=model_sub_id, reset_num_timesteps=False, progress_bar=True)
            agent.save(f"{model_dir}/{sub_id}_{save_interval*interval}")

        # agent.save(f"{model_dir}/{sub_id}")


if __name__ == '__main__':
    train()

# python train_SAC.py
# tensorboard --logdir=logs
