import datetime
import torch
from citylearn.citylearn import CityLearnEnv
from rewards.user_reward import SubmissionReward
from env_wrapper import CityEnvForTraining
from stable_baselines3 import PPO
from utils import init_environment, CustomCallback


def train():
    # Model name
    model_id = 1
    model_id = "PPO_" + str(model_id)
    model_dir = f"my_models/{model_id}"
    log_dir = f"logs/" + datetime.datetime.now().strftime("%m%d")

    # Hyperparameters
    policy = 'MlpPolicy'  # Multi Layer Perceptron Policy
    learning_rate = 3e-3  # 3e-3, 5e-4 later lower lr, value net higher lr?
    pi_network = [250, 250]  # [250, 250]
    v_network = [250, 250]  # [250, 250]
    activation_fn = torch.nn.ReLU  # LeakyReLU
    batch_size = 360  # 72, 200, 500
    clip_range = 0.2

    total_timesteps = 1_000_000  # total timesteps to run in the environment
    n_steps = 720  # 720, 2000, 10.000 number of steps to run per update
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
        agent = PPO(policy=policy,
                    policy_kwargs=dict(activation_fn=activation_fn, net_arch=dict(pi=pi_network, vf=v_network)),
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=1,
                    clip_range=clip_range,
                    tensorboard_log=log_dir,
                    verbose=2)

        env.set_evaluation_model(agent)  # allow CityEnvForTraining access to the model
        custom_callback = CustomCallback(eval_interval=eval_interval)

        sub_id = 'm' + str(i)
        model_sub_id = model_id + '_' + sub_id

        # Train the agent
        for interval in range(1, int(total_timesteps/save_interval)):
            agent.learn(total_timesteps=save_interval, callback=[custom_callback], log_interval=1,
                        tb_log_name=model_sub_id, reset_num_timesteps=False, progress_bar=True)
            agent.save(f"{model_dir}/{sub_id}_{save_interval*interval}")

        # agent.save(f"{model_dir}/{sub_id}")


if __name__ == '__main__':
    train()

# python train_PPO.py --model_id 1 --lr 3e-3 --steps 20000
# tensorboard --logdir=logs
