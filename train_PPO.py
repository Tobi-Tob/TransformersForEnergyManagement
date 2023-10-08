import argparse
import datetime

# import tensorflow
import torch
from citylearn.citylearn import CityLearnEnv
from rewards.user_reward import SubmissionReward
from env_wrapper import CityEnvForTraining
from stable_baselines3 import PPO
from utils import init_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=float)

    args = parser.parse_args()
    model_id = args.model_id
    model_id = "PPO_" + model_id
    model_dir = f"my_models/{model_id}"
    log_dir = f"logs/" + datetime.datetime.now().strftime("%m%d")
    # tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    learning_rate = args.lr
    training_steps = args.steps

    for i in [1, 2, 3]:
        training_buildings = [1, 2, 3]
        training_buildings.remove(i)
        env = init_environment(training_buildings, [0, 719])
        env = CityEnvForTraining(env)  # Environment only for training
        env.reset()

        agent = init_ppo(env, learning_rate, log_dir)
        sub_id = 'm' + str(i)
        model_sub_id = model_id + '_' + sub_id

        agent.learn(total_timesteps=training_steps, callback=None, log_interval=1,
                    tb_log_name=model_sub_id, reset_num_timesteps=True, progress_bar=True)

        agent.save(f"{model_dir}/{sub_id}")


def init_ppo(env, learning_rate, log_dir):

    # Hyperparameters
    policy = 'MlpPolicy'  # Multi Layer Perceptron Policy
    pi_network = [64, 64]
    v_network = [64, 64]
    clip_range = 0.2

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,  # vllt tanh als letzte Aktivierung?
                         net_arch=dict(pi=pi_network, vf=v_network))

    agent = PPO(policy=policy,
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=learning_rate,
                n_steps=1440,
                batch_size=48,  # 48
                gamma=1,
                clip_range=clip_range,
                use_sde=False,
                tensorboard_log=log_dir,
                verbose=2)

    env.set_evaluation_model(agent)  # allow CityEnvForTraining access to the model

    return agent


if __name__ == '__main__':
    main()

# TODO print value net output while training
# python train_PPO.py --model_id 1 --lr 3e-3 --steps 20000

