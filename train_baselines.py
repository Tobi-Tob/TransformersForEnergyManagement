import argparse
import datetime

# import tensorflow
import torch
from citylearn.citylearn import CityLearnEnv
from rewards.user_reward import SubmissionReward
from env_wrapper import CityEnvForTraining
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"])
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=float)

    args = parser.parse_args()
    algo = args.algo
    model_id = args.model_id
    model_dir = f"models/{model_id}"
    log_dir = f"logs/" + datetime.datetime.now().strftime("%m%d")
    # tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    continue_training = args.continue_training
    learning_rate = args.lr
    training_steps = args.steps

    env = CityLearnEnv('data/schemas/warm_up/schema.json', reward_function=SubmissionReward)
    env = CityEnvForTraining(env)  # Environment only for training
    env.reset()

    # load model if exist
    if algo == "PPO":
        agent = init_ppo(env, learning_rate, log_dir)
    elif algo == "SAC":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Train the agent
    agent.learn(total_timesteps=training_steps, callback=None, log_interval=1, tb_log_name=model_id, reset_num_timesteps=True, progress_bar=True)

    agent.save(model_dir)

    return agent


def init_ppo(env, learning_rate, log_dir):
    # Actor and Critic network parameters
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))

    agent = PPO(policy='MlpPolicy',
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=learning_rate,
                gamma=1,
                clip_range=0.2,
                use_sde=True,
                tensorboard_log=log_dir,
                verbose=2)

    env.set_agent(agent)  # allow CityEnvForTraining access to the model

    return agent


if __name__ == '__main__':
    main()

# python train_baselines.py --algo PPO --model_id ppo1 --lr 0.0003 --steps 2000
