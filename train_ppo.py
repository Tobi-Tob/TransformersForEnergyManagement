import argparse
import os
from citylearn.citylearn import CityLearnEnv
from rewards.user_reward import SubmissionReward
from city_gym_env import CityGymEnv
from stable_baselines3 import PPO


def train_ppo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=float)

    args = parser.parse_args()
    model_dir = args.model_dir
    continue_training = args.continue_training
    learning_rate = args.lr
    training_steps = args.steps

    env = CityLearnEnv('data/schemas/warm_up/schema.json', reward_function=SubmissionReward)
    env = CityGymEnv(env)

    env.reset()

    # load model if exist
    if os.path.exists(model_dir) and continue_training:
        agent = PPO.load(model_dir)
    else:
        agent = PPO('MlpPolicy', env, learning_rate=learning_rate, gamma=1)

    # Train the agent
    agent.learn(total_timesteps=training_steps, progress_bar=True, log_interval=100, reset_num_timesteps=True)

    agent.save(model_dir)

    return agent


if __name__ == '__main__':
    train_ppo()

# python train_ppo.py --model_dir models\ppo1 --lr 0.0003 --steps 10000
