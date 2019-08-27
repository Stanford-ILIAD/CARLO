import argparse
import gym
from single_agent_env import make_single_env
from stable_baselines import PPO2, SAC
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


def train(experiment_name, logdir, total_timesteps):
    env = DummyVecEnv([make_single_env])
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=logdir, n_steps=500)
    model.learn(total_timesteps=total_timesteps)
    model.save(experiment_name)


def train_sac(experiment_name, logdir, total_timesteps):
    env = DummyVecEnv([make_single_env])
    model = SAC(SacMlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps)
    model.save(experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--logdir", type=str, default="/tmp/ppo_driving")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    args = parser.parse_args()
    train(args.experiment_name, args.logdir, args.total_timesteps)
