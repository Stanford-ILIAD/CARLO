import wandb

wandb.init(project="hr_adaptation", sync_tensorboard=True)
import os
import argparse
import gym
import numpy as np
from single_agent_env import make_single_env
from stable_baselines import PPO2, SAC, TD3
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import LnMlpPolicy as SacLnMlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize


def train(experiment_name, logdir, total_timesteps):
    env = VecNormalize(SubprocVecEnv(3 * [make_single_env]), norm_reward=False)
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=logdir, n_steps=300, ent_coef=5e-4)
    model.learn(total_timesteps=total_timesteps)
    model.save(experiment_name)
    avgs_folder = "{}_avgs".format(experiment_name)
    os.makedirs(avgs_folder)
    env.save_running_average(avgs_folder)
    wandb.save("./{:s}.pkl".format(experiment_name))


def train_sac(experiment_name, logdir, total_timesteps):
    env = DummyVecEnv([make_single_env])
    model = SAC(SacLnMlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    model.save(experiment_name)
    wandb.save("./{:s}.pkl".format(experiment_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--logdir", type=str, default="/tmp/driving")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    args = parser.parse_args()
    train(args.experiment_name, args.logdir, args.total_timesteps)
