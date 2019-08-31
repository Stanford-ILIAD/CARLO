import wandb

wandb.init(project="hr_adaptation", sync_tensorboard=True)
import os
import shutil
import argparse
import gin
import gym
import numpy as np
from single_agent_env import make_single_env
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

SAC = gin.external_configurable(SAC)
VecNormalize = gin.external_configurable(VecNormalize)


@gin.configurable
def train(
    logdir, num_envs=1, experiment_name=gin.REQUIRED, timesteps=gin.REQUIRED, layer_norm=True
):
    env = VecNormalize(DummyVecEnv([make_single_env]))
    policy_cls = LnMlpPolicy if layer_norm else MlpPolicy
    model = SAC(policy_cls, env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=timesteps)
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)
    os.makedirs(experiment_name)
    model.save(os.path.join(experiment_name, "model"))
    env.save_running_average(experiment_name)
    wandb.save(os.path.join(experiment_name, "*.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logdir", type=str, default="/tmp/driving")
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    wandb.save(args.config)
    train(args.logdir)
