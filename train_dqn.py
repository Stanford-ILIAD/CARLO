import wandb

wandb.init(project="hr_adaptation", sync_tensorboard=True)
import os
import shutil
import gin
import gym
import numpy as np
from single_agent_env import make_single_env
from stable_baselines import DQN
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("name", None, "Name of experiment")
flags.DEFINE_multi_string("gin_file", "configs/dqn.gin", "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings."
)
flags.DEFINE_string("logdir", "/tmp/driving", "Logdir")

DQN = gin.external_configurable(DQN)
VecNormalize = gin.external_configurable(VecNormalize)


@gin.configurable
def train(experiment_name, logdir, timesteps=gin.REQUIRED):
    env = VecNormalize(DummyVecEnv([lambda: make_single_env(discrete=True)]))
    model = DQN(LnMlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=timesteps)
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)
    os.makedirs(experiment_name)
    model.save(os.path.join(experiment_name, "model"))
    env.save_running_average(experiment_name)
    wandb.save(os.path.join(experiment_name, "*.pkl"))


if __name__ == "__main__":
    flags.mark_flag_as_required("name")
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    for gin_file in FLAGS.gin_file:
        wandb.save(gin_file)
    train(FLAGS.name, FLAGS.logdir)
