import wandb

wandb.init(project="hr_adaptation", sync_tensorboard=True)
import os
import shutil
import gin
import gym
import numpy as np
from single_agent_env import make_single_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("name", None, "Name of experiment")
flags.DEFINE_multi_string("gin_file", "configs/ppo.gin", "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings."
)
flags.DEFINE_string("logdir", "/tmp/driving", "Logdir")

PPO2 = gin.external_configurable(PPO2)
VecNormalize = gin.external_configurable(VecNormalize)


@gin.configurable
def train(experiment_name, logdir, num_envs=1, timesteps=gin.REQUIRED, recurrent=False):
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)
    os.makedirs(experiment_name)
    env = VecNormalize(SubprocVecEnv(num_envs * [make_single_env]))
    policy = MlpLnLstmPolicy if recurrent else MlpPolicy
    model = PPO2(policy, env, verbose=1, tensorboard_log=logdir)
    op_config_path = os.path.join(experiment_name, "operative_config.gin")
    with open(op_config_path, "w") as f:
        f.write(gin.operative_config_str())
        wandb.save(op_config_path)
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(experiment_name, "model"))
    env.save_running_average(experiment_name)
    wandb.save(os.path.join(experiment_name, "*.pkl"))


if __name__ == "__main__":
    flags.mark_flag_as_required("name")
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train(FLAGS.name, FLAGS.logdir)
