"""PPO training code."""

import collections
import copy
import csv
from functools import partial
import os
import pickle
import shutil
import time
import gin
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from tensorflow import flags
import wandb
from single_agent_env import make_single_env, get_human_policies

FLAGS = flags.FLAGS
flags.DEFINE_string("name", "ppo_driving", "Name of experiment")
flags.DEFINE_multi_string("gin_file", "configs/ppo.gin", "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings."
)
flags.DEFINE_string("logdir", "/tmp/driving", "Logdir")

PPO2 = gin.external_configurable(PPO2)
gin_VecNormalize = gin.external_configurable(VecNormalize)


@gin.configurable
def train(
    experiment_name,
    logdir,
    num_envs=1,
    timesteps=gin.REQUIRED,
    recurrent=False,
    eval_save_period=100,
    human_mode="bc_2",
    split_train_eval=True,
    nonreplace=False,
):
    """Train and eval ppo agent."""
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)
    os.makedirs(experiment_name)
    rets_path = os.path.join(experiment_name, "eval.csv")
    wandb.save(experiment_name)
    human_policies = get_human_policies(human_mode, 0.1)
    if nonreplace:
        assert num_envs == len(human_policies)
        env_fns = [
            partial(make_single_env, human_policies=[h_pol]) for h_pol in human_policies
        ]
    else:
        env_fns = num_envs * [lambda: make_single_env(human_policies=human_policies)]
    env = gin_VecNormalize(SubprocVecEnv(env_fns))
    if split_train_eval:
        eval_human_policies = get_human_policies(human_mode, 0.1)
    else:
        eval_human_policies = copy.deepcopy(human_policies)
    eval_envs = []
    for human_policy in eval_human_policies:
        eval_env_fns = num_envs * [partial(make_single_env, human_policies=[human_policy])]
        # Get true (un-normalized) rewards out from eval env.
        eval_envs.append(
            VecNormalize(DummyVecEnv(eval_env_fns), training=False, norm_reward=False)
        )
    policy = MlpLnLstmPolicy if recurrent else MlpPolicy
    model = PPO2(policy, env, verbose=1, tensorboard_log=logdir)
    op_config_path = os.path.join(experiment_name, "operative_config.gin")
    with open(op_config_path, "w") as f:
        f.write(gin.operative_config_str())

    def evaluate(model, eval_dir):
        # Need to transfer running avgs from env->eval_env
        model.save(os.path.join(eval_dir, "model.pkl"))
        env.save_running_average(eval_dir)
        for eval_env in eval_envs:
            eval_env.load_running_average(eval_dir)
        task_rets = []
        data_dicts = []
        for eval_env in eval_envs:
            obs = eval_env.reset()
            rets = np.zeros(num_envs)
            state, dones = None, [False for _ in range(num_envs)]
            ever_done = np.zeros((num_envs,), dtype=np.bool)
            # Maps env_idx -> (state, action, reward, done) tuples
            task_data = collections.defaultdict(list)
            while not np.all(ever_done):
                true_states = [
                    inner_env.multi_env.world.state for inner_env in eval_env.venv.envs
                ]
                action, state = model.predict(obs, state=state, mask=dones, deterministic=True)
                next_obs, rewards, dones, _info = eval_env.step(action)
                for env_idx, data in enumerate(zip(true_states, action, rewards, dones)):
                    if not ever_done[env_idx]:
                        task_data[env_idx].append(data)
                        rets[env_idx] += rewards[env_idx]
                ever_done = np.logical_or(ever_done, dones)
                obs = next_obs
            task_rets.append(np.mean(rets))
            data_dicts.append(task_data)
        with open(os.path.join(eval_dir, "data_dicts.pkl"), "wb") as f:
            pickle.dump(data_dicts, f)
        return task_rets

    n_steps = 0  # pylint: disable=unused-variable

    def callback(_locals, _globals):
        nonlocal n_steps
        model = _locals["self"]
        if (n_steps + 1) % eval_save_period == 0:
            start_eval_time = time.time()
            eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
            os.makedirs(eval_dir)
            rets = evaluate(model, eval_dir)
            avg_ret = np.mean(rets)
            wandb.log({"avg_eval_ret": avg_ret}, step=_locals["self"].num_timesteps)
            with open(rets_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([n_steps, avg_ret] + [ret for ret in rets])
            end_eval_time = time.time() - start_eval_time
            print("Finished evaluation in {:.2f} seconds".format(end_eval_time))
        n_steps += 1
        return True

    model.learn(total_timesteps=timesteps, callback=callback)
    final_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
    os.makedirs(final_dir)
    rets = evaluate(model, final_dir)
    avg_ret = np.mean(rets)
    with open(rets_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([n_steps, avg_ret] + [ret for ret in rets])


if __name__ == "__main__":
    wandb.init(project="hr_adaptation", sync_tensorboard=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train(FLAGS.name, FLAGS.logdir)
