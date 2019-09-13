"""Train with PPO."""

import csv
from functools import partial
import os
import shutil
import time
import gin
from moviepy.editor import ImageSequenceClip
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from tensorflow import flags
import wandb
from single_agent_env import make_single_env

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
    train_human_max_accs=None,
    eval_human_max_accs=None,
):
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name)
    os.makedirs(experiment_name)
    best_dir = os.path.join(experiment_name, "best")
    final_dir = os.path.join(experiment_name, "final")
    rets_path = os.path.join(experiment_name, "eval.csv")
    os.makedirs(best_dir)
    os.makedirs(final_dir)
    wandb.save(experiment_name)
    if train_human_max_accs is None:
        train_human_max_accs = np.linspace(2, 4, num=10)
    if eval_human_max_accs is None:
        eval_human_max_accs = np.linspace(2, 4, num=num_envs)
    train_env_fn = partial(make_single_env, human_max_accs=train_human_max_accs)
    env = gin_VecNormalize(SubprocVecEnv(num_envs * [train_env_fn]))
    eval_env_fns = [
        partial(
            make_single_env, human_max_accs=[eval_human_max_accs[i % len(eval_human_max_accs)]]
        )
        for i in range(num_envs)
    ]
    # Get true returns out from eval env.
    eval_env = VecNormalize(DummyVecEnv(eval_env_fns), training=False, norm_reward=False)
    policy = MlpLnLstmPolicy if recurrent else MlpPolicy
    model = PPO2(policy, env, verbose=1, tensorboard_log=logdir)
    op_config_path = os.path.join(experiment_name, "operative_config.gin")
    with open(op_config_path, "w") as f:
        f.write(gin.operative_config_str())
    n_steps, best_mean = 0, -np.inf  # pylint: disable=unused-variable

    def evaluate(model, eval_dir, videos=True):
        # Need to transfer running avgs from env->eval_env
        model.save(os.path.join(eval_dir, "model.pkl"))
        env.save_running_average(eval_dir)
        eval_env.load_running_average(eval_dir)
        obs = eval_env.reset()
        if videos:
            imgs = []
            imgs.append(eval_env.get_images())
        rets = 0
        state = None
        ever_done = False
        state_history = []
        for _ in range(60):
            action, state = model.predict(obs, state=state, deterministic=True)
            next_obs, rewards, dones, _info = eval_env.step(action)
            state_history.append(
                [inner_env.multi_env.world.state for inner_env in eval_env.venv.envs]
            )
            ever_done = np.logical_or(dones, ever_done)
            rets += rewards * np.logical_not(ever_done)
            if videos:
                imgs.append(eval_env.get_images())
            obs = next_obs
        state_history = np.array(state_history)
        np.save(os.path.join(eval_dir, "state_history.npy"), state_history)
        if videos:
            for i in range(num_envs):
                clip = ImageSequenceClip([img[i] for img in imgs], fps=10)
                clip.write_videofile(os.path.join(eval_dir, "eval{:d}.mp4".format(i)))
            for inner_env in eval_env.venv.envs:
                inner_env.multi_env.world.close()
        return rets

    def callback(_locals, _globals):
        nonlocal n_steps, best_mean
        model = _locals["self"]
        if (n_steps + 1) % eval_save_period == 0:
            start_eval_time = time.time()
            eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
            os.makedirs(eval_dir)
            rets = evaluate(model, eval_dir)
            avg_ret = np.mean(rets)
            with open(rets_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([n_steps] + [ret for ret in rets])
            if avg_ret > best_mean:
                best_mean = avg_ret
                shutil.rmtree(best_dir)
                shutil.copytree(eval_dir, best_dir)
            end_eval_time = time.time() - start_eval_time
            print("Finished evaluation in {:.2f} seconds".format(end_eval_time))
        n_steps += 1
        return True

    model.learn(total_timesteps=timesteps, callback=callback)
    rets = evaluate(model, final_dir)
    with open(rets_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([-1] + [ret for ret in rets])


if __name__ == "__main__":
    wandb.init(project="hr_adaptation", sync_tensorboard=True)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train(FLAGS.name, FLAGS.logdir)
