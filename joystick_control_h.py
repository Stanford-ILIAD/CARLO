"""Joystick control the human car."""
import os
import time
from absl import app, flags
import gym
import pygame
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv
import wandb
import driving_envs  # pylint: disable=unused-import
from generate_videos import get_best_step
from single_agent_env import VecSingleEnv
from joystick_utils import LEFT_Y_AXIS, RIGHT_X_AXIS, TURN_SCALING, ACC_SCALING

FLAGS = flags.FLAGS
# NOTE: generate_videos.py already defines the run_id flag.
# flags.DEFINE_string("run_id", None, "Wandb experiment run id.")


class JoystickPolicy:
    """Takes actions according to a pre-trained behavior cloning model."""

    def __init__(self, joystick):
        self.joystick = joystick

    def action(self, obs):
        del obs
        pygame.event.pump()
        joystick = self.joystick
        return (
            TURN_SCALING * joystick.get_axis(RIGHT_X_AXIS),
            ACC_SCALING * joystick.get_axis(LEFT_Y_AXIS),
        )

    def reset(self):
        pass


def main(_argv):
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    assert len(joysticks) >= 1, "Need a joystick to be connected."
    joystick = joysticks[0]
    joystick.init()

    api = wandb.Api()
    run_path = "ayzhong/hr_adaptation/" + FLAGS.run_id
    local_dir = "/tmp/{}".format(os.path.basename(run_path))
    run = api.run(run_path)
    best_step = get_best_step(run, local_dir)
    print("Using best step: {}.".format(best_step))
    eval_folder = "ppo_driving/eval{}".format(best_step)
    model_path = os.path.join(eval_folder, "model.pkl")
    obs_rms_path = os.path.join(eval_folder, "obs_rms.pkl")
    ret_rms_path = os.path.join(eval_folder, "ret_rms.pkl")
    model_file = run.file(model_path)
    obs_rms_file = run.file(obs_rms_path)
    ret_rms_file = run.file(ret_rms_path)
    model_file.download(root=local_dir, replace=True)
    obs_rms_file.download(root=local_dir, replace=True)
    ret_rms_file.download(root=local_dir, replace=True)

    local_eval_path = os.path.join(local_dir, eval_folder)
    num_envs = 8
    model = PPO2.load(os.path.join(local_eval_path, "model.pkl"))
    env_fns = num_envs * [lambda: gym.make("Merging-v0")]
    human_policy = JoystickPolicy(joystick)
    env = VecNormalize(VecSingleEnv(SubprocVecEnv(env_fns), human_policies=[human_policy]))
    env.load_running_average(local_eval_path)

    done = False
    obs = env.reset()
    episode_data = []
    state, dones = None, [False for _ in range(num_envs)]

    multi_env = gym.make("Merging-v0")
    multi_env.reset()
    multi_env.state = env.get_attr("state")[0]
    multi_env.update_text()
    multi_env.render()

    time.sleep(5)

    for i in range(60):
        print(i)
        r_action, state = model.predict(obs, state=state, mask=dones)
        next_obs, rew, dones, _debug = env.step(r_action)
        multi_env.state = env.get_attr("state")[0]
        multi_env.update_text()
        multi_env.render()
        episode_data.append((obs, r_action, rew, next_obs, done))
        obs = next_obs
        time.sleep(0.1)
    return


if __name__ == "__main__":
    app.run(main)
