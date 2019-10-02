"""Joystick control the human car."""
import os
import pickle
import time
from absl import app, flags
import gym
import pygame
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv
import driving_envs  # pylint: disable=unused-import
from driving_envs.graphics import Point, Text
from single_agent_env import VecSingleEnv
from joystick_utils import (
    LEFT_Y_AXIS,
    RIGHT_X_AXIS,
    TURN_SCALING,
    ACC_SCALING,
    BUTTON_A,
    BUTTON_B,
    display_countdown,
    draw_text,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt_folder", None, "Path to checkpoint folder.")
flags.DEFINE_string("out_path", None, "Path to save generated data in.")


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

    num_envs = 8
    model = PPO2.load(os.path.join(FLAGS.ckpt_folder, "model.pkl"))
    env_fns = num_envs * [lambda: gym.make("Merging-v0")]
    human_policy = JoystickPolicy(joystick)
    env = VecNormalize(VecSingleEnv(SubprocVecEnv(env_fns), human_policies=[human_policy]))
    env.load_running_average(FLAGS.ckpt_folder)

    multi_env = gym.make("Merging-v0")

    eval_data = []
    for eval_idx in range(10):
        obs = env.reset()
        state, dones = None, [False for _ in range(num_envs)]

        done = False
        multi_env.reset()
        multi_env.state = env.get_attr("state")[0]
        multi_env.update_text()
        multi_env.render()
        eval_type = "A" if eval_idx < 5 else "B"
        txt = draw_text(multi_env.world.visualizer, 60, 60, "Eval type: {}".format(eval_type))
        time.sleep(2)
        txt.undraw()
        display_countdown(multi_env, start_delay=5)
        ep_data = []
        while not done:
            true_states = env.get_attr("state")
            r_action, state = model.predict(obs, state=state, mask=dones)
            next_obs, rewards, dones, _debug = env.step(r_action)
            done = dones[0]
            ep_data.append((true_states[0], rewards[0], dones[0]))
            multi_env.state = env.get_attr("state")[0]
            multi_env.update_text()
            multi_env.render()
            obs = next_obs
            time.sleep(0.1)
        if len(ep_data) != 60:
            print("Warning: episode length: {}.".format(len(ep_data)))
        eval_data.append({"eval_type": eval_type, "ep_data": ep_data})
    multi_env.world.close()
    with open(FLAGS.out_path, "wb") as f:
        pickle.dump(eval_data, f)
    print("Saved data successfully.")


if __name__ == "__main__":
    flags.mark_flag_as_required("ckpt_folder")
    flags.mark_flag_as_required("out_path")
    app.run(main)
