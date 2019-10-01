"Collect demonstrations from 2 joystick controllers."
import os
import time
from absl import app, flags
import gym
import numpy as np
import pygame
import driving_envs  # pylint: disable=unused-import
from joystick_utils import display_countdown
from joystick_utils import LEFT_Y_AXIS, RIGHT_X_AXIS, TURN_SCALING, ACC_SCALING

FLAGS = flags.FLAGS
flags.DEFINE_string("folder", "demos", "Folder to save file in.")


def main(_argv):
    folder = FLAGS.folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    if len(joysticks) != 2:
        raise Exception("Need 2 joysticks connected")
    for joystick in joysticks:
        joystick.init()
    multi_env = gym.make("Merging-v0")
    done = False
    _obs = multi_env.reset()
    multi_env.render()
    display_countdown(multi_env, start_delay=5)
    i = 0
    demonstration_data = []
    while not done:
        state = multi_env.world.state
        pygame.event.pump()
        r_action = (
            TURN_SCALING * joysticks[0].get_axis(RIGHT_X_AXIS),
            ACC_SCALING * joysticks[0].get_axis(LEFT_Y_AXIS),
        )
        # Human stays in lane and does not turn.
        h_action = (0, ACC_SCALING * joysticks[1].get_axis(LEFT_Y_AXIS))
        action = np.array(h_action + r_action)
        _obs, _rew, done, _debug = multi_env.step(action)
        demonstration_data.append((state, action))
        multi_env.render()
        time.sleep(multi_env.dt)
        i += 1
    multi_env.world.close()
    states, actions = [np.stack(x) for x in zip(*demonstration_data)]
    if len(demonstration_data) != 60:
        print(
            "WARNING: len(demonstration_data)=={}. Not saving data.".format(
                len(demonstration_data)
            )
        )
    else:
        time_in_secs = int(time.time())
        np.savez(
            os.path.join(folder, "{}.npz".format(time_in_secs)), states=states, actions=actions
        )


if __name__ == "__main__":
    app.run(main)
