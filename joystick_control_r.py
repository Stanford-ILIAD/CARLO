"""Joystick control the robot car, while the human car is controlled by BC policies."""
import time
import pygame
import gym
from stable_baselines.common.vec_env import DummyVecEnv
import driving_envs  # pylint: disable=unused-import
from single_agent_env import VecSingleEnv, BCPolicy


LEFT_Y_AXIS = 1
RIGHT_X_AXIS = 3


def main():
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    assert len(joysticks) >= 1, "Need a joystick to be connected."
    joystick = joysticks[0]
    joystick.init()
    multi_env = gym.make("Merging-v0")
    env = VecSingleEnv(
        DummyVecEnv([lambda: multi_env]),
        human_policies=[BCPolicy("bc_weights/typeA.h5"), BCPolicy("bc_weights/typeB.h5")],
        discrete=False,
    )
    for _ in range(10):
        done = False
        env.reset()
        multi_env.render()
        print("Starting in 5!")
        time.sleep(5)
        while not done:
            pygame.event.pump()
            action = (
                -0.05 * joystick.get_axis(RIGHT_X_AXIS),
                -4 * joystick.get_axis(LEFT_Y_AXIS),
            )
            env.step([action])
            multi_env.render()
            time.sleep(multi_env.dt)
    multi_env.world.close()
    return


if __name__ == "__main__":
    main()
