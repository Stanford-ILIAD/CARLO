import time
from typing import Tuple
import gym
import driving_envs  # pylint: disable=unused-import
import numpy as np


class PidAgent:
    """PID controller."""

    def __init__(
        self,
        dt: float,
        target_dist: float,
        max_acc: float,
        max_vel: float,
        params: Tuple[float, float, float] = (3.0, 0.0, 6.0),
    ):
        self._target_dist = target_dist
        self._max_acc = max_acc
        self._max_vel = max_vel
        self.integral = 0
        self.errors = []
        self.dt = dt
        self.Kp, self.Ki, self.Kd = params

    def action(self, obs):
        # Assume that the agent is the Human.
        my_y, their_y = obs[1], obs[8]
        my_y_dot, their_y_dot = obs[3], obs[10]
        if their_y > my_y:
            target = their_y - self._target_dist
        else:
            target = their_y + self._target_dist
        error = target - my_y
        derivative = their_y_dot - my_y_dot
        self.integral = self.integral + self.dt * error
        acc = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        acc = np.clip(acc, -np.inf, self._max_acc)
        if my_y_dot >= self._max_vel:
            acc = 0
        self.errors.append(error)
        return np.array((0, acc))

    def reset(self):
        self.integral = 0
        self.errors = []


def main():
    env = gym.make("Merging-v0")
    done = False
    obs = env.reset()
    env.render()
    episode_data = []
    human = PidAgent(env.dt, 8, 2.5, 10)
    human.reset()
    while not done:
        h_action = human.action(obs)
        r_action = np.array((0.0, 4.0))
        action = np.concatenate((h_action, r_action))
        next_obs, rew, done, debug = env.step(action)
        del debug
        episode_data.append((obs, action, rew, next_obs, done))
        obs = next_obs
        env.render()
        time.sleep(env.dt)
    env.world.close()
    return


if __name__ == "__main__":
    main()