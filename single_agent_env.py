import time
from typing import Tuple
import gym
from gym import spaces
import driving_envs  # pylint: disable=unused-import
import numpy as np


class PidPolicy:
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


class PidSingleEnv(gym.Env):
    """Wrapper that turns multi-agent driving env into single agent, using simulated human."""

    def __init__(self, multi_env):
        self.multi_env = multi_env
        self._pid_human = PidPolicy(multi_env.dt, 8.0, 4.0, 12.0)
        self.action_space = spaces.Box(np.array((-np.pi, -4.)), np.array((np.pi, 4.)))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))

    def step(self, action):
        h_action = self._pid_human.action(self.previous_obs)
        multi_action = np.concatenate((h_action, action))
        obs, rew, done, debug = self.multi_env.step(multi_action)
        self.previous_obs = obs
        return obs, rew["R"], done, debug

    def reset(self):
        obs = self.multi_env.reset()
        self.previous_obs = obs
        return obs

    def render(self, mode="human"):
        return self.multi_env.render(mode=mode)


def main():
    multi_env = gym.make("Merging-v0")
    env = PidSingleEnv(multi_env)
    done = False
    obs = env.reset()
    env.render()
    episode_data = []
    while not done:
        action = np.array((0.0, 4.0))
        next_obs, rew, done, debug = env.step(action)
        del debug
        episode_data.append((obs, action, rew, next_obs, done))
        obs = next_obs
        env.render()
        time.sleep(env.multi_env.dt)
    env.multi_env.world.close()
    return


if __name__ == "__main__":
    main()
