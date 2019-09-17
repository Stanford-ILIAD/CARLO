import itertools
import math
import time
from typing import List, Optional, Tuple
import gin
import gym
from gym import spaces
import driving_envs  # pylint: disable=unused-import
import numpy as np


class PidPosPolicy:
    """PID controller that maintains fixed distance in front of R."""

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
        my_y, their_y = obs[1], obs[7]
        my_y_dot, their_y_dot = obs[3], obs[9]
        if their_y > my_y + 2:
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


class PidVelPolicy:
    """PID controller that maintains a fixed velocity."""

    def __init__(
        self,
        dt: float,
        target_vel: float,
        params: Tuple[float, float, float] = (3.0, 1.0, 6.0),
    ):
        self._target_vel = target_vel
        self.previous_error = 0
        self.integral = 0
        self.errors = []
        self.dt = dt
        self.Kp, self.Ki, self.Kd = params

    def action(self, obs):
        my_y_dot = obs[3]
        error = self._target_vel - my_y_dot
        derivative = (error - self.previous_error) * self.dt
        self.integral = self.integral + self.dt * error
        acc = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.errors.append(error)
        return np.array((0, acc))

    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.errors = []


class PidSingleEnv(gym.Env):
    """Wrapper that turns multi-agent driving env into single agent, using simulated human."""

    def __init__(self, multi_env, discrete: bool = False, human_policies=None):
        self.multi_env = multi_env
        self.discrete = discrete
        if human_policies:
            self.human_policies = human_policies
        else:
            self.human_policies = [
                PidPosPolicy(self.multi_env.dt, 10, 3.5, np.inf),
                PidVelPolicy(self.multi_env.dt, 10),
            ]
        if discrete:
            self.num_bins = (5, 5)
            self.binner = [np.linspace(-1, 1, num=n) for n in self.num_bins]
            self.action_space = spaces.Discrete(int(np.prod(self.num_bins)))
            self.int_to_tuple = list(itertools.product(*[range(x) for x in self.num_bins]))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,))
        else:
            self.action_space = spaces.Box(np.array((-1.0, -1.0)), np.array((1.0, 1.0)))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,))

    def step(self, action):
        if self.discrete:
            action = self.int_to_tuple[action]
            action = np.array([self.binner[i][a] for i, a in enumerate(action)])
        processed_action = np.array((action[0] * 0.1, action[1] * 4))
        h_action = self._pid_human.action(self.previous_obs)
        multi_action = np.concatenate((h_action, processed_action))
        obs, rew, done, debug = self.multi_env.step(multi_action)
        self.previous_obs = obs
        return obs, rew["R"], done, debug

    def reset(self):
        self._pid_human = np.random.choice(self.human_policies)
        obs = self.multi_env.reset()
        self.previous_obs = obs
        return obs

    def render(self, mode="human"):
        return self.multi_env.render(mode=mode)


@gin.configurable
def make_single_env(name="Merging-v1", **kwargs):
    multi_env = gym.make(name)
    env = PidSingleEnv(multi_env, **kwargs)
    return env


def main():
    env = make_single_env(name="Merging-v1")
    done = False
    obs = env.reset()
    env.render()
    episode_data = []
    i = 0
    ret = 0
    while not done:
        action = (0, 4)
        next_obs, rew, done, debug = env.step(action)
        ret += rew
        del debug
        episode_data.append((obs, action, rew, next_obs, done))
        obs = next_obs
        env.render()
        time.sleep(env.multi_env.dt)
        i += 1
    print("i: {}, Return: {}".format(i, ret))
    env.multi_env.world.close()


if __name__ == "__main__":
    main()
