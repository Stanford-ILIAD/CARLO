import itertools
import math
import time
from typing import Tuple
import gin
import gym
from gym import spaces
import driving_envs  # pylint: disable=unused-import
from driving_envs.graphics import Transform
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


@gin.configurable
class PidSingleEnv(gym.Env):
    """Wrapper that turns multi-agent driving env into single agent, using simulated human."""

    def __init__(self, multi_env, discrete: bool = False, human_max_accs=[2.3, 3.5]):
        self.multi_env = multi_env
        self.discrete = discrete
        self.human_max_accs = human_max_accs
        if discrete:
            self.num_bins = (11, 11)
            self.binner = [np.linspace(-1, 1, num=n) for n in self.num_bins]
            self.action_space = spaces.Discrete(int(np.prod(self.num_bins)))
            self.int_to_tuple = list(itertools.product(*[range(x) for x in self.num_bins]))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))
        else:
            self.action_space = spaces.Box(np.array((-1.0, -1.0)), np.array((1.0, 1.0)))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,))

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
        max_acc = np.random.choice(self.human_max_accs)
        max_vel = {2.3: 12, 3.5: 15}[max_acc]
        self._pid_human = PidPolicy(self.multi_env.dt, 10, max_acc, max_vel)
        obs = self.multi_env.reset()
        self.previous_obs = obs
        return obs

    def render(self, mode="human"):
        return self.multi_env.render(mode=mode)


@gin.configurable
def make_single_env(name="Merging-v1", discrete=False):
    multi_env = gym.make(name)
    env = PidSingleEnv(multi_env, discrete=discrete)
    return env


if __name__ == "__main__":
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
