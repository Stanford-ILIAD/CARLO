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

    def __str__(self):
        return "PidPosPolicy({},{}, {}, {})".format(
            self.dt, self._target_dist, self._max_acc, self._max_vel
        )


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

    def __str__(self):
        return "PidVelPolicy({},{})".format(self.dt, self._target_vel)


def get_human_policies(mode, dt):
    if mode == "fixed_1":
        return [PidVelPolicy(dt, 10)]
    elif mode == "fixed_2":
        return [PidPosPolicy(dt, 10, 3.5, np.inf), PidVelPolicy(dt, 10)]
    elif mode == "random_10":
        target_vels = 2 * np.random.rand(5) + 10  # 5 samples from Uniform([10, 12])
        vel_pols = [PidVelPolicy(dt, target_vel) for target_vel in target_vels]
        target_accs = 4 * np.random.rand(5)  # 5 samples from Uniform([0, 4])
        pos_pols = [PidPosPolicy(dt, 10, target_acc, np.inf) for target_acc in target_accs]
        return vel_pols + pos_pols
    elif mode == "interp_10":
        target_vels = np.linspace(10, 12, num=5)
        vel_pols = [PidVelPolicy(dt, target_vel) for target_vel in target_vels]
        target_accs = np.linspace(0, 4, num=5)
        pos_pols = [PidPosPolicy(dt, 10, target_acc, np.inf) for target_acc in target_accs]
        return vel_pols + pos_pols
    else:
        raise ValueError("Unrecognized mode {}".format(mode))


class PidSingleEnv(gym.Env):
    """Wrapper that turns multi-agent driving env into single agent, using simulated human."""

    def __init__(self, multi_env, human_policies=None, discrete: bool = False, random=True):
        self.multi_env = multi_env
        self.human_policies = human_policies
        self.discrete = discrete
        self.random = random
        self._policy_idx = 0
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
        if self.random:
            self._pid_human = np.random.choice(self.human_policies)
        else:
            self._pid_human = self.human_policies[self._policy_idx]
            self._policy_idx = (self._policy_idx + 1) % len(self.human_policies)
        obs = self.multi_env.reset()
        self.previous_obs = obs
        return obs

    def render(self, mode="human"):
        return self.multi_env.render(mode=mode)


@gin.configurable
def make_single_env(name="Merging-v1", human_mode=None, **kwargs):
    multi_env = gym.make(name)
    if human_mode is not None:
        assert "human_policies" not in kwargs, "Set one of human_mode or human_policies."
        human_policies = get_human_policies(human_mode, multi_env.dt)
        kwargs["human_policies"] = human_policies
    else:
        assert "human_policies" in kwargs, "Set one of human_mode or human_policies."
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
