"""Defines a single agent env where the robot is the only agent (human is part of env)."""
import itertools
import time
from typing import Tuple
import gin
import gym
from gym import spaces
import numpy as np
from tensorflow.keras.models import load_model
import driving_envs  # pylint: disable=unused-import


class PidPosPolicy:
    """PID controller for H that maintains fixed distance in front of R."""

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
    """PID controller for H that maintains its initial velocity."""

    def __init__(self, dt: float, params: Tuple[float, float, float] = (3.0, 1.0, 6.0)):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []
        self.dt = dt
        self.Kp, self.Ki, self.Kd = params

    def action(self, obs):
        my_y_dot = obs[3]
        if self._target_vel is None:
            self._target_vel = my_y_dot
        error = self._target_vel - my_y_dot
        derivative = (error - self.previous_error) * self.dt
        self.integral = self.integral + self.dt * error
        acc = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.errors.append(error)
        return np.array((0, acc))

    def reset(self):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []

    def __str__(self):
        return "PidVelPolicy({})".format(self.dt)


class BCPolicy:
    """Takes actions according to a pre-trained behavior cloning model."""

    def __init__(self, path):
        self.model = load_model(path)

    def action(self, obs):
        return self.model.predict(obs[None])[0]

    def reset(self):
        pass


def get_human_policies(mode, dt):
    """Helper that returns a list of human policies based on mode."""
    if mode == "fixed_1":
        return [PidVelPolicy(dt)]
    elif mode == "fixed_2":
        return [PidPosPolicy(dt, 10, 4.0, np.inf), PidVelPolicy(dt)]
    elif mode == "random_10":
        vel_pols = [PidVelPolicy(dt) for _ in range(5)]
        target_accs = np.random.uniform(3.9, 4, size=5)
        pos_pols = [PidPosPolicy(dt, 10, target_acc, np.inf) for target_acc in target_accs]
        return vel_pols + pos_pols
    elif mode == "bc_2":
        return [BCPolicy("bc_weights/typeA.h5"), BCPolicy("bc_weights/typeB.h5")]
    else:
        raise ValueError("Unrecognized mode {}".format(mode))


class PidSingleEnv(gym.Env):
    """Wrapper that turns multi-agent driving env into a single agent env (the robot)."""

    def __init__(self, human_policies=None, discrete: bool = False, random=True, **kwargs):
        self.multi_env = gym.make("Merging-v0", **kwargs)
        self.human_policies = human_policies
        self.discrete = discrete
        self.random = random
        self._policy_idx = 0
        self._human_pol = None
        self.previous_obs = None
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
        h_action = self._human_pol.action(self.previous_obs)
        multi_action = np.concatenate((h_action, processed_action))
        obs, rew, done, debug = self.multi_env.step(multi_action)
        self.previous_obs = obs
        return obs, rew["R"], done, debug

    def reset(self):
        if self.random:
            self._human_pol = np.random.choice(self.human_policies)
        else:
            self._human_pol = self.human_policies[self._policy_idx]
            self._policy_idx = (self._policy_idx + 1) % len(self.human_policies)
        obs = self.multi_env.reset()
        self.previous_obs = obs
        return obs

    def render(self, mode="human"):
        return self.multi_env.render(mode=mode)


@gin.configurable
def make_single_env(human_mode=None, **kwargs):
    """Helper function to create the human policies and then the single agent env."""
    if human_mode is not None:
        assert "human_policies" not in kwargs, "Set one of human_mode or human_policies."
        human_policies = get_human_policies(human_mode, 0.1)
        kwargs["human_policies"] = human_policies
    else:
        assert "human_policies" in kwargs, "Set one of human_mode or human_policies."
    env = PidSingleEnv(**kwargs)
    return env


def main():
    """Test that the single agent env works."""
    env = make_single_env()
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
