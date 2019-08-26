from typing import Dict, Text, Tuple
import gym
import numpy as np
from world import World
from agents import Car, Building, Pedestrian, Painting
from geometry import Point
import time


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


class DrivingEnv(gym.Env):
    """Driving gym interface."""

    def __init__(self, dt: float = 0.04, width: int = 120, height: int = 120):
        self.dt, self.width, self.height = dt, width, height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings = [
            Building(Point(28.5, 60), Point(57, 120), "gray80"),
            Building(Point(91.5, 50), Point(57, 100), "gray80"),
            Building(Point(90, 110), Point(60, 20), "gray80"),
        ]
        self.cars = {
            "H": Car(Point(58.5, 10), np.pi / 2),
            "R": Car(Point(61.5, 5), np.pi / 2, "blue"),
        }
        for building in self.buildings:
            self.world.add(building)
        # NOTE: Order that dynamic agents are added to world determines
        # the concatenated state and action representation.
        self.world.add(self.cars["H"])
        self.world.add(self.cars["R"])

    def step(self, action: np.ndarray):
        offset = 0
        for agent in self.world.dynamic_agents:
            agent.set_control(*action[offset : offset + 2])
            offset += 2
        self.world.tick()  # This ticks the world for one time step (dt second)
        done = False
        reward = {name: self._get_car_reward(name) for name in self.cars.keys()}
        if self.cars["R"].collidesWith(self.cars["H"]):
            reward["H"] -= 100
            reward["R"] -= 100
            done = True
        for car_name, car in self.cars.items():
            for building in self.buildings:
                if car.collidesWith(building):
                    reward[car_name] -= 100
                    done = True
            if car.y >= self.height:
                done = True
        return self.world.state, reward, done, {}

    def _get_car_reward(self, name: Text):
        car = self.cars[name]
        forward_vel = car.velocity.y
        control_cost = -np.square(car.inputAcceleration)
        return 0.2 * forward_vel - control_cost

    def reset(self):
        self.cars["H"].velocity = Point(0, 7)
        self.cars["R"].velocity = Point(0, 7)
        return self.world.state

    def render(self, mode="human"):
        self.world.render()


def main():
    env = DrivingEnv()
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
