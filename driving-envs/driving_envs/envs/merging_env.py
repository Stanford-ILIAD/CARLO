from typing import Dict, Text, Tuple
import gym
import numpy as np
from driving_envs.world import World
from driving_envs.agents import Car, Building, Pedestrian, Painting
from driving_envs.geometry import Point
import time


class MergingEnv(gym.Env):
    """Driving gym interface."""

    def __init__(self, dt: float = 0.04, width: int = 120, height: int = 120):
        super(MergingEnv, self).__init__()
        self.dt, self.width, self.height = dt, width, height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.buildings, self.cars = [], {}

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
        self.world.reset()
        self.buildings = [
            Building(Point(28.5, 60), Point(57, 120), "gray80"),
            Building(Point(91.5, 50), Point(57, 100), "gray80"),
            Building(Point(90, 110), Point(60, 20), "gray80"),
        ]
        self.cars = {
            "H": Car(Point(58.5, 10), np.pi / 2),
            "R": Car(Point(58.5, 5), np.pi / 2, "blue"),
        }
        for building in self.buildings:
            self.world.add(building)
        # NOTE: Order that dynamic agents are added to world determines
        # the concatenated state and action representation.
        self.world.add(self.cars["H"])
        self.world.add(self.cars["R"])
        self.cars["H"].velocity = Point(0, 7)
        self.cars["R"].velocity = Point(0, 7)
        return self.world.state

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Unsupported mode: {}".format(mode))
        self.world.render()
