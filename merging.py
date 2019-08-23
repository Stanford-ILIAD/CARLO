from typing import Dict, Text, Tuple
import gym
import numpy as np
from world import World
from agents import Car, Building, Pedestrian, Painting
from geometry import Point
import time


class DrivingEnv(gym.Env):
    """Driving gym interface."""

    def __init__(self, dt: float = 0.04):
        self.dt = dt
        self.world = World(self.dt, width=120, height=120, ppm=6)
        self.buildings = [
            Building(Point(28.5, 60), Point(57, 120), "gray80"),
            Building(Point(91.5, 60), Point(57, 120), "gray80"),
        ]
        for building in self.buildings:
            self.world.add(building)
        self.cars = {
            "H": Car(Point(59.5, 25), np.pi / 2),
            "R": Car(Point(60.5, 20), np.pi / 2, "blue"),
        }
        self.world.add(self.cars["H"])
        self.world.add(self.cars["R"])

    def step(self, action: Dict[Text, Tuple[float, float]]):
        for car_name, car_action in action.items():
            self.cars[car_name].set_control(*car_action)
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
        return self._get_state(), reward, done, {}

    def _get_car_reward(self, name: Text):
        car = self.cars[name]
        forward_vel = car.velocity.y
        control_cost = -np.square(car.inputAcceleration)
        return 0.2 * forward_vel - control_cost

    def _get_state(self) -> np.ndarray:
        return np.concatenate((self.cars["H"].state, self.cars["R"].state))

    def reset(self):
        self.cars["H"].velocity = Point(0, 6)
        self.cars["R"].velocity = Point(0, 6)
        return self._get_state()

    def render(self, mode="human"):
        self.world.render()


def main():
    env = DrivingEnv()
    done = False
    obs = env.reset()
    env.render()
    episode_data = []
    while not done:
        action = {"H": (0, 0), "R": (0, 0)}
        next_obs, rew, done, debug = env.step(action)
        del debug
        episode_data.append((obs, action, rew, next_obs, done))
        obs = next_obs
        env.render()
        time.sleep(env.dt)
    return


if __name__ == "__main__":
    main()
