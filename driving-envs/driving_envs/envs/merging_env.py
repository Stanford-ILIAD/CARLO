import io
from typing import Text
import gym
from PIL import Image
import numpy as np
import scipy.special
from driving_envs.world import World
from driving_envs.entities import TextEntity
from driving_envs.agents import Car, Building
from driving_envs.geometry import Point

expit = scipy.special.expit


class MergingEnv(gym.Env):
    """Driving gym interface."""

    def __init__(
        self,
        dt: float = 0.1,
        width: int = 120,
        height: int = 120,
        ctrl_cost_weight: float = 0.0,
        time_limit: int = 60,
        random_initial: bool = False,
    ):
        super(MergingEnv, self).__init__()
        self.dt, self.width, self.height = dt, width, height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.r_speed = TextEntity(Point(75, 5))
        self.h_speed = TextEntity(Point(75, 10))
        self.buildings, self.cars = [], {}
        self._ctrl_cost_weight = ctrl_cost_weight
        self.time_limit = time_limit
        self.randomize_initial_state = random_initial

    def step(self, action: np.ndarray):
        self.step_num += 1
        offset = 0
        for agent in self.world.dynamic_agents:
            agent.set_control(*action[offset : offset + 2])
            offset += 2
        self.world.tick()  # This ticks the world for one time step (dt second)
        done = False
        reward = {name: self._get_car_reward(name) for name in self.cars.keys()}
        if self.cars["R"].collidesWith(self.cars["H"]):
            done = True
        for car_name, car in self.cars.items():
            for building in self.buildings:
                if car.collidesWith(building):
                    done = True
            if car_name == "R" and car.y >= self.height or car.y <= 0:
                raise ValueError("Car went out of bounds!")
        self.update_text()
        if self.step_num >= self.time_limit:
            done = True
        return self._get_obs(), reward, done, {}

    def update_text(self):
        self.r_speed.text = "R speed: {:.1f}".format(self.cars["R"].speed)
        self.h_speed.text = "H speed: {:.1f}".format(self.cars["H"].speed)

    def _get_obs(self):
        return np.concatenate((self.world.state[:6], self.world.state[7:13]))

    def _get_car_reward(self, name: Text):
        car = self.cars[name]
        vel_rew = 0.1 * car.velocity.y
        right_lane_cost = 0.3 * expit((car.y - 60) / 5) * max(car.x - 59, 0)
        # right_lane_cost = .1 * max(car.x - 59, 0)
        control_cost = np.square(car.inputAcceleration)
        return vel_rew - right_lane_cost - self._ctrl_cost_weight * control_cost

    def reset(self):
        self.world.reset()
        self.buildings = [
            Building(Point(28.5, 60), Point(57, 120), "gray80"),
            Building(Point(91.5, 60), Point(57, 120), "gray80"),
            Building(Point(62, 90), Point(2, 60), "gray80"),
        ]
        h_y, r_y = 5, 5
        if self.randomize_initial_state:
            h_y = np.random.uniform(4, 6)
            r_y = np.random.uniform(4, 6)
        self.cars = {
            "H": Car(Point(58.5, h_y), np.pi / 2),
            "R": Car(Point(61.5, r_y), np.pi / 2, "blue"),
        }
        for building in self.buildings:
            self.world.add(building)
        # NOTE: Order that dynamic agents are added to world determines
        # the concatenated state and action representation.
        self.world.add(self.cars["H"])
        self.world.add(self.cars["R"])
        h_yvel, r_yvel = 10, 10
        if self.randomize_initial_state:
            h_yvel = np.random.uniform(9.5, 10.5)
            r_yvel = np.random.uniform(9.5, 10.5)
        self.cars["H"].velocity = Point(0, h_yvel)
        self.cars["R"].velocity = Point(0, r_yvel)
        self.step_num = 0
        self.world.add(self.r_speed)
        self.world.add(self.h_speed)
        self.update_text()
        return self._get_obs()

    def render(self, mode="human"):
        self.world.render()
        if mode == "rgb_array":
            cnv = self.world.visualizer.win
            ps = cnv.postscript(colormode="color")
            img = Image.open(io.BytesIO(ps.encode("utf-8")))
            return np.array(img)
