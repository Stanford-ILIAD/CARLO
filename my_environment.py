import time

import numpy as np

from world import World
from agents import Car, RectangleBuilding
from geometry import Point


dt = 0.1

world = World(dt=dt, width=150, height=150, ppm=6)

# >>> Map
# right side:
world.add(RectangleBuilding(Point(125, 75), Point(50, 155)))
world.add(RectangleBuilding(Point(75, 125), Point(62.5, 50)))

# left side:
world.add(RectangleBuilding(Point(37.5, 37.5), Point(100, 100)))
world.add(RectangleBuilding(Point(12.5, 125), Point(37.5, 100)))

# up - 40, 145; down - 93.75, 5
car = Car(Point(37.5, 145), (3 * np.pi / 2), 'yellow')
car.velocity = Point(3.0, 0)
world.add(car)

world.render()


# >>> Algorithm (simple)
car.set_control(0, 0.05)

for time_stp in range(400):
    # turn to the right in the direction of
    if time_stp == 153:
        car.set_control(0.45, 0.3)
    if time_stp == 193:
        car.set_control(0, 0.3)

    # turn to the left in the direction of
    if time_stp == 273:
        car.set_control(-0.45, 0.3)
    if time_stp == 295:
        car.set_control(0, 0.3)

    # acceleration
    if time_stp == 298:
        car.set_control(0, 1.5)

    world.tick()
    world.render()

    time.sleep(dt / 4)

world.close()

# >>> Algorithm (Q-learning)
# TODO
