import numpy as np
from geometry import Point, Rectangle, Circle
from typing import Union
import copy


class Entity:
    def __init__(
        self, center: Point, heading: float, movable: bool = True, friction: float = 0
    ):
        self.center = center  # this is x, y
        self.heading = heading
        self.movable = movable
        self.color = "ghost white"
        self.collidable = True
        if movable:
            self.friction = friction
            self.velocity = Point(0, 0)  # this is xp, yp
            self.acceleration = 0  # this is vp (or speedp)
            self.angular_velocity = 0  # this is headingp
            self.inputSteering = 0
            self.inputAcceleration = 0
            self.max_speed = np.inf
            self.min_speed = 0

    @property
    def speed(self) -> float:
        return self.velocity.norm(p=2) if self.movable else 0

    def set_control(self, inputSteering: float, inputAcceleration: float):
        self.inputSteering = inputSteering
        self.inputAcceleration = inputAcceleration

    def tick(self, dt: float):
        if self.movable:
            speed = self.speed
            heading = self.heading

            new_angular_velocity = speed * self.inputSteering
            new_acceleration = self.inputAcceleration - self.friction * speed

            new_heading = (
                heading + (self.angular_velocity + new_angular_velocity) * dt / 2.0
            )
            new_speed = np.clip(
                speed + (self.acceleration + new_acceleration) * dt / 2.0,
                self.min_speed,
                self.max_speed,
            )

            new_velocity = Point(
                ((speed + new_speed) / 2.0) * np.cos((new_heading + heading) / 2.0),
                ((speed + new_speed) / 2.0) * np.sin((new_heading + heading) / 2.0),
            )

            new_center = self.center + (self.velocity + new_velocity) * dt / 2.0

            self.center = new_center
            self.heading = new_heading
            self.velocity = new_velocity
            self.acceleration = new_acceleration
            self.angular_velocity = new_angular_velocity

            self.buildGeometry()

    def collidesWith(self, other) -> bool:
        raise NotImplementedError

    def buildGeometry(self):  # builds the obj
        raise NotImplementedError

    def collidesWith(self, other: Union["Point", "Entity"]) -> bool:
        if isinstance(other, Entity):
            return self.obj.intersectsWith(other.obj)
        elif isinstance(other, Point):
            return self.obj.intersectsWith(other)
        else:
            raise NotImplementedError

    def distanceTo(self, other: Union["Point", "Entity"]) -> float:
        if isinstance(other, Entity):
            return self.obj.distanceTo(other.obj)
        elif isinstance(other, Point):
            return self.obj.distanceTo(other)
        else:
            raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y

    @property
    def xp(self):
        return self.velocity.x

    @property
    def yp(self):
        return self.velocity.y


class RectangleEntity(Entity):
    def __init__(
        self,
        center: Point,
        heading: float,
        size: Point,
        movable: bool = True,
        friction: float = 0,
    ):
        super(RectangleEntity, self).__init__(center, heading, movable, friction)
        self.size = size
        self.buildGeometry()

    @property
    def edge_centers(self):
        edge_centers = np.zeros((4, 2), dtype=np.float32)
        x = self.center.x
        y = self.center.y
        w = self.size.x
        h = self.size.y
        edge_centers[0] = [
            x + w / 2.0 * np.cos(self.heading),
            y + w / 2.0 * np.sin(self.heading),
        ]
        edge_centers[1] = [
            x - h / 2.0 * np.sin(self.heading),
            y + h / 2.0 * np.cos(self.heading),
        ]
        edge_centers[2] = [
            x - w / 2.0 * np.cos(self.heading),
            y - w / 2.0 * np.sin(self.heading),
        ]
        edge_centers[3] = [
            x + h / 2.0 * np.sin(self.heading),
            y - h / 2.0 * np.cos(self.heading),
        ]
        return edge_centers

    @property
    def corners(self):
        ec = self.edge_centers
        c = np.array([self.center.x, self.center.y])
        corners = []
        corners.append(Point(*(ec[1] + ec[0] - c)))
        corners.append(Point(*(ec[2] + ec[1] - c)))
        corners.append(Point(*(ec[3] + ec[2] - c)))
        corners.append(Point(*(ec[0] + ec[3] - c)))
        return corners

    def buildGeometry(self):
        C = self.corners
        self.obj = Rectangle(*C[:-1])


class CircleEntity(Entity):
    def __init__(
        self,
        center: Point,
        heading: float,
        radius: float,
        movable: bool = True,
        friction: float = 0,
    ):
        super(CircleEntity, self).__init__(center, heading, movable, friction)
        self.radius = radius
        self.buildGeometry()

    def buildGeometry(self):
        self.obj = Circle(self.center, self.radius)
