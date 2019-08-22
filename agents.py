from entities import RectangleEntity, CircleEntity
from geometry import Point

# For colors, we use tkinter colors. See http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter


class Car(RectangleEntity):
    def __init__(self, center: Point, heading: float, color: str = "red"):
        size = Point(4.0, 2.0)
        movable = True
        friction = 0.06
        super(Car, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = True


class Pedestrian(CircleEntity):
    def __init__(self, center: Point, heading: float, color: str = "LightSalmon2"):
        radius = 0.4
        movable = True
        friction = 0.2
        super(Pedestrian, self).__init__(center, heading, radius, movable, friction)
        self.color = color
        self.collidable = True


class Building(RectangleEntity):
    def __init__(self, center: Point, size: Point, color: str = "gray26"):
        heading = 0.0
        movable = False
        friction = 0.0
        super(Building, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = True


class Painting(RectangleEntity):
    def __init__(self, center: Point, size: Point, color: str = "gray26"):
        heading = 0.0
        movable = False
        friction = 0.0
        super(Painting, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = False
