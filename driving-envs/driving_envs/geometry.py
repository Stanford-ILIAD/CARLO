import numpy as np
from typing import Union


class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def norm(self, p: int = 2) -> float:
        return (self.x ** p + self.y ** p) ** (1.0 / p)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y

    def __mul__(self, other: float) -> "Point":
        return Point(other * self.x, other * self.y)

    def __rmul__(self, other: float) -> "Point":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Point":
        return self.__mul__(1.0 / other)

    def isInside(self, other: Union["Line", "Rectangle", "Circle"]) -> bool:
        if isinstance(other, Line):
            AM = Line(other.p1, self)
            MB = Line(self, other.p2)
            return np.isclose(np.abs(AM.dot(MB)), AM.length * MB.length)

        elif isinstance(other, Rectangle):
            # Based on https://stackoverflow.com/a/2763387
            AB = Line(other.c1, other.c2)
            AM = Line(other.c1, self)
            BC = Line(other.c2, other.c3)
            BM = Line(other.c2, self)

            return 0 <= AB.dot(AM) <= AB.dot(AB) and 0 <= BC.dot(BM) <= BC.dot(BC)

        elif isinstance(other, Circle):
            return (self - other.m).norm(p=2) <= other.r

        raise NotImplementedError

    def distanceTo(self, other: Union["Point", "Line", "Rectangle", "Circle"]) -> float:
        if isinstance(other, Point):
            return (self - other).norm(p=2)

        elif isinstance(other, Line):
            # Based on https://math.stackexchange.com/a/330329
            s2_minus_s1 = other.p2 - other.p1
            that = (self - other.p1).dot(s2_minus_s1) / s2_minus_s1.dot(s2_minus_s1)
            tstar = np.minimum(1, np.maximum(0, that))
            return (other.p1 + tstar * s2_minus_s1 - self).norm(p=2)

        elif isinstance(other, Rectangle):
            if self.isInside(other):
                return 0
            E = other.edges
            return np.min([self.distanceTo(e) for e in E])

        elif isinstance(other, Circle):
            return np.maximum(0, self.distanceTo(other.m) - other.r)

        else:
            try:
                return other.distanceTo(self)
            except NameError:
                raise NotImplementedError
            print("Something went wrong!")


def onSegment(p: Point, q: Point, r: Point) -> bool:
    """
    Given three colinear points p, q, r, the function checks if 
    point q lies on line segment 'pr' 
    """
    return (
        q.x <= np.maximum(p.x, r.x)
        and q.x >= np.minimum(p.x, r.x)
        and q.y <= np.maximum(p.y, r.y)
        and q.y >= np.minimum(p.y, r.y)
    )


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    To find orientation of ordered triplet (p, q, r). 
    The function returns following values 
    0 --> p, q and r are colinear 
    1 --> Clockwise 
    2 --> Counterclockwise 
    """
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/ for details of below formula.
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return "Line(" + str(self.p1) + ", " + str(self.p2) + ")"

    def intersectsWith(self, other: Union["Line", "Rectangle", "Circle"]):
        if isinstance(other, Line):
            p1 = self.p1
            q1 = self.p2
            p2 = other.p1
            q2 = other.p2

            # Based on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
            # Find the four orientations needed for general and special cases
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special Cases
            # p1, q1 and p2 are colinear and p2 lies on segment p1q1
            if o1 == 0 and onSegment(p1, p2, q1):
                return True

            # p1, q1 and q2 are colinear and q2 lies on segment p1q1
            if o2 == 0 and onSegment(p1, q2, q1):
                return True

            # p2, q2 and p1 are colinear and p1 lies on segment p2q2
            if o3 == 0 and onSegment(p2, p1, q2):
                return True

            # p2, q2 and q1 are colinear and q1 lies on segment p2q2
            if o4 == 0 and onSegment(p2, q1, q2):
                return True

            return False  # Doesn't fall in any of the above cases

        elif isinstance(other, Rectangle):
            if self.p1.isInside(other) or self.p2.isInside(other):
                return True
            E = other.edges
            for edge in E:
                if self.intersectsWith(edge):
                    return True
            return False

        elif isinstance(other, Circle):
            return other.m.distanceTo(self) <= other.r

        raise NotImplementedError

    @property
    def length(self):
        return self.p1.distanceTo(self.p2)

    def dot(self, other: "Line") -> float:  # assumes Line is a vector from p1 to p2
        v1 = self.p2 - self.p1
        v2 = other.p2 - other.p1
        return v1.dot(v2)

    def distanceTo(self, other: "Point") -> float:
        if isinstance(other, Point):
            return other.distanceTo(self)


class Rectangle:
    # 3 points are enough to represent a rectangle
    def __init__(self, c1: Point, c2: Point, c3: Point):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c3 + c1 - c2

    def __str__(self):
        return "Rectangle({}, {}, {}, {})".format(self.c1, self.c2, self.c3, self.c4)

    @property
    def edges(self):
        e1 = Line(self.c1, self.c2)
        e2 = Line(self.c2, self.c3)
        e3 = Line(self.c3, self.c4)
        e4 = Line(self.c4, self.c1)
        return [e1, e2, e3, e4]

    @property
    def corners(self):
        return [self.c1, self.c2, self.c3, self.c4]

    def intersectsWith(self, other: Union["Line", "Rectangle", "Circle"]) -> bool:
        if isinstance(other, Line):
            return other.intersectsWith(self)

        elif isinstance(other, Rectangle) or isinstance(other, Circle):
            E = self.edges
            for e in E:
                if e.intersectsWith(other):
                    return True
            return False

        raise NotImplementedError

    def distanceTo(self, other: "Point") -> float:
        if isinstance(other, Point):
            return other.distanceTo(self)


class Circle:
    def __init__(self, m: Point, r: float):
        self.m = m
        self.r = r

    def __str__(self):
        return "Circle(" + str(self.m) + ", radius = " + str(self.r) + ")"

    def intersectsWith(self, other: Union["Line", "Rectangle", "Circle"]):
        if isinstance(other, Line) or isinstance(other, Rectangle):
            return other.intersectsWith(self)

        elif isinstance(other, Circle):
            return self.m.distanceTo(other.m) <= self.r + other.r

        raise NotImplementedError

    def distanceTo(self, other: "Point") -> float:
        if isinstance(other, Point):
            return other.distanceTo(self)
