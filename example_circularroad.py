import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
import time

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120
world_height = 120
inner_building_radius = 30
num_lanes = 2
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

w = World(dt, width = world_width, height = world_height, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.



# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
# A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.

# To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
w.add(cb)
w.add(RingBuilding(Point(world_width/2, world_height/2), inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width, 1+np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80'))

# Let's also add some lane markers on the ground. This is just decorative. Because, why not.
for lane_no in range(num_lanes - 1):
    lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
    lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
    for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))
    

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(91.75,60), np.pi/2)
w.add(c1)

w.render() # This visualizes the world we just constructed.


# Let's implement some simple policy for the car c1
desired_lane = 0
c1.set_control(0., 0.5) # Initially, the car will have 0 steering and 0.5 acceleration.
for k in range(600):
    v = c1.center - cb.center
    v = (np.arctan2(v.y, v.x) + np.pi/2 + np.pi) % (2 * np.pi) - np.pi
    if c1.heading < v:
        c1.set_control(0.1, 0.5)
    else:
        c1.set_control(0, 0.5)
    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt/4) # Let's watch it 4x

    if w.collision_exists(): # Or we can check if there is any collision at all.
        print('Collision exists somewhere...')

