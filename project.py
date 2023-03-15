import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
import time
from tkinter import *
import sys 

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120 # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 1
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 10

def make_optimal_angle_for_quadrant(x, y): 
    center_x = world_width/2
    center_y = world_height/2
    # quad 1 or quad 4
    if (x >= center_x and y >= center_y) or (x >= center_x and y <= center_y):
        return np.pi/2
    # quad 2 or quad 3
    else: 
        return -np.pi/2
    

w = World(dt, width = world_width, height = world_height, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
w.add(cb)
rb = RingBuilding(Point(world_width/2, world_height/2), inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width, 1+np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80')
w.add(rb)
inner_radius = inner_building_radius
outer_radius = inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width 

# calculate the areas of the circles and the overlap
inner_area = np.pi * inner_radius**2
outer_area = np.pi * outer_radius**2
overlap_area = np.pi * (outer_radius ** 2 - inner_radius ** 2) 
example_car = Car(Point(91.75, 60), np.pi/2)

# set the number of cars in the world
num_cars = 3
colors = ['red', 'green', 'blue']
cars = []
while num_cars > 0:
    # generate random polar coordinates
    radius = np.random.uniform(inner_radius, outer_radius )
    angle = np.random.uniform(0, 2*np.pi)

    # convert polar to cartesian coordinates
    x = radius * np.cos(angle) + world_width/2
    y = radius * np.sin(angle) + world_height/2

    # check if the point lies within the overlap area
    if (radius <= outer_radius - 2 and radius >= inner_radius + 2 and x >= 0 and x <= world_width and y >= 0 and y <= world_height):
        print(f"Generated point: ({x}, {y})")
        
        c1 = Car(Point(x, y), make_optimal_angle_for_quadrant(x, y), colors[len(colors) % (num_cars)]) # rn set each car to this heading angle 
        num_cars = num_cars - 1
        cars.append(c1)
        
for car in cars:
    w.add(car)
    
# add lane markers [just decorative]-> unnecessary now #TODO remove
for lane_no in range(num_lanes - 1):
    lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
    lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
    for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))
    

w.render() # This visualizes the world we just constructed.

# Let's implement some simple policy for all the cars 
# this is adapted from the original example circular road, but applies to all cars in cars list
desired_lane = 1
for k in range(600):
    # vectorize losses to have a loss for each car in the world
    lp = [0.] * len(cars)
    # lp = 0.
    for car_index in range(len(cars)): 
        if car.distanceTo(cb) < desired_lane*(lane_width + lane_marker_width) + 0.2:
            lp[car_index] += 0
        elif car.distanceTo(rb) < (num_lanes - desired_lane - 1)*(lane_width + lane_marker_width) + 0.3:
            lp[car_index] += 1.
    v = [car.center - cb.center for car in cars]
    for i in range(len(v)): 
        v[i] = np.mod(np.arctan2(v[i].y, v[i].x) + np.pi/2, 2*np.pi)

    for car_index in range(len(cars)): 
        if car.heading < v[car_index]:
            lp[car_index] += 0.7
        else: 
            lp[car_index] += 0.
    
    for car_index in range(len(cars)): 
        if np.random.rand() < lp[car_index]:
            cars[car_index].set_control(0.2, 0.1)
        else: 
            cars[car_index].set_control(-0.1, 0.1)
    
    w.tick() # This ticks the world for one time step (dt second)
    w.render()
    time.sleep(dt/4) # Let's watch it 4x
    if w.collision_exists(): # We can check if there is any collision at all.
        print('Collision exists somewhere...')
w.close()