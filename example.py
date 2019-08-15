import numpy as np
from world import World
from agents import Car, Building, Pedestrian, Painting
from geometry import Point
import time

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and buildings.
# A Painting object is a rectangle that the vehicles cannot be collided with. So we use them for the sidewalks.
# A Building object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.
w.add(Painting(Point(71.5, 106.5), Point(97, 27), 'gray80')) # We build a sidewalk.
w.add(Building(Point(72.5, 107.5), Point(95, 25))) # The building is then on top of the sidewalk, with some margin.

# Let's repeat this for 4 different buildings.
w.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
w.add(Building(Point(7.5, 107.5), Point(15, 25)))

w.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
w.add(Building(Point(7.5, 40), Point(15, 80)))

w.add(Painting(Point(71.5, 41), Point(97, 82), 'gray80'))
w.add(Building(Point(72.5, 40), Point(95, 80)))

# Let's also add some zebra crossings, because why not.
w.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(20,20), np.pi/2)
w.add(c1)

c2 = Car(Point(100,90), np.pi, 'blue')
c2.velocity = Point(5.3,0) # We can also specify an initial velocity just like this.
w.add(c2)

# Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
p1 = Pedestrian(Point(30,81), np.pi)
p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
w.add(p1)

w.render() # This visualizes the world we just constructed.

p1.set_control(0, 0.2) # The pedestrian will have 0 steering and 0.2 acceleration. So it will not change its direction.
c1.set_control(0, 1.2)
for k in range(400):
	# All movable objects will keep their control the same as long as we don't change it.
	if k == 100: # Let's say the first Car will release throttle (and start slowing down due to friction)
		c1.set_control(0, 0)
	elif k == 170: # The first Car starts pushing the brake a little bit. The second Car starts turning right with some acceleration.
		c1.set_control(0, -0.2)
		c2.set_control(-0.1, 0.5)
	elif k == 215: # The second Car stops turning.
		c2.set_control(0, 0.5)
	w.tick() # This ticks the world for one time step (dt second)
	w.render()
	time.sleep(dt/4) # Let's watch it 4x

	if w.collision_exists(p1): # We can check if the Pedestrian is currently involved in a collision. We could also check c1 or c2.
		print('Pedestrian has died, good job!')
	elif w.collision_exists(): # Or we can check if there is any collision at all.
		print('Collision exists somewhere...')

