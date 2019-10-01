"""Utilities for human joystick control (demo collection, evaluation, ...)."""
import time
from driving_envs.graphics import Point, Text

# Pygame joystick ids.
LEFT_X_AXIS = 0
LEFT_Y_AXIS = 1
RIGHT_X_AXIS = 3
RIGHT_Y_AXIS = 4

# Controls how sensitive the car control is to the joystick motion.
TURN_SCALING = -0.02
ACC_SCALING = -4


def display_countdown(multi_env, start_delay=5):
    """Display countdown text. Assume render() has already been called."""
    viz = multi_env.world.visualizer
    # Display text at center of screen.
    txt_x, txt_y = multi_env.width // 2, multi_env.height // 2
    txt = None
    for i in range(start_delay):
        txt = Text(
            Point(viz.ppm * txt_x, viz.display_height - viz.ppm * txt_y),
            "Starting in {:d}".format(start_delay - i),
        )
        txt.setSize(20)
        txt.draw(viz.win)
        time.sleep(1)
        txt.undraw()
