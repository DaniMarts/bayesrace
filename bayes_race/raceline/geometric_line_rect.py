import numpy as np
import matplotlib.pyplot as plt

from bayes_race.tracks import Rectangular
from bayes_race.params import F110
from bayes_race.raceline import calcMinimumTimeSpeedInputs

from bayes_race.raceline.plot_raceline import plot_track, make_colorbar


params = F110()  # the parameters of the F1Tenth car

# parameters of the centreline of the track, in m
LENGTH = 8
BREADTH = 9
width = 0.8
r_corner = 1

track = Rectangular(LENGTH, BREADTH, width, r_corner)

theta = np.pi/2  # rad, the angle of the corner

r_i = r_corner - width/2  # m, inner radius
r_i = r_i if r_i > 0 else 0
r_o = r_corner + width/2  # m, outer radius
r_r = r_i + (r_o - r_i)/(1-np.cos(theta/2))  # m, radius of the geometric line

SCALE = 0.85  # how much of the track width can be explored
# parameters of the geometric line of the track, in m
length = LENGTH + width * SCALE
breadth = BREADTH + width * SCALE
racing_line = Rectangular.make_rect(length, breadth, r_r, 200)
initial_segment = np.array([np.full((10,), racing_line[0][0]), np.linspace(-BREADTH/2+r_o, racing_line[1][0], 10)])
racing_line = np.hstack([racing_line, initial_segment])
time, speed, inputs = calcMinimumTimeSpeedInputs(*racing_line, **params)

v_params = {"name": "speed", "cmap": "viridis", "arr": speed, "label": "Speed [$m/s$]",
            "title": "Optimal speed profile"}

acc_params = {"name": "acc", "cmap": "RdYlGn", "arr": inputs[0], "label": "Longitudinal acceleration [$m/s^2$]",
              "title": "Optimal acceleration profile"}

steer_params = {"name": "steer", "cmap": "brg", "arr": inputs[1], "label": "Steering angle [$deg$]",
                "title": "Steering profile"}

make_colorbar(v_params, track, *racing_line)
make_colorbar(acc_params, track, *racing_line)
make_colorbar(steer_params, track, *racing_line)

# fig = plot_track(track)
# plt.plot(*racing_line)
# plt.axis("equal")
# plt.show()

