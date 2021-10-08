import numpy as np
import matplotlib.pyplot as plt

from bayes_race.utils import Spline2D
from bayes_race.tracks import *
from bayes_race.params import ORCA, F110
from bayes_race.raceline import randomTrajectory
from bayes_race.raceline import calcMinimumTime


rec = Rectangular(length=20, breadth=40, width=6)

SCALE = 0.80
N_WAYPOINTS = 25
traj = randomTrajectory(rec, N_WAYPOINTS)

width_random = traj.sample_nodes(scale=SCALE)

# find corresponding x,y coordinates
# here we choose terminal point to be the first point to prevent crashing before finishing
wx_random, wy_random = traj.calculate_xy(
	width_random,
	last_index=1,
	)

n_samples = 500
x_random, y_random = traj.fit_cubic_splines(
	wx=wx_random,
	wy=wy_random,
	n_samples=n_samples
	)

# plot
fig = rec.plot(color='k', grid=False)
x_center, y_center = rec.x_center, rec.y_center
plt.plot(x_center, y_center, '--k', alpha=0.5, lw=0.5)
plt.plot(x_random, y_random, label='splines', lw=1.5)
plt.plot(wx_random, wy_random, 'x', label='way points')
plt.plot(wx_random[0], wy_random[0], 'o', label='start')
plt.plot(wx_random[-1], wy_random[-1], 'o', label='finish')
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc=0)
plt.show()


# uncomment to calculate minimum time to traverse
# params = F110()
# t_random = calcMinimumTime(x_random, y_random, **params)
# print('time to traverse random trajectory: {}'.format(t_random))