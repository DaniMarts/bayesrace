"""	Plot optimal racing lines from saved results, for the rectangular track.
	See generate_raceline_ucb.py, generate_raceline_ethz.py, generate_raceline_ethzmobil.py
Todo:
	* plot inputs (controls)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'

import numpy as np

from bayes_race.tracks import Rectangular
from bayes_race.params import F110, ORCA
from bayes_race.raceline import randomTrajectory
from bayes_race.raceline import calcMinimumTimeSpeedInputs
from bayes_race.utils import Spline2D

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

#####################################################################
# which data

# SAVE_RESULTS = True
SAVE_RESULTS = False

# saved results 1
# savestr = '20211007203017'  # garbage line
# savestr = '20211007230356'  # garbage line
# savestr = '20211007234343'  # garbage line
# savestr = '20211008225852'  # good line on 6x4, 0.5m wide track, 1 iteration, F110
# savestr = '20211008234318'  # good line on 6x4, 0.8m wide track, 3 iterations, F110
# savestr = '20211009125025'  # good line on 6x4, 0.8m wide track, 3 iterations, F110
# savestr = '20211009182342'  # good line on 6x4, 0.8m wide track, 3 iterations, F110

# savestr = '20211013231001'  # ok line on rounded 6x4, 0.8m wide track, 1 iterations, F110
# savestr = '20211013233938'  # ok line on rounded 6x4, 0.8m wide track, 1 iterations, F110
# savestr = '20211013235614'  # good line on rounded 6x4, 0.8m wide track, 1 iterations, F110
# savestr = '20211014001407'  # good line on rounded 4x6, 0.8m wide track, 1 iterations, F110
savestr = '2021-10-14-17_17_09'  # good line on rounded 4x6, 0.8m wide track, 3 iterations, F110


TRACK_NAME = 'Rectangular'

if savestr is '' or TRACK_NAME is '':
	sys.exit('\nspecify which file to load... \n')

# choose vehicle params and specify indices of the nodes
if TRACK_NAME is 'Rectangular':
	params = F110()
	track = Rectangular(4, 6, 0.8, 0.3)
	LASTIDX = 0

#####################################################################
# region load saved data

data = np.load('results/{}_raceline_data-{}.npz'.format(TRACK_NAME, savestr))
y_ei = data['y_ei']
y_nei = data['y_nei']
y_rnd = data['y_rnd']
iters = data['iters']
train_x_all_ei = data['train_x_all_ei']
train_x_all_nei = data['train_x_all_nei']
train_x_all_random = data['train_x_all_random']
train_y_all_ei = data['train_y_all_ei'].squeeze(-1)
train_y_all_nei = data['train_y_all_nei'].squeeze(-1)
train_y_all_random = data['train_y_all_random'].squeeze(-1)
N_TRIALS = train_x_all_ei.shape[0]
N_DIMS = train_x_all_ei.shape[-1]
# endregion
#####################################################################
# region plot best lap times
filepath = 'results/{}_convergence.png'.format(TRACK_NAME)


def ci(y):
	return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)


plt.figure()
plt.grid(True)

plt.gca().set_prop_cycle(None)
plt.plot(iters, y_rnd.mean(axis=0), linewidth=1.5)
plt.plot(iters, y_ei.mean(axis=0), linewidth=1.5)
plt.plot(iters, y_nei.mean(axis=0), linewidth=1.5)

plt.gca().set_prop_cycle(None)
plt.fill_between(iters, y_rnd.mean(axis=0) - ci(y_rnd), y_rnd.mean(axis=0) + ci(y_rnd), label="random", alpha=0.2)
plt.fill_between(iters, y_ei.mean(axis=0) - ci(y_ei), y_ei.mean(axis=0) + ci(y_ei), label="EI", alpha=0.2)
plt.fill_between(iters, y_nei.mean(axis=0) - ci(y_nei), y_nei.mean(axis=0) + ci(y_nei), label="NEI", alpha=0.2)

plt.xlabel('# number of observations (beyond initial points)')
plt.ylabel('best lap times [s]')
plt.xlim([0, 50])
plt.legend(loc='lower left')

if SAVE_RESULTS:
	plt.savefig(filepath, dpi=600, bbox_inches='tight')
# endregion plot best lap times

#####################################################################
# region plot best trajectory
filepath = 'results/{}_bestlap.png'.format(TRACK_NAME)  # path for saving result

n_waypoints = N_DIMS
n_samples = 500

x_inner, y_inner = track.x_inner, track.y_inner
x_center, y_center = track.x_center, track.y_center
x_outer, y_outer = track.x_outer, track.y_outer

rand_traj = randomTrajectory(track=track, n_waypoints=n_waypoints)


def gen_traj(x_all, idx, sim):
	"""Calculates x, y points of the resampled optimum raceline"""
	w_idx = x_all[sim][idx]
	wx, wy = rand_traj.calculate_xy(
		width=w_idx,
		last_index=LASTIDX,
		# theta=theta,
	)
	sp = Spline2D(wx, wy)
	s = np.linspace(0, sp.s[-1] - 0.001, n_samples)
	x, y = [], []
	for i_s in s:
		ix, iy = sp.calc_position(i_s)
		x.append(ix)
		y.append(iy)
	return wx, wy, x, y


fig = plt.figure(0)
ax = plt.gca()
ax.axis('equal')
# plotting the track
plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)
plt.plot(x_outer, y_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(x_inner, y_inner, 'k', lw=0.5, alpha=0.5)

# best trajectory (assuming qNEI is the best)
sim, pidx = np.unravel_index(np.argmin(train_y_all_nei), train_y_all_nei.shape)
# getting the best x, y from the raceline
wx_nei, wy_nei, x_nei, y_nei = gen_traj(train_x_all_nei, pidx, sim)
# plotting the waypoints of the raceline
plt.plot(wx_nei[:-1], wy_nei[:-1], linestyle='', marker='D', ms=5)
# calculating minumum possible time, speeds and inputs at each point in the raceline, for a particular car
time, speed, inputs = calcMinimumTimeSpeedInputs(x_nei, y_nei, **params)
x, y = np.array(x_nei), np.array(y_nei)
points = np.array([x, y]).T.reshape(-1, 1, 2)
# making the speed color-bar
# each row is a segment. One layer for x, other for y. Row n Column 1 has one point, Row n Col 2 has the other.
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(speed)
lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax, label="Speed [m/s]")

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.title("Optimal speed profile")
plt.show()

# region plotting acceleration
acc=inputs[0]
fig = plt.figure(2)
ax = plt.gca()
ax.axis('equal')
# plotting the track
plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)
plt.plot(x_outer, y_outer, 'k', lw=0.5, alpha=0.5)
plt.plot(x_inner, y_inner, 'k', lw=0.5, alpha=0.5)

plt.plot(wx_nei[:-1], wy_nei[:-1], linestyle='', marker='D', ms=5)

acc_norm = plt.Normalize(acc.min(), acc.max())
lc = LineCollection(segments, cmap='RdYlGn', norm=acc_norm)
lc.set_array(acc)
lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax, label="Longitudinal acceleration [$m/s^2$]")

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.title("Optimal acceleration profile")
plt.show()

# endregion plotting acceleration
if SAVE_RESULTS:
	np.savez('results/{}_optimalxy-{}.npz'.format(TRACK_NAME, savestr), x=x, y=y)
	np.savez('results/{}_raceline-{}.npz'.format(TRACK_NAME, savestr), x=x, y=y, time=time, speed=speed, inputs=inputs)
	plt.savefig(filepath, dpi=600, bbox_inches='tight')
# endregion
#####################################################################
