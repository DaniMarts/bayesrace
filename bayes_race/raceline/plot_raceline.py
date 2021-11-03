"""	Plot optimal racing lines from saved results, a general track.
"""

__author__ = 'Danilo Martins'
__email__ = 'mrtdan014@myuct.ac.za'

import numpy as np

from bayes_race.raceline import randomTrajectory, calcMinimumTimeSpeedInputs
from bayes_race.utils import Spline2D

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_raceline(Track_type, class_params: dict, LASTIDX, Car_type, savestr, save_results=False):
	"""

	Args:
		Track_type:
		class_params:
		LASTIDX:
		Car_type:
		savestr:
		save_results:

	"""

	TRACK_NAME = Track_type.__name__
	try:
		track = Track_type(**class_params)
	except:
		track = Track_type(*class_params)
	params = Car_type()

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

	if save_results:
		plt.savefig(filepath, dpi=600, bbox_inches='tight')
	# endregion

	# region plot best trajectory
	filepath = 'results/{}_bestlap-{}.png'.format(TRACK_NAME, savestr)  # path for saving result

	n_waypoints = N_DIMS
	n_samples = 500

	rand_traj = randomTrajectory(track=track, n_waypoints=n_waypoints)

	# best trajectory (assuming qNEI is the best)
	sim, pidx = np.unravel_index(np.argmin(train_y_all_nei), train_y_all_nei.shape)
	# getting the best x, y from the raceline
	best_x = train_x_all_nei[sim][pidx]
	wx_nei, wy_nei, x_nei, y_nei = gen_traj(rand_traj, best_x, n_samples, LASTIDX)

	# calculating minimum possible time, speeds and inputs at each point in the raceline, for a particular car
	time, speed, inputs = calcMinimumTimeSpeedInputs(x_nei, y_nei, **params)
	x, y = np.array(x_nei), np.array(y_nei)

	v_params = {"name": "speed", "cmap": "viridis", "arr": speed, "label": "Speed [$m/s$]",
	            "title": "Optimal speed profile"}

	acc_params = {"name": "acc", "cmap": "RdYlGn", "arr": inputs[0], "label": "Longitudinal acceleration [$m/s^2$]",
	              "title": "Optimal acceleration profile"}

	steer_params = {"name": "steer", "cmap": "brg", "arr": inputs[1], "label": "Steering angle [$deg$]",
	                "title": "Steering profile"}

	make_colorbar(v_params, track, x, y, wx_nei, wy_nei)
	make_colorbar(acc_params, track, x, y, wx_nei, wy_nei)
	make_colorbar(steer_params, track, x, y, wx_nei, wy_nei)

	if save_results:
		np.savez('results/{}_optimalxy-{}.npz'.format(TRACK_NAME, savestr), x=x, y=y)
		np.savez('results/{}_raceline-{}.npz'.format(TRACK_NAME, savestr), x=x, y=y, time=time, speed=speed,
		         inputs=inputs)
		plt.savefig(filepath, dpi=600, bbox_inches='tight')
	# endregion
	return


def make_colorbar(params, track, x, y, wx=None, wy=None):
	if wy is None:
		wy = []
	if wx is None:
		wx = []
	arr = params["arr"]
	cmap = params["cmap"]
	label = params["label"]
	title = params["title"]

	points = np.array([x, y]).T.reshape(-1, 1, 2)

	# making the speed color-bar
	# each row is a segment. One layer for x, other for y. Row n Column 1 has one point, Row n Col 2 has the other.
	segments = np.concatenate([points[:-1], points[1:]], axis=1)

	fig = plot_track(track)
	ax = plt.gca()
	ax.axis('equal')
	if wx is not None and wy is not None:
		# plotting the waypoints of the raceline
		plt.plot(wx[:-1], wy[:-1], linestyle='', marker='D', ms=5)
	norm = plt.Normalize(arr.min(), arr.max())
	lc = LineCollection(segments, cmap=cmap, norm=norm)
	lc.set_array(arr)
	lc.set_linewidth(2)
	line = ax.add_collection(lc)
	fig.colorbar(line, ax=ax, label=label)
	ax.set_xlabel('x [m]')
	ax.set_ylabel('y [m]')
	plt.title(title)
	plt.tight_layout()
	plt.show()


def plot_track(track):
	"""Plots the track"""
	x_inner, y_inner = track.x_inner, track.y_inner
	x_center, y_center = track.x_center, track.y_center
	x_outer, y_outer = track.x_outer, track.y_outer

	fig = plt.figure()
	# plotting the track
	plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)
	plt.plot(x_outer, y_outer, 'k', lw=0.5, alpha=0.5)
	plt.plot(x_inner, y_inner, 'k', lw=0.5, alpha=0.5)

	return fig


def gen_traj(traj: randomTrajectory, best_waypoints, n_samples, last_index=0, theta=None):
	"""Calculates x, y points of the resampled optimum raceline"""
	wx, wy = traj.calculate_xy(
		width=best_waypoints,
		last_index=last_index,
		theta=theta,
	)
	sp = Spline2D(wx, wy)
	s = np.linspace(0, sp.s[-1] - 0.001, n_samples)
	x, y = [], []
	for i_s in s:
		ix, iy = sp.calc_position(i_s)
		x.append(ix)
		y.append(iy)
	return wx, wy, x, y


if __name__ == "__main__":
	from bayes_race.tracks import Rectangular
	from bayes_race.params import F110

	track_params = {"length": 4, "breadth": 6, "width": 0.8, "r_corner": 0.3}
	savestr = '2021-10-14-17_17_09'  # good line on rounded 4x6, 0.8 m wide track, 3 iterations, F110

	plot_raceline(Rectangular, track_params, 0, F110, savestr, save_results=False)
