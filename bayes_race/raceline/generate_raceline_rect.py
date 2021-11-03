""" Compute racing line using Bayesian Optimization (BayesOpt).
    This script compares EI, noisyEI and random strategies for sampling.
"""

__author__ = 'Danilo Martins'
__email__ = 'mrtdan014@myuct.ac.za'

import time
import numpy as np

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model

from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from bayes_race.tracks import Rectangular
from bayes_race.params import F110
from bayes_race.raceline import randomTrajectory
from bayes_race.raceline import calcMinimumTime, calcMinimumTimeSpeedInputs
from bayes_race.utils import Spline2D

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

#####################################################################
# set device in torch

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

#####################################################################
# simulation settings

SEED = np.random.randint(999)
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 1  # useful for parallelization, DON'T change
N_TRIALS = 3  # number of times bayesopt is run
N_BATCH = 100  # new observations after initialization
MC_SAMPLES = 64  # monte carlo samples
N_INITIAL_SAMPLES = 10  # samples to initialize GP
PLOT_RESULTS = True  # whether to plot results
PLOT_RACELINE = True  # whether to plot the best raceline
SAVE_RESULTS = True  # whether to save results
VERBOSE = False  # whether to print progress to terminal
INTERACTIVE = True  # whether to plot the trajectories of each iteration
N_WAYPOINTS = 100  # resampled waypoints
SCALE = 0.9  # shrinking factor for track width
LASTIDX = 0  # fixed node at the end DO NOT CHANGE
TRACK_NAME = "Rectangular"
# define indices for the nodes
# NODES = [33, 67, 116, 166, 203, 239, 274, 309, 344, 362, 382, 407, 434, 448, 470, 514, 550, 586, 622, 657, 665]

#####################################################################
# track specific data

params = F110()
track = Rectangular(5, 6, 0.8, 1)
fig = track.plot(plot_centre=True)
plt.show()

track_width = track.track_width * SCALE
# theta = track.theta_track[NODES]
# N_DIMS = len(NODES)
N_DIMS = 15
n_waypoints = N_DIMS

# an object representing a random trajectory within the track
rand_traj = randomTrajectory(track=track, n_waypoints=n_waypoints)

# defining the boundary of the vector containing the random distances normal to the centreline
bounds = torch.tensor([[-track_width / 2] * N_DIMS, [track_width / 2] * N_DIMS], device=device, dtype=dtype)


def evaluate_y(x_eval, mean_y=None, std_y=None):
	"""
	evaluate true output (time) for given x (distance of nodes from center line)

	Args:
		x_eval: an array containing the distance of each point to the centreline. Could be multiple rows,
			if parallelization is implemented
		mean_y: mean time, from previous evaluations
		std_y: standard deviation of time, from previous evaluations

	Returns:
		(array): a list containing the times taken to traverse each trajectory parameterized by the rows of x_eval
	Todo:
		* parallelize evaluations

	"""

	if type(x_eval) is torch.Tensor:
		is_tensor = True
		x_eval = x_eval.cpu().numpy()  # convert the Tensor to a numpy array
	else:
		is_tensor = False

	if len(x_eval.shape) == 1:  # if the shape is something like (n,)
		x_eval = x_eval.reshape(1, -1)  # reshape it to (1, n)
	n_eval = x_eval.shape[0]  # the number of elements in x_eval (i.e., n)

	y_eval = np.zeros(n_eval)  # stores the time taken to complete a trajectory parameterized by x_eval
	for ids in range(n_eval):  # for each row of input widths in x_eval
		# getting the coordinate for each point in the trajectory parameterized by x_eval
		wx, wy = rand_traj.calculate_xy(
			width=x_eval[ids],
			last_index=LASTIDX,
			start_width=-track_width/2,
			end_width=-track_width/2,
			# theta=theta,
		)
		# fitting splines on the (wx, wy) points and resampling new points from it
		x, y = rand_traj.fit_cubic_splines(
			wx=wx,
			wy=wy,
			n_samples=N_WAYPOINTS,
		)
		# calculating minimum time to complete the trajectory
		y_eval[ids] = -calcMinimumTime(x, y, **params)  # we want to max negative lap times

	if mean_y and std_y:
		y_eval = normalize(y_eval, mean_y, std_y)

	if is_tensor:
		return torch.tensor(y_eval, device=device, dtype=dtype).unsqueeze(-1)
	else:
		return y_eval.ravel()


def generate_initial_data(n_samples=10):
	"""generate training data.

	Args:
		n_samples (int): number of sample trajectories used to train

	Returns:
		train_x (Tensor): n_samples x n_waypoints vector, with each row containing random widths used to train
		train_y (Tensor): n_samples x 1 vector containing the times taken to traverse the line
		best_y (float): best time taken
		mean_y (float): mean of the times taken
		std_y (float): standard deviation of the times taken
	"""
	train_x = np.zeros([n_samples, n_waypoints])
	train_y_ = np.zeros([n_samples, 1])

	for ids in range(n_samples):
		width_random = rand_traj.sample_nodes(scale=SCALE)  # generating a (1, n_samples) vector with random widths
		t_random = evaluate_y(width_random)  # calculating the time to complete a lap parameterized by width_random
		train_x[ids, :] = width_random  # filling this row with the random widths
		train_y_[ids, :] = t_random  # filling this row with the times taken

	mean_y, std_y = train_y_.mean(), train_y_.std()
	train_y = normalize(train_y_, mean_y, std_y)
	train_x = torch.tensor(train_x, device=device, dtype=dtype)
	train_y = torch.tensor(train_y, device=device, dtype=dtype)
	best_y = train_y.max().item()
	return train_x, train_y, best_y, mean_y, std_y


def normalize(y_eval, mean_y, std_y):
	""" normalize outputs for GP
	"""
	return (y_eval - mean_y) / std_y


#####################################################################
# modeling and optimization functions called in closed-loop

def initialize_model(train_x, train_y, state_dict=None):
	"""initialize GP model with/without initial states
	"""
	model = SingleTaskGP(train_x, train_y).to(train_x)
	mll = ExactMarginalLogLikelihood(model.likelihood, model)
	# load state dict if it is passed
	if state_dict is not None:
		model.load_state_dict(state_dict)
	return mll, model


def optimize_acqf_and_get_observation(acq_func, mean_y=None, std_y=None):
	"""Optimize acquisition function and evaluate new candidates.

	Args:
		acq_func: the acquisition function to be used (random, qEI, qNEI)
		mean_y:
		std_y:

	Returns:
		new_x, new_y (array, array): new sampled widths and time taken to traverse
	"""

	# optimize
	candidates, _ = optimize_acqf(
		acq_function=acq_func,
		bounds=bounds,
		q=BATCH_SIZE,
		num_restarts=10,
		raw_samples=512,  # used for initialization heuristic
	)

	# observe new values
	new_x = candidates.detach()
	new_y = evaluate_y(new_x, mean_y=mean_y, std_y=std_y)  # evaluating the new candidates of widths
	return new_x, new_y


def sample_random_observations(mean_y, std_y):
	"""sample a random trajectory
	"""
	rand_x = torch.tensor(rand_traj.sample_nodes(scale=SCALE).reshape(1, -1), device=device, dtype=dtype)
	rand_y = evaluate_y(rand_x, mean_y=mean_y, std_y=std_y)
	return rand_x, rand_y


#####################################################################
# main simulation loop

# define the qEI and qNEI acquisition modules using a QMC sampler
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)


def optimize():
	verbose = True
	interactive = True

	best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []  # best observed times
	train_x_all_ei, train_x_all_nei, train_x_all_random = [], [], []  # all training inputs (widths)
	train_y_all_ei, train_y_all_nei, train_y_all_random = [], [], []  # all training outputs (times)

	# statistics over multiple trials
	for trial in range(1, N_TRIALS + 1):

		print('\nTrial {} of {}'.format(trial, N_TRIALS))
		best_observed_ei, best_observed_nei = [], []
		best_random = []

		# generate initial (random) training data and initialize model
		print('\nGenerating {} random samples'.format(N_INITIAL_SAMPLES))
		# initial data is the same for all acquisition methods (random, qEI, qNEI)
		train_x_ei, train_y_ei, best_y_ei, mean_y, std_y = generate_initial_data(n_samples=N_INITIAL_SAMPLES)
		mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)
		# initial data is the same for all acquisition methods (random, qEI, qNEI)
		train_x_nei, train_y_nei, best_y_nei = train_x_ei, train_y_ei, best_y_ei
		mll_nei, model_nei = initialize_model(train_x_nei, train_y_nei)
		# initial data is the same for all acquisition methods (random, qEI, qNEI)
		train_x_random, train_y_random, best_y_random = train_x_ei, train_y_ei, best_y_ei

		denormalize = lambda x: -(x * std_y + mean_y)

		# best observed times for each acquisition functions (same for all, here)
		best_observed_ei.append(denormalize(best_y_ei))
		best_observed_nei.append(denormalize(best_y_nei))
		best_random.append(denormalize(best_y_random))

		# run N_BATCH rounds of BayesOpt after the initial random batch
		for iteration in range(1, N_BATCH + 1):

			print('\nBatch {} of {}\n'.format(iteration, N_BATCH))
			t0 = time.time()  # initializing timer for this batch

			# fit the models
			fit_gpytorch_model(mll_ei)
			fit_gpytorch_model(mll_nei)

			# update acquisition functions with new data
			qEI = qExpectedImprovement(
				model=model_ei,
				best_f=train_y_ei.max(),
				sampler=qmc_sampler,
			)

			qNEI = qNoisyExpectedImprovement(
				model=model_nei,
				X_baseline=train_x_nei,
				sampler=qmc_sampler,
			)

			# optimize acquisition function and evaluate new sample
			# educated guess with qEI
			new_x_ei, new_y_ei = optimize_acqf_and_get_observation(qEI, mean_y=mean_y, std_y=std_y)
			print('EI: time to traverse is {:.4f}s'.format(-(new_y_ei.flatten()[0] * std_y + mean_y)))
			# educated guess with qNEI
			new_x_nei, new_y_nei = optimize_acqf_and_get_observation(qNEI, mean_y=mean_y, std_y=std_y)
			print('NEI: time to traverse is {:.4f}s'.format(-(new_y_nei.flatten()[0] * std_y + mean_y)))
			# random guess
			new_x_random, new_y_random = sample_random_observations(mean_y=mean_y, std_y=std_y)
			print('Random: time to traverse is {:.4f}s'.format(-(new_y_random.flatten()[0] * std_y + mean_y)))

			# update training points, by appending the last data
			train_x_ei = torch.cat([train_x_ei, new_x_ei])
			train_y_ei = torch.cat([train_y_ei, new_y_ei])

			train_x_nei = torch.cat([train_x_nei, new_x_nei])
			train_y_nei = torch.cat([train_y_nei, new_y_nei])
			# never actually used to train anything
			train_x_random = torch.cat([train_x_random, new_x_random])
			train_y_random = torch.cat([train_y_random, new_y_random])

			# update progress
			best_value_ei = denormalize(train_y_ei.max().item())  # best observed time so far, for qEI
			best_value_nei = denormalize(train_y_nei.max().item())  # best observed time so far, for qNEI
			best_value_random = denormalize(train_y_random.max().item())  # best observed time so far, for random

			# appending the last best observed time to the best observed times list, even if it did not improve
			best_observed_ei.append(best_value_ei)
			best_observed_nei.append(best_value_nei)
			best_random.append(best_value_random)

			# reinitialize the models with new data, so they are ready for fitting on next iteration
			# use the current state dict to speed up fitting
			mll_ei, model_ei = initialize_model(
				train_x_ei,
				train_y_ei,
				model_ei.state_dict(),
			)
			mll_nei, model_nei = initialize_model(
				train_x_nei,
				train_y_nei,
				model_nei.state_dict(),
			)

			t1 = time.time()  # time used to calculate the duration of this iteration

			# plotting the raceline
			if INTERACTIVE:
				# plotting the track
				fig = track.plot(plot_centre=True)
				fig_params = [{"color": "orange", "label": "random"},
				              {"color": "c", "label": "qEI"},
				              {"color": "m", "label": "qNEI"}]

				modes = np.vstack([new_x_random[0], new_x_ei[0], new_x_nei[0]])

				for i in range(modes.shape[0]):
					# a random trajectory based on the latest data
					wx, wy = rand_traj.calculate_xy(
						width=modes[i],
						last_index=LASTIDX,
					)

					track.load_raceline(n_samples=500, raceline=[wx, wy], smooth=True)
					plt.plot(*track.raceline, alpha=0.75, lw=0.9, **fig_params[i])
					plt.scatter(wx, wy, c="k", marker='o', s=2)
					plt.title("Trajectories taken")
					plt.xlabel("x [m]")
					plt.ylabel("y [m]")

				plt.legend()
				plt.show()

			if VERBOSE:
				print(
					'best lap time (random, qEI, qNEI) = {:.2f}, {:.2f}, {:.2f}, time to compute = {:.2f}s'.format(
						best_value_random,
						best_value_ei,
						best_value_nei,
						t1 - t0  # duration of this iteration
					)
				)
			else:
				print(".")

		# appending to the best observed trial times
		best_observed_all_ei.append(best_observed_ei)
		best_observed_all_nei.append(best_observed_nei)
		best_random_all.append(best_random)

		train_x_all_ei.append(train_x_ei.cpu().numpy())
		train_x_all_nei.append(train_x_nei.cpu().numpy())
		train_x_all_random.append(train_x_random.cpu().numpy())

		train_y_all_ei.append(denormalize(train_y_ei.cpu().numpy()))
		train_y_all_nei.append(denormalize(train_y_nei.cpu().numpy()))
		train_y_all_random.append(denormalize(train_y_random.cpu().numpy()))

	iters = np.arange(N_BATCH + 1) * BATCH_SIZE
	y_ei = np.asarray(best_observed_all_ei)
	y_nei = np.asarray(best_observed_all_nei)
	y_rnd = np.asarray(best_random_all)
	savestr = time.strftime('%Y-%m-%d-%H_%M_%S')

	#####################################################################
	# save results

	if SAVE_RESULTS:
		np.savez(
			'results/{}_raceline_data-{}.npz'.format(TRACK_NAME, savestr),
			y_ei=y_ei,
			y_nei=y_nei,
			y_rnd=y_rnd,
			iters=iters,
			train_x_all_ei=np.asarray(train_x_all_ei),
			train_x_all_nei=np.asarray(train_x_all_nei),
			train_x_all_random=np.asarray(train_x_all_random),
			train_y_all_ei=np.asarray(train_y_all_ei),
			train_y_all_nei=np.asarray(train_y_all_nei),
			train_y_all_random=np.asarray(train_y_all_random),
			SEED=SEED,
		)

	#####################################################################
	# plot results

	if PLOT_RESULTS:
		def ci(y):
			return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

		plt.figure()
		plt.gca().set_prop_cycle(None)
		plt.plot(iters, y_rnd.mean(axis=0), linewidth=1.5)
		plt.plot(iters, y_ei.mean(axis=0), linewidth=1.5)
		plt.plot(iters, y_nei.mean(axis=0), linewidth=1.5)
		plt.gca().set_prop_cycle(None)
		plt.fill_between(iters, y_rnd.mean(axis=0) - ci(y_rnd), y_rnd.mean(axis=0) + ci(y_rnd), label='random',
		                 alpha=0.2)
		plt.fill_between(iters, y_ei.mean(axis=0) - ci(y_ei), y_ei.mean(axis=0) + ci(y_ei), label='qEI', alpha=0.2)
		plt.fill_between(iters, y_nei.mean(axis=0) - ci(y_nei), y_nei.mean(axis=0) + ci(y_nei), label='qNEI', alpha=0.2)
		plt.xlabel('number of observations (beyond initial points)')
		plt.ylabel('best lap times')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig('results/{}_laptimes-{}.png'.format(TRACK_NAME, savestr), dpi=600)
		plt.show()

	if PLOT_RACELINE:
		n_samples = 500

		x_inner, y_inner = track.x_inner, track.y_inner
		x_center, y_center = track.x_center, track.y_center
		x_outer, y_outer = track.x_outer, track.y_outer

		def gen_traj(x_all, idx, sim):
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

		fig = plt.figure()
		ax = plt.gca()
		ax.axis('equal')
		plt.plot(x_center, y_center, '--k', lw=0.5, alpha=0.5)
		plt.plot(x_outer, y_outer, 'k', lw=0.5, alpha=0.5)
		plt.plot(x_inner, y_inner, 'k', lw=0.5, alpha=0.5)

		# best trajectory (assuming qNEI is the best)
		sim, pidx = np.unravel_index(np.argmin(train_y_all_nei), train_y_all_nei.shape)
		wx_nei, wy_nei, x_nei, y_nei = gen_traj(train_x_all_nei, pidx, sim)
		plt.plot(wx_nei[:-1], wy_nei[:-1], linestyle='', marker='D', ms=5)
		laptime, speed, inputs = calcMinimumTimeSpeedInputs(x_nei, y_nei, **params)
		x = np.array(x_nei)
		y = np.array(y_nei)
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		norm = plt.Normalize(speed.min(), speed.max())
		lc = LineCollection(segments, cmap='viridis', norm=norm)
		lc.set_array(speed)
		lc.set_linewidth(2)
		line = ax.add_collection(lc)
		fig.colorbar(line, ax=ax, label="Speed [m/s]")
		ax.set_xlabel('x [m]')
		ax.set_ylabel('y [m]')

		if SAVE_RESULTS:
			filepath = 'results/{}_bestlap-{}.png'.format(TRACK_NAME, savestr)  # path for saving result
			np.savez('results/{}_optimalxy-{}.npz'.format(TRACK_NAME, savestr), x=x, y=y)
			np.savez('results/{}_raceline-{}.npz'.format(TRACK_NAME, savestr), x=x, y=y, time=laptime, speed=speed,
			         inputs=inputs)
			plt.savefig(filepath, dpi=600, bbox_inches='tight')


if __name__ == '__main__':
	start = time.time()
	optimize()
	dur = time.time() - start
	print(f"\nIt took {dur}s to run using {device}.")
