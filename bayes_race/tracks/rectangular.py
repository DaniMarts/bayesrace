"""	Rectangular track.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import os
import sys
import numpy as np
from bayes_race.tracks import Track
from bayes_race.utils import Spline2D
import matplotlib.pyplot as plt
from bayes_race.tracks.compute_io import ComputeIO

class Rectangular(Track):
	"""	Rectangular with any length, breadth, width and corner radius (r_corner)"""

	def __init__(self, length, breadth, width, r_corner=0):
		self.r_corner = r_corner  # radius of the corner along the centreline
		self.length = length - 2*r_corner  # length of the straights
		self.breadth = breadth - 2*r_corner  # breadth of the straights
		self.width = width  # track width
		self.track_width = width

		n_samples = 500

		if r_corner == 0:
			self.x_center, self.y_center = self._trajectory(
				n_waypoints=n_samples,
				n_samples=n_samples
				)
		else:
			# defining the centreline manually
			angles = np.linspace(0, np.pi / 2, n_samples//4)
			arc1 = np.array([r_corner * (1 - np.cos(angles)), length + r_corner * np.sin(angles)])
			arc2 = np.array([breadth + r_corner * (1 + np.sin(angles)), length + r_corner * np.cos(angles)])
			arc3 = np.array([breadth + r_corner * (1 + np.cos(angles)), - r_corner * np.sin(angles)])
			arc4 = np.array([r_corner * (1 - np.sin(angles)), - r_corner * np.cos(angles)])
			centreline = np.hstack((arc1, arc2, arc3, arc4))
			centreline[0] -= r_corner + breadth/2   # shifting the track left
			centreline[1] -= length/2  # shifting the track up
			self.x_center, self.y_center = centreline

		# calculating centreline, track length and theta_track
		# do not call super init, parametric is faster
		self._parametric()

		if r_corner == 0:
			self.x_outer, self.y_outer = self._trajectory(
				n_waypoints=n_samples,
				n_samples=n_samples,
				length=length+width/2,
				breadth=breadth+width/2,
				r_corner=r_corner + width/2 if r_corner > 0 else 0  # only round the corner if a positive radius is provided
				)
			self.x_inner, self.y_inner = self._trajectory(
				n_waypoints=n_samples,
				n_samples=n_samples,
				length=length-width/2,
				breadth=breadth-width/2,
				r_corner=r_corner - width/2 if r_corner >= width/2 else 0
				)
		else:
			self.x_inner, self.y_inner, self.x_outer, self.y_outer = ComputeIO(self)

		# self.load_raceline()
		#
		# self.psi_init = 0.
		# self.x_init = self.x_raceline[0]
		# self.y_init = self.y_raceline[0]
		# self.vx_init = 0.

	def _parametric(self):
		"""	calculate track length, center line and theta for center line
			alternate approach is to call __init__ in superclass Track
			but this is much faster since we mostly have straight lines
		"""
		if self.r_corner == 0:
			# this only applies to normal rectangles
			l, b = self.length, self.breadth
			self.track_length = 2 * l + 2 * b
			self.center_line = np.array([
				[0, l / 2, l / 2, -l / 2, -l / 2],
				[-b / 2, -b / 2, b / 2, b / 2, -b / 2]
			])
			self.theta_track = np.array([0, l / 2, l / 2 + b, l / 2 + l + b, l / 2 + l + 2 * b])
		else:
			super(Rectangular, self).__init__()


	def _trajectory(self, n_waypoints=25, n_samples=100, return_waypoints_only=True, **kwargs):
		""" center, inner and outer track lines
			n_waypoints	: no of points used to fit cubic splines
			n_samples 	: no of points finally sampled after fitting cubic splines
		"""
		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]
		if "r_corner" not in kwargs:
			r_corner = self.r_corner
		else:
			r_corner = kwargs["r_corner"]

		# arc length is 2*(breadth+length) + 4*(1/4*2*pi*r_corner)
		s = np.linspace(0, 2*(length+breadth)+2*np.pi*r_corner-1e-2, n_waypoints)
		print( 2*(length+breadth)+2*np.pi*r_corner)
		wx = np.empty([n_waypoints])
		wy = np.empty([n_waypoints])
		if r_corner == 0:
			for ids, theta in enumerate(s):  # theta is the distance along the track centreline
				wx[ids], wy[ids] = self.param_to_xy(theta, **kwargs)  # getting the coordinates of the waypoints
		else:
			for ids, theta in enumerate(s):
				wx[ids], wy[ids] = self._param2xy(theta)  # getting the coordinates of the waypoints

		if return_waypoints_only:
			return wx, wy

		sp = Spline2D(wx, wy)
		s = np.arange(0, sp.s[-1], self.track_length/n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return x, y  # the coordinates of each point in the spline

	def param_to_xy(self, theta, **kwargs):
		"""	convert distance along the track to x, y coordinates
			alternate is to call self._param2xy(theta)
			this is much faster since we mostly have straight lines
		"""
		if "length" not in kwargs:
			length = self.length
		else:
			length = kwargs["length"]
		if "breadth" not in kwargs:
			breadth = self.breadth
		else:
			breadth = kwargs["breadth"]

		theta = theta%(2*(length+breadth))
		if theta<=length/2:
			x = theta
			y = -breadth/2
		elif theta>length/2 and theta<=length/2+breadth:
			x = length/2
			y = -breadth/2 + (theta - length/2)
		elif theta>length/2+breadth and theta<=3/2*length+breadth:
			x = length/2 - (theta - length/2 - breadth)
			y = breadth/2
		elif theta>3/2*length+breadth and theta<=3/2*length+2*breadth:
			x = -length/2
			y = breadth/2 - (theta - 3/2*length - breadth)
		elif theta>3/2*length+2*breadth and theta<=2*length+2*breadth:
			x = -length/2 + (theta - 3/2*length - 2*breadth)
			y = -breadth/2
		return x, y

	def xy_to_param(self, x, y):
		"""	convert x, y coordinates to distance along the track
		"""
		theta = self._xy2param(x, y)
		return theta

	def load_raceline(self):
		"""	load raceline stored in npz file with keys 'x' and 'y'
		TODO: Add raceline.npz to this folder
		"""
		file_name = 'rectangular_raceline.npz'
		file_path = os.path.join(os.path.dirname(__file__), 'src', file_name)
		raceline = np.load(file_path)
		n_samples = 500
		self._load_raceline(
			wx=raceline['x'],
			wy=raceline['y'],
			n_samples=n_samples
			)

	def plot(self, **kwargs):
		""" plot center, inner and outer track lines
		"""
		fig = self._plot(**kwargs)
		return fig