"""	Rectangular track.
"""

__author__ = 'Danilo Martins'
__email__ = 'mrtdan014@myuct.ac.za'

import os
import sys
import numpy as np
from bayes_race.tracks import Track
from bayes_race.utils import Spline2D
import matplotlib.pyplot as plt


class Rectangular(Track):
	"""	Rectangular with any length, breadth, width and corner radius (r_corner)"""

	def __init__(self, length, breadth, width, r_corner=0.):
		self.r_corner = r_corner  # radius of the corner along the centreline
		self.length = length - 2 * r_corner  # length of the straights
		self.breadth = breadth - 2 * r_corner  # breadth of the straights
		self.width = width  # track width
		self.track_width = width

		n_samples = 200  # number of points on the track centre line

		self.x_center, self.y_center = self.make_rect(length, breadth, r_corner, n_samples)

		# calculating centreline, track length and theta_track
		# do not call super init, parametric is faster
		self._parametric()

		self.x_outer, self.y_outer = self.make_rect(
			length=length + width,
			breadth=breadth + width,
			r_corner=r_corner + width / 2 if r_corner > 0 else 0,
			n_samples=n_samples
			# only round the corner if a positive radius is provided
		)

		self.x_inner, self.y_inner = self.make_rect(
			length=length - width,
			breadth=breadth - width,
			r_corner=r_corner - width / 2 if r_corner >= width / 2 else 0,
			n_samples=n_samples
			# only round the corner if a positive radius is provided
		)

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
		s = np.linspace(0, 2 * (length + breadth) + 2 * np.pi * r_corner - 1e-2, n_waypoints)
		wx = np.empty([n_waypoints])
		wy = np.empty([n_waypoints])
		for ids, theta in enumerate(s):  # theta is the distance along the track centreline
			wx[ids], wy[ids] = self.param_to_xy(theta, **kwargs)  # getting the coordinates of the waypoints

		if return_waypoints_only:
			return wx, wy

		sp = Spline2D(wx, wy)
		s = np.arange(0, sp.s[-1], self.track_length / n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return np.asarray(x), np.asarray(y)  # the coordinates of each point in the spline

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
		if "r_corner" not in kwargs:
			r_corner = self.r_corner
		else:
			r_corner = kwargs["r_corner"]

		if r_corner == 0:
			theta = theta % (2 * (length + breadth))
			if theta <= length / 2:
				x = theta
				y = -breadth / 2
			elif theta > length / 2 and theta <= length / 2 + breadth:
				x = length / 2
				y = -breadth / 2 + (theta - length / 2)
			elif theta > length / 2 + breadth and theta <= 3 / 2 * length + breadth:
				x = length / 2 - (theta - length / 2 - breadth)
				y = breadth / 2
			elif theta > 3 / 2 * length + breadth and theta <= 3 / 2 * length + 2 * breadth:
				x = -length / 2
				y = breadth / 2 - (theta - 3 / 2 * length - breadth)
			elif theta > 3 / 2 * length + 2 * breadth and theta <= 2 * length + 2 * breadth:
				x = -length / 2 + (theta - 3 / 2 * length - 2 * breadth)
				y = -breadth / 2
		else:
			x, y = self._param2xy(theta)
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

	@staticmethod
	def make_rect(length, breadth, r_corner, n_samples):
		"""
		Makes a rectangle length x breadth, with r_corner of radius.

		Args:
			length: length of the side of the rectangle
			breadth: breadth of the side of the rectangle
			r_corner: radius of the corners
			n_samples: total number of points in the line

		Returns:
			(float[], float[]): two arrays containing the x, y points of the rectangle
		"""

		length -= 2 * r_corner  # length of the straights
		breadth -= 2 * r_corner  # breadth of the straights

		# number of points in the corner
		if r_corner > 0:
			n_corner_pts = n_samples // 2
			# angles ranging from 0 to 90deg, each split into n_corner_pts//4 segments
			angles = np.linspace(0, np.pi / 2, n_corner_pts // 4)
			arc1 = np.array([r_corner * (1 - np.cos(angles)), breadth + r_corner * np.sin(angles)])
			arc2 = np.array([length + r_corner * (1 + np.sin(angles)), breadth + r_corner * np.cos(angles)])
			arc3 = np.array([length + r_corner * (1 + np.cos(angles)), - r_corner * np.sin(angles)])
			arc4 = np.array([r_corner * (1 - np.sin(angles)), - r_corner * np.cos(angles)])
		else:
			n_corner_pts = 0
			arc1 = arc2 = arc3 = arc4 = np.empty((2,0))

		n_straight_pts = (n_samples - n_corner_pts) // 4  # number of points on each of the straights
		length_arr = np.linspace(0, length, n_straight_pts)  # array containing the length split into equal segments
		breadth_arr = np.linspace(0, breadth, n_straight_pts)  # array containing the breadth split into equal segments

		seg1 = np.array([np.zeros(n_straight_pts), breadth_arr])
		seg2 = np.array([r_corner + length_arr, np.full(n_straight_pts, breadth + r_corner)])
		seg3 = np.array([np.zeros(n_straight_pts) + 2 * r_corner + length, breadth_arr[::-1]])
		seg4 = np.array([length_arr[::-1] + r_corner, np.full(n_straight_pts, -r_corner)])

		line = np.hstack((seg1, arc1, seg2, arc2, seg3, arc3, seg4, arc4))
		line[0] -= r_corner + length / 2  # shifting the track left
		line[1] -= breadth / 2  # shifting the track up
		x, y = line

		return x, y


# Example
if __name__ == "__main__":
	rec = Rectangular(6, 4, 0.8, 0.3)
	fig = rec.plot(plot_centre=True)
	plt.show()
