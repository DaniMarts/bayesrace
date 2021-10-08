import numpy as np
import matplotlib.pyplot as plt
import csv

# def generete_square_track():
w = 0.8  # m, track width
Length = 6  # m, the length of the track along the centreline
Breadth = 4  # m, the breath of the track along the centreline
r_i = 3*w/4  # m, inner radius
r_o = r_i + w  # outer radius
r_c = (r_i + r_o)/2  # radius of the centre line

breadth = Breadth - 2*r_c
length = Length - 2*r_c

theta = np.pi/2  # rad, the angle of the corner
r_r = r_i + (r_o - r_i)/(1-np.cos(theta/2))  # radius of the geometric line

pts = 20  # number of points in the straights
length_arr = np.linspace(0, length, pts)
breath_arr = np.linspace(0, breadth, pts)
ones = np.ones(pts)

thetas = np.linspace(0, np.pi/2, 10)

seg1 = np.array([np.zeros(pts), length_arr])
arc1 = np.array([r_c * (1 - np.cos(thetas)), length + r_c * np.sin(thetas)])
seg2 = np.array([r_c + breath_arr, (length + r_c) * ones])
arc2 = np.array([breadth + r_c * (1 + np.sin(thetas)), length + r_c * np.cos(thetas)])
seg3 = np.array([np.zeros(pts) + 2 * r_c + breadth, length_arr[::-1]])
arc3 = np.array([breadth + r_c * (1 + np.cos(thetas)), -r_c * np.sin(thetas)])
seg4 = np.array([breath_arr[::-1] + r_c, -r_c * ones])
arc4 = np.array([r_c * (1 - np.sin(thetas)), -r_c * np.cos(thetas)])

centreline = np.hstack((seg1, arc1, seg2, arc2, seg3, arc3, seg4, arc4))
centreline[1] += r_c  # shifting the track up

with open("rect_track.csv", 'w') as rect_track:
	writer = csv.writer(rect_track, delimiter=',', lineterminator="\n")
	writer.writerows(centreline.T)

fig, ax = plt.subplots()
ax.plot(*centreline)
ax.set_aspect("equal")
# plt.plot(*arc1, color="red")
plt.show()

